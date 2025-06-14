import torch
import lorann
import numpy as np
from torch import Tensor
from typing import List, Optional, Sequence

torch.set_float32_matmul_precision("high")


def run_kmeans(
    train,
    n_clusters,
    euclidean=False,
    balanced=True,
    max_balance_diff=16,
    penalty_factor=2.0,
    verbose=True,
):
    kmeans = lorann.KMeans(
        n_clusters=n_clusters,
        iters=10,
        euclidean=euclidean,
        balanced=balanced,
        max_balance_diff=max_balance_diff,
        penalty_factor=penalty_factor,
        verbose=verbose,
    )

    cluster_map = kmeans.train(train)
    return kmeans, kmeans.get_centroids(), cluster_map


def compute_V(
    X: torch.Tensor,
    rank: int,
    *,
    n_iter: int = 4,
) -> torch.Tensor:
    *batch_dims, m, n = X.shape
    k_eff = min(n, rank)

    _, _, Vh = torch.svd_lowrank(X, q=k_eff, niter=n_iter)
    V_computed = Vh[..., :k_eff]

    V_out = X.new_zeros(*batch_dims, n, rank)
    V_out[..., :n, :k_eff] = V_computed

    return V_out


def compute_estimates(q: torch.Tensor, pts: torch.Tensor, transform: torch.Tensor):
    beta_hat = (pts @ transform).transpose(1, 2)
    q_tr = q @ transform
    y_hat = torch.bmm(q_tr, beta_hat)
    return beta_hat, y_hat


class LorannIndex:

    def __init__(
        self,
        data: np.ndarray | Tensor,
        n_clusters: int,
        global_dim: int,
        rank: int = 24,
        train_size: int = 5,
        euclidean: bool = False,
        batch_size: int = 1,
        max_balance_diff: int = 16,
        penalty_factor: float = 2.0,
        verbose: bool = False,
        *,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initializes a LorannIndex object and builds the index.

        Args:
            data: Index points as an $m \\times d$ numpy array or PyTorch tensor.
            n_clusters: Number of clusters. In general, for $m$ index points, a good starting point
                is to set n_clusters as around $\\sqrt{m}$.
            global_dim: Globally reduced dimension ($s$). Must be either None or an integer that is
                a multiple of 32. Higher values increase recall but also increase the query latency.
                In general, a good starting point is to set global_dim = None if $d < 200$,
                global_dim = 128 if $200 \\leq d \\leq 1000$, and global_dim = 256 if $d > 1000$.
            rank: Rank ($r$) of the parameter matrices. Defaults to 24.
            train_size: Number of nearby clusters ($w$) used for training the reduced-rank
                regression models. Defaults to 5, but lower values can be used if
                $m \\gtrsim 500 000$ to speed up the index construction.
            euclidean: Whether to use Euclidean distance instead of (negative) inner product as the
                dissimilarity measure. Defaults to False.
            balanced: Whether to use balanced clustering. Defaults to False.
            penalty_factor: Penalty factor for balanced clustering. Higher values can be used for
                faster clustering at the cost of clustering quality. Used only if balanced = True.
                Defaults to 2.0.
            verbose: Whether to use verbose output for index construction. Defaults to False.
            device: The device used for building and storing the index.
            dtype: The dtype for the index structures. Defaults to torch.float32.

        Returns:
            None
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        if isinstance(data, np.ndarray):
            data_t = torch.as_tensor(data, dtype=dtype, device=self.device)
        else:
            data_t = data.to(self.device, dtype=dtype)
        self.data: Tensor = data_t
        self.euclidean = euclidean
        _, dim = data_t.shape

        if global_dim == -1 or global_dim is None:
            global_dim = dim

        if global_dim < dim:
            Y_global: Tensor = data_t.T @ data_t
            _, vecs = torch.linalg.eigh(Y_global.to(torch.float32))
            self.global_transform: Tensor = vecs[:, -global_dim:].to(dtype)
            reduced_train = data_t @ self.global_transform
            reduced_train_np = reduced_train.to(torch.float32).cpu().numpy()
            kmeans, centroids, self.cluster_map = run_kmeans(
                reduced_train_np,
                n_clusters,
                euclidean,
                balanced=True,
                max_balance_diff=max_balance_diff,
                penalty_factor=penalty_factor,
                verbose=verbose,
            )
            centroid_train_map: Sequence[np.ndarray] = kmeans.assign(reduced_train_np, train_size)
        else:
            if isinstance(data, np.ndarray):
                kmeans, centroids, self.cluster_map = run_kmeans(
                    data,
                    n_clusters,
                    euclidean,
                    balanced=True,
                    max_balance_diff=max_balance_diff,
                    penalty_factor=penalty_factor,
                    verbose=verbose,
                )
                centroid_train_map = kmeans.assign(data, train_size)
            else:
                kmeans, centroids, self.cluster_map = run_kmeans(
                    data_t.cpu().numpy(),
                    n_clusters,
                    euclidean,
                    balanced=True,
                    max_balance_diff=max_balance_diff,
                    penalty_factor=penalty_factor,
                    verbose=verbose,
                )
                centroid_train_map = kmeans.assign(data_t.cpu().numpy(), train_size)
            self.global_transform = None

        self.centroids = torch.as_tensor(centroids, dtype=dtype, device=self.device)
        self.max_cluster_size: int = max(len(c) for c in self.cluster_map)

        if euclidean:
            self.global_centroid_norms: Tensor = (torch.linalg.norm(self.centroids, dim=1) ** 2).to(
                dtype
            )
            self.data_norms: Tensor = (torch.linalg.norm(data_t, dim=1) ** 2).to(dtype)
            self.cluster_norms: List[Tensor] = []
        else:
            self.global_centroid_norms = None
            self.data_norms = None
            self.cluster_norms = None

        self.A: list[torch.Tensor] = []
        self.B: list[torch.Tensor] = []

        for start in range(0, n_clusters, batch_size):
            batch_ids = list(range(start, min(start + batch_size, n_clusters)))

            m_sizes = [len(self.cluster_map[i]) for i in batch_ids]
            l_sizes = [len(centroid_train_map[i]) for i in batch_ids]
            m_max, l_max = max(m_sizes), max(l_sizes)
            B_sz = len(m_sizes)

            pts_batched = torch.zeros(B_sz, m_max, dim, dtype=dtype, device=self.device)
            q_batched = torch.zeros(B_sz, l_max, dim, dtype=dtype, device=self.device)

            for j, cid in enumerate(batch_ids):
                m_i, l_i = m_sizes[j], l_sizes[j]

                pts_batched[j, :m_i] = data_t[self.cluster_map[cid]]
                q_batched[j, :l_i] = data_t[centroid_train_map[cid]]

            if global_dim < dim:
                beta_hat, y_hat = compute_estimates(q_batched, pts_batched, self.global_transform)
            else:
                beta_hat = pts_batched.transpose(1, 2)
                y_hat = torch.bmm(q_batched, beta_hat)

            V = compute_V(y_hat.to(torch.float32), rank).to(dtype)
            A_mb = torch.bmm(beta_hat, V)

            for j, cid in enumerate(batch_ids):
                m_i = m_sizes[j]

                self.A.append(A_mb[j, :, :])

                B_pad = torch.zeros(rank, self.max_cluster_size, dtype=dtype, device=self.device)
                B_pad[:, :m_i] = V[j, :m_i].T
                self.B.append(B_pad)

        for i in range(n_clusters):
            sz: int = len(self.cluster_map[i])
            pad_cols: int = self.max_cluster_size - sz

            if self.cluster_norms is not None:
                self.cluster_norms.append(
                    torch.cat(
                        (
                            self.data_norms[self.cluster_map[i]],
                            torch.zeros(
                                self.max_cluster_size - sz,
                                dtype=self.data_norms.dtype,
                                device=self.data_norms.device,
                            ),
                        )
                    )
                )

            self.cluster_map[i] = np.concatenate(
                [self.cluster_map[i], np.zeros(pad_cols, dtype=np.int32)]
            )

        self.A = torch.stack(self.A)
        self.B = torch.stack(self.B)
        self.cluster_map = torch.tensor(
            np.array(self.cluster_map, dtype=np.int32), device=self.device
        )
        if self.cluster_norms is not None:
            self.cluster_norms = torch.stack(self.cluster_norms)

    # @torch.compile(fullgraph=True)
    def _search_impl(self, q, k, clusters_to_search, points_to_rerank):
        if self.euclidean:
            q = -2 * q
        else:
            q = -q

        batch_size = q.shape[0]
        clusters_to_search = min(clusters_to_search, self.centroids.shape[0])
        estimate_size = (batch_size, clusters_to_search * self.max_cluster_size)

        if self.global_transform is not None:
            transformed_query = q @ self.global_transform
        else:
            transformed_query = q

        d = transformed_query @ self.centroids.T
        if self.euclidean:
            d += self.global_centroid_norms

        _, I = torch.topk(d, clusters_to_search, largest=False)

        transformed_query = transformed_query.reshape((batch_size, 1, 1, -1))
        res = (transformed_query @ self.A[I] @ self.B[I]).reshape(estimate_size)
        if self.euclidean:
            res += self.cluster_norms[I].reshape(estimate_size)
        idx = self.cluster_map[I].reshape(estimate_size)

        if points_to_rerank >= clusters_to_search * self.max_cluster_size:
            cs = idx
            points_to_rerank = clusters_to_search * self.max_cluster_size
        else:
            _, idx_cs = torch.topk(res, points_to_rerank, largest=False)
            cs = torch.gather(idx, 1, idx_cs)

        if points_to_rerank <= k:
            return cs

        final_dists = (self.data[cs] @ q[:, :, None]).reshape((batch_size, points_to_rerank))
        if self.euclidean:
            final_dists += self.data_norms[cs]

        k = min(k, final_dists.shape[1])
        _, idx_final = torch.topk(final_dists, k, largest=False)
        return torch.gather(cs, 1, idx_final)

    def search(self, q, k, clusters_to_search, points_to_rerank):
        q = torch.tensor(q, dtype=self.dtype, device=self.device)
        return self._search_impl(q, k, clusters_to_search, points_to_rerank)
