import torch
import lorann
import psutil
import numpy as np
import numpy.typing as npt
from torch import Tensor
from typing import List, Optional

torch.set_float32_matmul_precision("high")


def run_kmeans(
    train,
    n_clusters,
    distance=lorann.IP,
    balanced=True,
    max_balance_diff=32,
    penalty_factor=2.0,
    verbose=True,
):
    kmeans = lorann.KMeans(
        n_clusters=n_clusters,
        iters=10,
        distance=distance,
        balanced=balanced,
        max_balance_diff=max_balance_diff,
        penalty_factor=penalty_factor,
    )

    cluster_map = kmeans.train(train, verbose=verbose)
    return kmeans, kmeans.get_centroids(), cluster_map


def compute_V(
    X: torch.Tensor,
    rank: int,
    *,
    n_iter: int = 4,
) -> torch.Tensor:
    m, n = X.shape
    k_eff = min(n, rank)
    _, _, V = torch.svd_lowrank(X, q=k_eff, niter=n_iter)
    V_out = X.new_zeros(n, rank)
    V_out[:, :k_eff] = V
    return V_out


class LorannIndex:

    def __init__(
        self,
        data: npt.NDArray[np.float32] | Tensor,
        n_clusters: int,
        global_dim: int,
        rank: int = 24,
        train_size: int = 5,
        distance: int = lorann.IP,
        penalty_factor: float = 2.0,
        *,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initializes a LorannIndex object. The initializer does not build the index.

        Args:
            data: Index points as an $m \\times d$ numpy array or PyTorch tensor.
            n_clusters: Number of clusters. In general, for $m$ index points, a good starting point
                is to set n_clusters as around $\\sqrt{m}$.
            global_dim: Globally reduced dimension ($s$). Must be either None or an integer.
                Higher values increase recall but also increase the query latency.
                In general, a good starting point is to set global_dim = None if $d < 200$,
                global_dim = 128 if $200 \\leq d \\leq 1000$, and global_dim = 256 if $d > 1000$.
            rank: Rank ($r$) of the parameter matrices. Defaults to 24.
            train_size: Number of nearby clusters ($w$) used for training the reduced-rank
                regression models. Defaults to 5, but lower values can be used if
                $m \\gtrsim 500 000$ to speed up the index construction.
            distance: The distance measure to use. Either IP or L2. Defaults to IP.
            penalty_factor: Penalty factor for balanced clustering. Higher values can be used for
                faster clustering at the cost of clustering quality. Used only if balanced = True.
                Defaults to 2.0.
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
        self.n_clusters = n_clusters
        self.rank = rank
        self.train_size = train_size
        self.distance = distance
        self.penalty_factor = penalty_factor

        _, dim = data_t.shape

        if global_dim is None or global_dim <= 0 or global_dim > dim:
            global_dim = dim

        self.global_dim = global_dim
        self.dim = dim
        self.built = False

    def build(
        self, verbose=False, *, training_queries: Optional[npt.NDArray[np.float32] | Tensor] = None
    ):
        """
        Builds the index.

        Args:
            verbose: Whether to use verbose output for index construction. Defaults to False.
            training_queries: An optional matrix of training queries used to build the index. Can be
                useful in the out-of-distribution setting where the training and query distributions
                differ. Ideally there should be at least as many training query points as there are
                index points.

        Returns:
            None
        """
        if training_queries is None:
            training_queries = self.data
        else:
            if isinstance(training_queries, np.ndarray):
                training_queries = torch.as_tensor(
                    training_queries, dtype=self.dtype, device=self.device
                )
            else:
                training_queries = training_queries.to(self.device, dtype=self.dtype)

        if self.global_dim < self.dim:
            rand_n = min(len(training_queries), 32768)
            rand_idx = torch.randperm(len(training_queries))[:rand_n]
            query_sample = training_queries[rand_idx].to(torch.float32)

            Y_global: Tensor = query_sample.T @ query_sample
            _, vecs = torch.linalg.eigh(Y_global.to(torch.float32))
            self.global_transform: Tensor = vecs[:, -self.global_dim :].contiguous().to(self.dtype)

            reduced_train = self.data @ self.global_transform
            reduced_train_cpu = reduced_train.to(torch.float32).cpu().numpy()

            if training_queries is not self.data:
                reduced_query = training_queries @ self.global_transform
                reduced_query_cpu = reduced_query.to(torch.float32).cpu().numpy()
            else:
                reduced_query_cpu = reduced_train_cpu

            kmeans, centroids, self.cluster_map = run_kmeans(
                reduced_train_cpu,
                self.n_clusters,
                self.distance,
                balanced=True,
                penalty_factor=self.penalty_factor,
                verbose=verbose,
            )
            cluster_train_map = kmeans.assign(reduced_query_cpu, self.train_size)
        else:
            data_cpu = self.data.to(torch.float32).cpu().numpy()
            if training_queries is not self.data:
                query_cpu = training_queries.to(torch.float32).cpu().numpy()
            else:
                query_cpu = data_cpu

            kmeans, centroids, self.cluster_map = run_kmeans(
                data_cpu,
                self.n_clusters,
                self.distance,
                balanced=True,
                penalty_factor=self.penalty_factor,
                verbose=verbose,
            )
            cluster_train_map = kmeans.assign(query_cpu, self.train_size)
            self.global_transform = None

        self.centroids = torch.as_tensor(centroids, dtype=self.dtype, device=self.device)
        self.max_cluster_size: int = max(len(c) for c in self.cluster_map)

        if self.distance == lorann.L2:
            self.global_centroid_norms: Tensor = (torch.linalg.norm(self.centroids, dim=1) ** 2).to(
                self.dtype
            )
            self.data_norms: Tensor = (torch.linalg.norm(self.data, dim=1) ** 2).to(self.dtype)
            self.cluster_norms: List[Tensor] = []
        else:
            self.global_centroid_norms = None
            self.data_norms = None
            self.cluster_norms = None

        self.A: list[torch.Tensor] = []
        self.B: list[torch.Tensor] = []

        for cid in range(self.n_clusters):
            m_i = len(self.cluster_map[cid])
            l_i = len(cluster_train_map[cid])

            pts = self.data[self.cluster_map[cid]]
            if l_i >= m_i:
                Q = training_queries[cluster_train_map[cid]]
            else:
                Q = pts

            if self.global_transform is not None:
                beta_hat = (pts @ self.global_transform).T
                y_hat = (Q @ self.global_transform) @ beta_hat
            else:
                beta_hat = pts.T
                y_hat = Q @ beta_hat

            V_i = compute_V(y_hat.to(torch.float32), self.rank).to(self.dtype)
            A_i = beta_hat @ V_i

            B_pad = torch.zeros(
                self.rank, self.max_cluster_size, dtype=self.dtype, device=self.device
            )
            B_pad[:, :m_i] = V_i.T

            self.A.append(A_i)
            self.B.append(B_pad)

            sz: int = len(self.cluster_map[cid])
            pad_cols: int = self.max_cluster_size - sz

            if self.cluster_norms is not None:
                self.cluster_norms.append(
                    torch.cat(
                        (
                            self.data_norms[self.cluster_map[cid]],
                            torch.zeros(
                                pad_cols,
                                dtype=self.data_norms.dtype,
                                device=self.data_norms.device,
                            ),
                        )
                    )
                )

            self.cluster_map[cid] = np.concatenate(
                [self.cluster_map[cid], np.zeros(pad_cols, dtype=np.int32)]
            )

        self.A = torch.stack(self.A)
        self.B = torch.stack(self.B)
        self.cluster_map = torch.tensor(
            np.array(self.cluster_map, dtype=np.int32), device=self.device
        )
        if self.cluster_norms is not None:
            self.cluster_norms = torch.stack(self.cluster_norms)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        self._search_impl = torch.compile(
            self._search_impl, backend="inductor", mode="default", dynamic=True
        )
        self.built = True

    def _search_impl(self, q, k, clusters_to_search, points_to_rerank):
        if self.distance == lorann.L2:
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
        if self.distance == lorann.L2:
            d += self.global_centroid_norms

        _, I = torch.topk(d, clusters_to_search, largest=False)

        transformed_query = transformed_query.reshape((batch_size, 1, 1, -1))
        approx_dists = (transformed_query @ self.A[I] @ self.B[I]).reshape(estimate_size)
        if self.distance == lorann.L2:
            approx_dists += self.cluster_norms[I].reshape(estimate_size)
        idx = self.cluster_map[I].reshape(estimate_size)

        n_selected = min(max(k, points_to_rerank), clusters_to_search * self.max_cluster_size)
        _, idx_cs = torch.topk(approx_dists, n_selected, largest=False)
        cs = torch.gather(idx, 1, idx_cs)

        if points_to_rerank == 0:
            return cs

        final_dists = (self.data[cs] @ q[:, :, None]).reshape((batch_size, n_selected))
        if self.distance == lorann.L2:
            final_dists += self.data_norms[cs]

        _, idx_final = torch.topk(final_dists, min(k, n_selected), largest=False)
        return torch.gather(cs, 1, idx_final)

    def search(
        self,
        q: np.ndarray | Tensor,
        k: int,
        clusters_to_search: int,
        points_to_rerank: int,
        batch_size: str | int = "auto",
    ):
        """
        Performs approximate nearest neighbor queries for multiple query vectors.

        The queries are given as a numpy matrix or torch tensor where each row contains a query.

        Args:
            q: The query object.
            k: The number of nearest neighbors to be returned.
            clusters_to_search: Number of clusters to search.
            points_to_rerank: Number of points to re-rank using exact search. If points_to_rerank is
                set to 0, no re-ranking is performed.
            batch_size: Number of queries to perform at once. Lower if running out of memory.
                Defaults to "auto".

        Returns:
            A Long tensor of indices of the approximate nearest neighbors.
        """
        if not self.built:
            raise RuntimeError("cannot query an index that has not been built")

        num_queries = q.shape[0]

        if batch_size == "auto":
            A_memory = clusters_to_search * self.global_dim * self.rank
            B_memory = clusters_to_search * self.rank * self.max_cluster_size
            data_memory = points_to_rerank * q.shape[1]
            memory_per_query = self.dtype.itemsize * (A_memory + B_memory + data_memory)

            if self.device.type == "cuda":
                available_memory = torch.cuda.mem_get_info(self.device.index)[0]
                batch_size = int(available_memory / memory_per_query / 8)
            elif self.device.type == "cpu":
                available_memory = psutil.virtual_memory().available
                batch_size = int(available_memory * 0.9 / memory_per_query)
            else:
                raise RuntimeError("cannot use batch_size=auto for device", self.device)

            batch_size = max(1, batch_size)

        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q, dtype=self.dtype, device=self.device)
        else:
            q = q.to(dtype=self.dtype, device=self.device)

        results = torch.empty((num_queries, k), dtype=torch.long, device=self.device)

        for i in range(0, num_queries, batch_size):
            end_idx = min(i + batch_size, num_queries)
            results[i:end_idx] = self._search_impl(
                q[i:end_idx], k, clusters_to_search, points_to_rerank
            )

        return results

    def _exact_search_impl(self, q, k):
        if self.distance == lorann.L2:
            q = -2 * q
        else:
            q = -q

        batch_size = q.shape[0]
        dists = (self.data @ q[:, :, None]).reshape((batch_size, len(self.data)))
        if self.distance == lorann.L2:
            dists += self.data_norms

        _, idx = torch.topk(dists, min(k, len(self.data)), largest=False)
        return idx

    def exact_search(
        self,
        q: np.ndarray | Tensor,
        k: int,
        batch_size: str | int = "auto",
    ):
        """
        Performs exact nearest neighbor queries for multiple query vectors.

        The queries are given as a numpy matrix or torch tensor where each row contains a query.

        Args:
            q: The query object.
            k: The number of nearest neighbors to be returned.
            batch_size: Number of queries to perform at once. Lower if running out of memory.
                Defaults to "auto".

        Returns:
            A Long tensor of indices of the exact nearest neighbors.
        """
        num_queries = q.shape[0]

        if batch_size == "auto":
            memory_per_query = self.dtype.itemsize * (len(self.data) * num_queries)

            if self.device.type == "cuda":
                available_memory = torch.cuda.mem_get_info(self.device.index)[0]
                batch_size = int(available_memory / memory_per_query / 8)
            elif self.device.type == "cpu":
                available_memory = psutil.virtual_memory().available
                batch_size = int(available_memory * 0.9 / memory_per_query)
            else:
                raise RuntimeError("cannot use batch_size=auto for device", self.device)

            batch_size = max(1, batch_size)

        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q, dtype=self.dtype, device=self.device)
        else:
            q = q.to(dtype=self.dtype, device=self.device)

        results = torch.empty((num_queries, k), dtype=torch.long, device=self.device)

        for i in range(0, num_queries, batch_size):
            end_idx = min(i + batch_size, num_queries)
            results[i:end_idx] = self._exact_search_impl(q[i:end_idx], k)

        return results

    def save(self, fname: str) -> None:
        """
        Saves the index to a file on the disk.

        Args:
            fname: The filename to save the index to.

        Raises:
            RuntimeError: If trying to save before building the index.
            OSError: If saving to the specified file fails.

        Returns:
            None
        """
        if not self.built:
            raise RuntimeError("Cannot save an index that has not been built")

        state_dict = {
            "data": self.data,
            "n_clusters": self.n_clusters,
            "rank": self.rank,
            "train_size": self.train_size,
            "distance": self.distance,
            "penalty_factor": self.penalty_factor,
            "global_dim": self.global_dim,
            "dim": self.dim,
            "built": self.built,
            "dtype": self.dtype,
            "centroids": self.centroids,
            "max_cluster_size": self.max_cluster_size,
            "cluster_map": self.cluster_map,
            "A": self.A,
            "B": self.B,
        }

        if hasattr(self, "global_transform") and self.global_transform is not None:
            state_dict["global_transform"] = self.global_transform

        if self.distance == lorann.L2:
            state_dict["global_centroid_norms"] = self.global_centroid_norms
            state_dict["data_norms"] = self.data_norms
            state_dict["cluster_norms"] = self.cluster_norms

        try:
            torch.save(state_dict, fname)
        except Exception as e:
            raise OSError(f"Failed to save index to {fname}: {str(e)}")

    @classmethod
    def load(cls, fname: str, device: Optional[torch.device | str] = None):
        """
        Loads a LorannIndex from a file on the disk.

        Args:
            fname: The filename to load the index from.
            device: The device to load the index to.

        Raises:
            OSError: If loading from the specified file fails.

        Returns:
            The loaded LorannIndex object.
        """
        try:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(device)

            state_dict = torch.load(fname, map_location=device)
        except Exception as e:
            raise OSError(f"Failed to load index from {fname}: {str(e)}")

        data = state_dict["data"]
        n_clusters = state_dict["n_clusters"]
        global_dim = state_dict["global_dim"]
        rank = state_dict["rank"]
        train_size = state_dict["train_size"]
        distance = state_dict["distance"]
        penalty_factor = state_dict["penalty_factor"]
        dtype = state_dict["dtype"]

        instance = cls(
            data=data,
            n_clusters=n_clusters,
            global_dim=global_dim,
            rank=rank,
            train_size=train_size,
            distance=distance,
            penalty_factor=penalty_factor,
            device=device,
            dtype=dtype,
        )

        instance.built = state_dict["built"]
        instance.dim = state_dict["dim"]
        instance.centroids = state_dict["centroids"]
        instance.max_cluster_size = state_dict["max_cluster_size"]
        instance.cluster_map = state_dict["cluster_map"]
        instance.A = state_dict["A"]
        instance.B = state_dict["B"]

        if "global_transform" in state_dict:
            instance.global_transform = state_dict["global_transform"]
        else:
            instance.global_transform = None

        if instance.distance == lorann.L2:
            instance.global_centroid_norms = state_dict["global_centroid_norms"]
            instance.data_norms = state_dict["data_norms"]
            instance.cluster_norms = state_dict["cluster_norms"]
        else:
            instance.global_centroid_norms = None
            instance.data_norms = None
            instance.cluster_norms = None

        if instance.built:
            instance._search_impl = torch.compile(
                instance._search_impl, backend="inductor", mode="default", dynamic=True
            )

        return instance
