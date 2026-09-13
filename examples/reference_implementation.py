import argparse
import shutil
import time
from pathlib import Path
from urllib.request import Request, urlopen

import h5py
import lorann
import numpy as np
from sklearn.utils.extmath import randomized_svd


KMEANS_ITERATIONS = 10
SAMPLED_POINTS_PER_CLUSTER = 256
GLOBAL_DIM_REDUCTION_SAMPLES = 16384
RSVD_OVERSAMPLES = 5
RSVD_N_ITER = 4


class Lorann:

    def __init__(
        self,
        data,
        n_clusters,
        global_dim=None,
        rank=32,
        train_size=5,
        euclidean=False,
        approximate=True,
        random_state=0,
        verbose=False,
    ):
        data = np.ascontiguousarray(data, dtype=np.float32)
        if data.ndim != 2 or min(data.shape) == 0:
            raise ValueError("data must be a nonempty matrix")
        n_samples, dim = data.shape
        if not 1 <= n_clusters <= n_samples:
            raise ValueError("n_clusters must be between 1 and the number of points")
        if not 1 <= train_size <= n_clusters:
            raise ValueError("train_size must be between 1 and n_clusters")
        if rank < 1:
            raise ValueError("rank must be positive")
        if global_dim is not None and not 1 <= global_dim <= dim:
            raise ValueError("global_dim must be None or between 1 and the data dimension")

        self.data = data
        self.euclidean = euclidean
        self.global_transform = None
        if verbose:
            print(
                f"Index settings: clusters={n_clusters}, global_dim={global_dim}, rank={rank}, "
                f"train_size={train_size}, distance={'L2' if euclidean else 'IP'}, "
                f"approximate={approximate}",
                flush=True,
            )
        if global_dim is not None and global_dim < dim:
            sample_size = min(n_samples, GLOBAL_DIM_REDUCTION_SAMPLES)
            if verbose:
                print(
                    f"Reducing dimension from {dim} to {global_dim} "
                    f"using {sample_size:,} sampled vectors...",
                    flush=True,
                )
            sample = data
            if sample_size < n_samples:
                rng = np.random.default_rng(random_state)
                sample = data[rng.choice(n_samples, sample_size, replace=False)]
            _, vectors = np.linalg.eigh(sample.T @ sample)
            self.global_transform = vectors[:, -global_dim:]
            reduced_data = data @ self.global_transform
        else:
            reduced_data = data

        samples_per_cluster = SAMPLED_POINTS_PER_CLUSTER
        if not approximate or samples_per_cluster * n_clusters > 0.5 * n_samples:
            samples_per_cluster = -1
        if verbose:
            clustering_samples = (
                n_samples if samples_per_cluster < 0 else samples_per_cluster * n_clusters
            )
            print(
                f"Training {n_clusters:,} clusters ({KMEANS_ITERATIONS} k-means iterations, "
                f"{clustering_samples:,} training vectors)...",
                flush=True,
            )
        start = time.perf_counter()
        kmeans = lorann.KMeans(
            n_clusters=n_clusters,
            iters=KMEANS_ITERATIONS,
            samples_per_cluster=samples_per_cluster,
            distance=lorann.L2 if euclidean else lorann.IP,
            balanced=False,
        )
        self.cluster_map = kmeans.train(reduced_data, verbose=verbose)
        self.centroids = kmeans.get_centroids()
        if verbose:
            sizes = [len(members) for members in self.cluster_map]
            print(
                f"Clustering finished in {time.perf_counter() - start:.2f} s. "
                f"Cluster sizes: min={min(sizes):,}, mean={np.mean(sizes):.1f}, "
                f"max={max(sizes):,}",
                flush=True,
            )
            print(
                f"Assigning training points to their {train_size} nearest clusters...", flush=True
            )
        start = time.perf_counter()
        training_map = kmeans.assign(reduced_data, train_size)
        if verbose:
            print(
                f"Training assignments finished in {time.perf_counter() - start:.2f} s.", flush=True
            )
            print("Building reduced-rank regression models...", flush=True)

        self.centroid_norms = np.einsum("ij,ij->i", self.centroids, self.centroids)
        self.data_norms = np.einsum("ij,ij->i", data, data)
        self.A, self.B = [], []
        start = time.perf_counter()
        progress_step = max(1, n_clusters // 10)
        for cluster, members in enumerate(self.cluster_map):
            if verbose and cluster % progress_step == 0:
                print(
                    f"  Models built: {cluster:,}/{n_clusters:,} "
                    f"({100 * cluster / n_clusters:.0f}%), "
                    f"elapsed: {time.perf_counter() - start:.2f} s",
                    flush=True,
                )
            if len(members) == 0:
                # Keep model indices aligned with cluster IDs, including empty clusters.
                self.A.append(np.empty((reduced_data.shape[1], 0), dtype=np.float32))
                self.B.append(np.empty((0, 0), dtype=np.float32))
                continue

            points = data[members]
            queries = data[training_map[cluster]]
            if len(queries) < len(points):
                queries = points
            if self.global_transform is not None:
                X = queries @ self.global_transform
                if approximate:
                    beta = (points @ self.global_transform).T
                else:
                    Y = queries @ points.T
                    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
                Y_hat = X @ beta
            else:
                beta = points.T
                Y_hat = queries @ beta

            # V contains the leading right singular vectors as rows. The model
            # q @ A @ B approximates the inner products q @ points.T.
            effective_rank = min(rank, *Y_hat.shape)
            if approximate:
                _, _, V = randomized_svd(
                    Y_hat,
                    n_components=effective_rank,
                    n_oversamples=RSVD_OVERSAMPLES,
                    n_iter=RSVD_N_ITER,
                    # Normalize power iterations to avoid overflow on unnormalized SIFT.
                    power_iteration_normalizer="LU",
                    random_state=random_state,
                )
            else:
                _, _, V = np.linalg.svd(Y_hat, full_matrices=False)
                V = V[:effective_rank]
            self.A.append(beta @ V.T)
            self.B.append(V)
        if verbose:
            print(
                f"  Models built: {n_clusters:,}/{n_clusters:,} (100%), "
                f"elapsed: {time.perf_counter() - start:.2f} s",
                flush=True,
            )

    def search(self, q, k, clusters_to_search, points_to_rerank):
        """Return up to k point IDs, ordered by increasing distance.

        points_to_rerank=0 uses approximate scores only. Otherwise, exactly score
        at least k candidates, or all visited points if fewer than k are available.
        """
        q = np.asarray(q, dtype=np.float32)
        if q.shape != (self.data.shape[1],):
            raise ValueError("q must be a single vector with the data dimension")
        if k < 1 or clusters_to_search < 1 or points_to_rerank < 0:
            raise ValueError(
                "k and clusters_to_search must be positive; reranking cannot be negative"
            )

        # ||q||^2 is constant across candidates, so it can be omitted when ranking.
        scaled_query = (-2 if self.euclidean else -1) * q
        reduced_query = scaled_query
        if self.global_transform is not None:
            reduced_query = scaled_query @ self.global_transform

        cluster_scores = reduced_query @ self.centroids.T
        if self.euclidean:
            cluster_scores += self.centroid_norms
        clusters = np.argsort(cluster_scores)[:clusters_to_search]

        scores, point_ids = [], []
        for cluster in clusters:
            members = self.cluster_map[cluster]
            estimates = reduced_query @ self.A[cluster] @ self.B[cluster]
            if self.euclidean:
                estimates += self.data_norms[members]
            scores.append(estimates)
            point_ids.append(members)
        scores = np.concatenate(scores)
        point_ids = np.concatenate(point_ids)

        count = max(k, points_to_rerank)
        candidates = point_ids[np.argsort(scores)[:count]]
        if points_to_rerank == 0:
            return candidates[:k]

        exact_scores = self.data[candidates] @ scaled_query
        if self.euclidean:
            exact_scores += self.data_norms[candidates]
        return candidates[np.argsort(exact_scores)[:k]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("sift-128-euclidean.hdf5"),
        help="path to the SIFT HDF5 file (downloaded if missing)",
    )
    parser.add_argument("--queries", type=int, help="evaluate only the first N test queries")
    args = parser.parse_args()
    if args.queries is not None and args.queries < 1:
        parser.error("--queries must be positive")

    if not args.dataset.exists():
        url = "https://ann-benchmarks.com/sift-128-euclidean.hdf5"
        print(f"Downloading {url} -> {args.dataset}...", flush=True)
        args.dataset.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.dataset.with_suffix(".hdf5.part")
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            temporary.replace(args.dataset)
        finally:
            temporary.unlink(missing_ok=True)

    k = 100
    print(f"Loading SIFT from {args.dataset}...", flush=True)
    start = time.perf_counter()
    with h5py.File(args.dataset, "r") as dataset:
        train = dataset["train"][:]
        test = dataset["test"][: args.queries]
        neighbors = dataset["neighbors"][: args.queries, :k]
    print(
        f"Loaded {len(train):,} training vectors and {len(test):,} queries "
        f"({train.shape[1]} dimensions, {train.dtype}) in {time.perf_counter() - start:.2f} s.",
        flush=True,
    )

    print(f"Building the index for {len(train):,} SIFT vectors...", flush=True)
    start = time.perf_counter()
    index = Lorann(train, n_clusters=1024, global_dim=None, euclidean=True, verbose=True)
    print(f"Build time (s): {time.perf_counter() - start:.2f}", flush=True)

    print(f"Querying the index with {len(test):,} vectors...", flush=True)
    clusters_to_search = 32
    points_to_rerank = 800
    print(
        f"Search settings: k={k}, clusters_to_search={clusters_to_search}, "
        f"points_to_rerank={points_to_rerank}",
        flush=True,
    )
    start = time.perf_counter()
    results = []
    progress_step = max(1, len(test) // 10)
    for count, q in enumerate(test, start=1):
        results.append(index.search(q, k, clusters_to_search, points_to_rerank))
        if count % progress_step == 0 or count == len(test):
            print(
                f"  Queries completed: {count:,}/{len(test):,} ({100 * count / len(test):.0f}%), "
                f"elapsed: {time.perf_counter() - start:.2f} s",
                flush=True,
            )
    elapsed = time.perf_counter() - start

    print("Computing recall against the ground-truth neighbors...", flush=True)
    recall = np.mean(
        [len(np.intersect1d(result, truth)) / k for result, truth in zip(results, neighbors)]
    )
    print(f"Recall@{k}: {recall:.4f}")
    print(f"Total query time (s): {elapsed:.2f}")
    print(f"Average query time (ms): {elapsed / len(test) * 1e3:.3f}")
    print(f"Queries per second: {len(test) / elapsed:.1f}")


if __name__ == "__main__":
    main()
