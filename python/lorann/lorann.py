import numpy as np
import numpy.typing as npt
import lorannlib
from typing import Optional, Tuple, Union, List

IP = lorannlib.IP
L2 = lorannlib.L2
HAMMING = lorannlib.HAMMING


def _check_data_matrix(data: np.ndarray) -> Tuple[int, int]:
    if isinstance(data, np.ndarray):
        if len(data) == 0 or len(data.shape) != 2:
            raise ValueError("The matrix should be non-empty and two-dimensional")
        if not data.flags["C_CONTIGUOUS"] or not data.flags["ALIGNED"]:
            raise ValueError("The matrix has to be C_CONTIGUOUS and ALIGNED")
        return data.shape
    else:
        raise ValueError("Data must be an ndarray")


class LorannBaseIndex(object):

    def __init__(
        self,
        data: np.ndarray,
        n_clusters: int,
        global_dim: Optional[int],
        quantization_bits: Optional[int] = 8,
        rank: int = 32,
        train_size: int = 5,
        distance: int = IP,
        balanced: bool = False,
        copy: bool = False,
    ) -> None:
        n_samples, dim = _check_data_matrix(data)

        assert (
            dim >= 64
        ), "LoRANN is meant for high-dimensional data: the dimensionality should be at least 64."

        assert n_clusters > 0, "n_clusters must be positive"
        assert rank > 0, "rank must be positive"
        assert train_size > 0, "train_size must be positive"

        if quantization_bits in [4, 8]:
            if rank not in [16, 32, 64]:
                raise ValueError("rank must be 16, 32, or 64")
            if global_dim is None:
                global_dim = (dim // 32) * 32
            if global_dim <= 0 or global_dim % 32:
                raise ValueError("global_dim must be a multiple of 32")
        elif quantization_bits is None:
            if global_dim is None or global_dim <= 0:
                global_dim = dim
            quantization_bits = -1
        else:
            raise ValueError("quantization_bits must be None, 4, or 8")

        self.dtype = data.dtype
        self.data_dim = data.shape[1]
        self.quantization_bits = quantization_bits
        self.global_dim = global_dim

        self.built = False

    def build(
        self,
        approximate: bool = True,
        verbose: bool = False,
        n_threads: int = -1,
        *,
        training_queries: np.ndarray = None,
    ) -> None:
        """
        Builds the LoRANN index.

        Args:
            approximate: Whether to turn on various approximations during index construction.
                Defaults to True. Setting approximate to False slows down the index construction but
                can slightly increase the recall, especially if no exact re-ranking is used in the
                query phase.
            verbose: Whether to enable verbose output. Defaults to False.
            n_threads: Number of CPU threads to use (set to -1 to use all cores)
            training_queries: An optional matrix of training queries used to build the index. Can be
                useful in the out-of-distribution setting where the training and query distributions
                differ. Ideally there should be at least as many training query points as there are
                index points.
        """
        if self.built:
            raise RuntimeError("The index has already been built")

        if training_queries is None:
            self.index.build(approximate, verbose, n_threads)
        else:
            _, dim = _check_data_matrix(training_queries)
            if dim != self.data_dim:
                raise ValueError(
                    "The training query matrix should have the same number of columns as the data matrix"
                )

            if training_queries.dtype != self.dtype:
                raise ValueError(
                    "The training query matrix should have the same dtype as the data matrix"
                )

            self.index.build(approximate, verbose, n_threads, training_queries)

        self.built = True

    def search(
        self,
        q: np.ndarray,
        k: int,
        clusters_to_search: int,
        points_to_rerank: int,
        return_distances: bool = False,
        n_threads: int = -1,
    ) -> Union[npt.NDArray[np.int32], Tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]]:
        """
        Performs an approximate nearest neighbor query for single or multiple query vectors.

        Can handle either a single query vector or multiple query vectors in parallel.
        The query is given as a numpy vector or as a numpy matrix where each row contains a query.

        Args:
            q: The query object. Can be either a single query vector or a matrix with one query
                vector per row.
            k: The number of nearest neighbors to be returned.
            clusters_to_search: Number of clusters to search.
            points_to_rerank: Number of points to re-rank using exact search. If points_to_rerank is
                set to 0, no re-ranking is performed and the original data does not need to be kept
                in memory. In this case the final returned distances are approximate distances.
            return_distances: Whether to also return distances. Defaults to False.
            n_threads: Number of CPU threads to use (set to -1 to use all cores). Only has effect if
                multiple query vectors are provided.

        Returns:
            If return_distances is False, returns a vector or matrix of indices of the approximate
            nearest neighbors in the original input data for the corresponding query. If
            return_distances is True, returns a tuple where the first element contains the nearest
            neighbors and the second element contains their distances to the query.
        """
        if not self.built:
            raise RuntimeError("Cannot query before building index")

        return self.index.search(
            q, k, clusters_to_search, points_to_rerank, return_distances, n_threads
        )

    def exact_search(
        self,
        q: np.ndarray,
        k: int,
        return_distances: bool = False,
        n_threads: int = -1,
    ) -> Union[npt.NDArray[np.int32], Tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]]:
        """
        Performs an exact nearest neighbor query for single or multiple query vectors.

        Can handle either a single query vector or multiple query vectors in parallel.
        The query is given as a numpy vector or as a numpy matrix where each row contains a query.

        Args:
            q: The query object. Can be either a single query vector or a matrix with one query
                vector per row.
            k: The number of nearest neighbors to be returned.
            return_distances: Whether to also return distances. Defaults to False.
            n_threads: Number of CPU threads to use (set to -1 to use all cores). Only has effect if
                multiple query vectors are provided.

        Returns:
            If return_distances is False, returns a vector or matrix of indices of the exact nearest
            neighbors in the original input data for the corresponding query. If return_distances is
            True, returns a tuple where the first element contains the nearest neighbors and the
            second element contains their distances to the query.
        """
        return self.index.exact_search(q, k, return_distances, n_threads)

    def save(self, fname: str) -> None:
        """
        Saves the index to a file on the disk.

        Args:
            fname: The filename to save the index to.

        Returns:
            None
        """
        if not self.built:
            raise RuntimeError("Cannot save an index that has not been built")

        self.index.save(fname)

    @classmethod
    def load(cls, fname: str):
        """
        Loads a LorannIndex from a file on the disk.

        Args:
            fname: The filename to load the index from.

        Returns:
            The loaded LorannIndex object.
        """
        self = cls.__new__(cls)
        self.index = lorannlib.load_index(fname)
        self.built = True
        return self

    def get_dissimilarity(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Calculates the dissimilarity between two vectors. The dimensions of the vectors should
        match the dimension of the index.

        Args:
            u: The first vector.
            v: The second vector.

        Returns:
            The dissimilarity between the two vectors.
        """
        if u.shape[0] != self.data_dim or v.shape[0] != self.data_dim:
            raise ValueError("The dimensionality of u and v should match the index dimension")

        return self.index.get_dissimilarity(u, v)

    @property
    def n_samples(self) -> int:
        """
        The number of samples in the index
        """
        return self.index.get_n_samples()

    @property
    def dim(self) -> int:
        """
        The dimensionality of the data in the index
        """
        return self.index.get_dim()

    @property
    def n_clusters(self) -> int:
        """
        The number of clusters in the index
        """
        return self.index.get_n_clusters()


class LorannIndex(LorannBaseIndex):

    def __init__(
        self,
        data: np.ndarray,
        n_clusters: int,
        global_dim: Optional[int],
        quantization_bits: Optional[int] = 8,
        rank: int = 32,
        train_size: int = 5,
        distance: int = IP,
        balanced: bool = False,
        copy: bool = False,
    ) -> None:
        """
        Initializes a LorannIndex object. The initializer does not build the actual index.

        Args:
            data: Index points as an $m \\times d$ numpy array.
            n_clusters: Number of clusters. In general, for $m$ index points, a good starting point
                is to set n_clusters as around $\\sqrt{m}$.
            global_dim: Globally reduced dimension ($s$). Must be either None or an integer that is
                a multiple of 32. Higher values increase recall but also increase the query latency.
                In general, a good starting point is to set global_dim = None if $d < 200$,
                global_dim = 128 if $200 \\leq d \\leq 1000$, and global_dim = 256 if $d > 1000$.
            quantization_bits: Number of bits used for quantizing the parameter matrices. Must be
                None, 4, or 8. Defaults to 8. None turns off quantization, and setting
                quantization_bits = 4 lowers the memory consumption without affecting the query
                latency but can lead to reduced recall on some data sets.
            rank: Rank ($r$) of the parameter matrices. Must be 16, 32, or 64 if quantization_bits
                is not None. Defaults to 32. Rank = 64 is mainly only useful if no exact re-ranking
                is performed in the query phase.
            train_size: Number of nearby clusters ($w$) used for training the reduced-rank
                regression models. Defaults to 5, but lower values can be used if
                $m \\gtrsim 500 000$ to speed up the index construction.
            distance: The distance measure to use. Either IP or L2. Defaults to IP.
            balanced: Whether to use balanced clustering. Defaults to False.
            copy: Whether to copy the input data. Defaults to False.

        Returns:
            None
        """
        super().__init__(
            data,
            n_clusters,
            global_dim,
            quantization_bits,
            rank,
            train_size,
            distance,
            balanced,
            copy,
        )

        match data.dtype:
            case np.float32:
                self.index_type = lorannlib.FP32LorannIndex
            case np.float16:
                if not hasattr(lorannlib, "FP16LorannIndex"):
                    raise RuntimeError("LoRANN not compiled with FP16 support")
                self.index_type = lorannlib.FP16LorannIndex
            case np.uint16:
                if not hasattr(lorannlib, "BF16LorannIndex"):
                    raise RuntimeError("LoRANN not compiled with BF16 support")
                self.index_type = lorannlib.BF16LorannIndex
            case np.uint8:
                self.index_type = lorannlib.U8LorannIndex
            case _:
                raise ValueError("invalid dtype for data matrix")

        n_samples, dim = data.shape
        self.index = self.index_type(
            data,
            n_samples,
            dim,
            self.quantization_bits,
            n_clusters,
            self.global_dim,
            rank,
            train_size,
            distance,
            balanced,
            copy,
        )


class LorannBinaryIndex(LorannBaseIndex):

    def __init__(
        self,
        data: npt.NDArray[np.uint8],
        n_clusters: int,
        global_dim: Optional[int],
        quantization_bits: Optional[int] = 8,
        rank: int = 32,
        train_size: int = 5,
        distance: int = IP,
        balanced: bool = False,
        copy: bool = False,
    ) -> None:
        """
        Initializes a LorannBinaryIndex object. The initializer does not build the actual index.

        Args:
            data: Index points as an $m \\times (d / 8)$ numpy array of type np.uint8, where bits
                have been packed into bytes, e.g. by using np.packbits.
            n_clusters: Number of clusters. In general, for $m$ index points, a good starting point
                is to set n_clusters as around $\\sqrt{m}$.
            global_dim: Globally reduced dimension ($s$). Must be either None or an integer that is
                a multiple of 32. Higher values increase recall but also increase the query latency.
                In general, a good starting point is to set global_dim = None if $d < 200$,
                global_dim = 128 if $200 \\leq d \\leq 1000$, and global_dim = 256 if $d > 1000$.
            quantization_bits: Number of bits used for quantizing the parameter matrices. Must be
                None, 4, or 8. Defaults to 8. None turns off quantization, and setting
                quantization_bits = 4 lowers the memory consumption without affecting the query
                latency but can lead to reduced recall on some data sets.
            rank: Rank ($r$) of the parameter matrices. Must be 16, 32, or 64 if quantization_bits
                is not None. Defaults to 32. Rank = 64 is mainly only useful if no exact re-ranking
                is performed in the query phase.
            train_size: Number of nearby clusters ($w$) used for training the reduced-rank
                regression models. Defaults to 5, but lower values can be used if
                $m \\gtrsim 500 000$ to speed up the index construction.
            distance: The distance measure to use. Either IP or L2. Defaults to IP.
            balanced: Whether to use balanced clustering. Defaults to False.
            copy: Whether to copy the input data. Defaults to False.

        Returns:
            None
        """
        if data.dtype != np.uint8:
            raise ValueError("Binary data should have dtype uint8")

        super().__init__(
            data,
            n_clusters,
            global_dim,
            quantization_bits,
            rank,
            train_size,
            distance,
            balanced,
            copy,
        )

        self.index_type = lorannlib.BinaryLorannIndex

        n_samples, dim = data.shape
        self.index = self.index_type(
            data,
            n_samples,
            dim * 8,
            self.quantization_bits,
            n_clusters,
            self.global_dim,
            rank,
            train_size,
            distance,
            balanced,
            copy,
        )


class KMeans(object):

    def __init__(
        self,
        n_clusters: int,
        iters: int = 25,
        distance: int = IP,
        balanced: bool = False,
        max_balance_diff: int = 16,
        penalty_factor: float = 1.4,
    ) -> None:
        """
        Initializes a KMeans object. The initializer does not perform the actual clustering.

        Args:
            n_clusters: The number of clusters ($k$).
            iters: The number of $k$-means iterations. Defaults to 25.
            distance: The distance measure to use. Either IP or L2. Defaults to IP.
            balanced: Whether to ensure clusters are balanced using an efficient balanced $k$-means
                algorithm. Defaults to False.
            max_balance_diff: The maximum allowed difference in cluster sizes for balanced
                clustering. Used only if balanced = True. Defaults to 16.
            penalty_factor: Penalty factor for balanced clustering. Higher values can be used for
                faster clustering at the cost of clustering quality. Used only if balanced = True.
                Defaults to 1.4.

        Returns:
            None
        """
        assert n_clusters > 0, "n_clusters must be positive"
        assert iters > 0, "iters must be positive"
        assert max_balance_diff > 0, "max_balance_diff must be positive"
        assert penalty_factor > 1, "penalty_factor must be greater than 1"

        self.index = lorannlib.KMeans(
            n_clusters,
            iters,
            distance,
            balanced,
            max_balance_diff,
            penalty_factor,
        )
        self.trained = False

    def train(
        self, data: npt.NDArray[np.float32], verbose: bool = False, n_threads: int = -1
    ) -> List[npt.NDArray[np.int32]]:
        """
        Performs the clustering on the provided data.

        Args:
            data: The data as an $n \\times d$ numpy array.
            verbose: Whether to enable verbose output. Defaults to False.
            n_threads: Number of CPU threads to use (set to -1 to use all cores)

        Returns:
            A list of numpy arrays, each containing the ids of the
            points assigned to the corresponding cluster.
        """
        if self.trained:
            raise RuntimeError("The clustering has already been trained")

        if data.dtype != np.float32:
            raise RuntimeError("The data matrix should have type float32")

        n_samples, dim = _check_data_matrix(data)

        if n_samples < self.n_clusters:
            raise RuntimeError(
                "The number of points should be at least as large as the number of clusters"
            )

        self.trained = True
        return self.index.train(data, n_samples, dim, verbose, n_threads)

    def assign(self, data: npt.NDArray[np.float32], k: int) -> List[npt.NDArray[np.int32]]:
        """
        Assigns given data points to their $k$ nearest clusters.

        The dimensionality of the data should match the dimensionality of the
        data that the clustering was trained on.

        Args:
            data: The data as an $m \\times d$ numpy array.
            k: The number of clusters each point is assigned to.

        Returns:
            A list of numpy arrays, one for each cluster, containing the ids of the data points
            assigned to corresponding cluster.
        """
        if not self.trained:
            raise RuntimeError("The clustering has not been trained")

        n_samples, dim = _check_data_matrix(data)

        if dim != self.dim:
            raise ValueError("The dimension should match the data dimension")

        return self.index.assign(data, n_samples, dim, k)

    def get_centroids(self) -> npt.NDArray[np.float32]:
        """
        Retrieves the centroids of the clusters.

        Returns:
            A matrix of centroids, where each row represents a centroid.
        """
        if not self.trained:
            raise RuntimeError("The clustering has not been trained")

        return self.index.get_centroids()

    @property
    def n_clusters(self) -> int:
        """
        The number of clusters
        """
        return self.index.get_n_clusters()

    @property
    def dim(self) -> int:
        """
        The dimensionality of the data the clustering was trained on
        """
        return self.get_centroids().shape[1]

    @property
    def iters(self) -> int:
        """
        The number of k-means iterations
        """
        return self.index.get_iters()

    @property
    def balanced(self) -> bool:
        """
        Whether the clustering is (approximately) balanced
        """
        return self.index.get_balanced()


def compute_V(
    A: npt.NDArray[np.float32], r: int, approximate: bool = True
) -> npt.NDArray[np.float32]:
    """
    Computes $V_r$, the first $r$ right singular vectors of $A$.

    Args:
        A
        k

    Returns:
        $V_r$
    """
    return lorannlib.compute_V(A, r)


def compute_recall(approx: npt.NDArray[np.int32], exact: npt.NDArray[np.int32]) -> float:
    """
    Computes recall given approximate and exact ground truth indices.

    Args:
        approx
        exact

    Returns:
        recall
    """
    if approx.shape != exact.shape:
        raise ValueError("Input matrices must have the same shape")

    intersection_sizes = np.zeros(len(approx))
    for i in range(len(approx)):
        intersection_sizes[i] = len(np.intersect1d(approx[i], exact[i]))

    return np.mean(intersection_sizes) / approx.shape[1]
