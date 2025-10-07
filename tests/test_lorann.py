import pytest
import numpy as np
import lorann
from sklearn.datasets import fetch_openml


@pytest.fixture(scope="module")
def mnist_data():
    X, _ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = np.ascontiguousarray(X, dtype=np.float32)
    return X


@pytest.fixture
def small_data():
    np.random.seed(42)
    return np.random.rand(1000, 128).astype(np.float32)


class TestComputeRecall:
    """Test the compute_recall function"""

    def test_compute_recall_perfect_match(self):
        """Test recall when approximate matches exact"""
        exact = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
        approx = exact.copy()
        recall = lorann.compute_recall(approx, exact)
        assert recall == 1.0

    def test_compute_recall_no_match(self):
        """Test recall when no matches"""
        exact = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        approx = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int32)
        recall = lorann.compute_recall(approx, exact)
        assert recall == 0.0

    def test_compute_recall_partial_match(self):
        """Test recall with partial matches"""
        exact = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
        approx = np.array([[1, 2, 9, 10], [5, 6, 11, 12]], dtype=np.int32)
        recall = lorann.compute_recall(approx, exact)
        assert recall == 0.5

    def test_compute_recall_shape_mismatch(self):
        """Test that shape mismatch raises ValueError"""
        exact = np.array([[1, 2, 3]], dtype=np.int32)
        approx = np.array([[1, 2]], dtype=np.int32)
        with pytest.raises(ValueError):
            lorann.compute_recall(approx, exact)


class TestComputeV:
    """Test the compute_V function"""

    def test_compute_v_basic(self):
        """Test basic functionality of compute_V"""
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        r = 2
        V = lorann.compute_V(A, r).T

        assert V.shape == (3, r)

        # Check that V has orthonormal columns
        VtV = V.T @ V
        np.testing.assert_allclose(VtV, np.eye(r), rtol=1e-5, atol=1e-5)

    def test_compute_v_rectangular(self):
        """Test compute_V with rectangular matrix"""
        A = np.random.rand(10, 5).astype(np.float32)
        r = 3
        V = lorann.compute_V(A, r).T

        assert V.shape == (5, r)

        # Check that V has orthonormal columns
        VtV = V.T @ V
        np.testing.assert_allclose(VtV, np.eye(r), rtol=1e-5, atol=1e-5)


class TestKMeans:
    """Test the KMeans clustering functionality"""

    def test_kmeans_initialization(self):
        """Test KMeans initialization"""
        kmeans = lorann.KMeans(n_clusters=10, iters=25, distance=lorann.IP)
        assert kmeans.n_clusters == 10
        assert kmeans.iters == 25
        assert not kmeans.trained

    def test_kmeans_train(self, mnist_data):
        """Test KMeans training on MNIST subset"""
        data = mnist_data[:10000]
        kmeans = lorann.KMeans(n_clusters=128, iters=10, distance=lorann.L2)

        cluster_assignments = kmeans.train(data, verbose=False, n_threads=-1)

        assert len(cluster_assignments) == 128
        assert kmeans.trained

        # Check that all points are assigned
        all_points = []
        for cluster in cluster_assignments:
            all_points.extend(cluster)
        assert len(all_points) == 10000
        assert len(set(all_points)) == 10000

    def test_kmeans_centroids(self, mnist_data):
        """Test getting centroids after training"""
        data = mnist_data[:1000]
        kmeans = lorann.KMeans(n_clusters=10)
        kmeans.train(data)

        centroids = kmeans.get_centroids()
        assert centroids.shape == (10, 784)
        assert kmeans.dim == 784

    def test_kmeans_assign(self, mnist_data):
        """Test assigning new points to clusters"""
        train_data = mnist_data[:1000]
        test_data = mnist_data[1000:1100]

        kmeans = lorann.KMeans(n_clusters=10)
        kmeans.train(train_data)

        assignments = kmeans.assign(test_data, k=3)

        assert len(assignments) == 10

        total_assignments = sum(len(cluster) for cluster in assignments)
        assert total_assignments == 100 * 3

    def test_kmeans_balanced(self, mnist_data):
        """Test balanced KMeans"""
        data = mnist_data[:5000]
        kmeans = lorann.KMeans(
            n_clusters=64, balanced=True, max_balance_diff=16, penalty_factor=1.4
        )

        cluster_assignments = kmeans.train(data)

        cluster_sizes = [len(cluster) for cluster in cluster_assignments]
        assert max(cluster_sizes) - min(cluster_sizes) <= 16
        assert kmeans.balanced

    def test_kmeans_errors(self, small_data):
        """Test error conditions"""
        kmeans = lorann.KMeans(n_clusters=10)

        # Test querying before training
        with pytest.raises(RuntimeError):
            kmeans.get_centroids()

        # Test assigning before training
        with pytest.raises(RuntimeError):
            kmeans.assign(small_data, k=1)

        # Test training twice
        kmeans.train(small_data)
        with pytest.raises(RuntimeError):
            kmeans.train(small_data)


class TestLorannBinaryIndex:
    """Test LorannBinaryIndex functionality"""

    @pytest.fixture
    def binary_mnist_data(self, mnist_data):
        return np.packbits(mnist_data > 0).reshape(len(mnist_data), -1)

    def test_binary_index_initialization(self, binary_mnist_data):
        """Test binary index initialization"""
        index = lorann.LorannBinaryIndex(
            data=binary_mnist_data, n_clusters=128, global_dim=128, quantization_bits=8, rank=32
        )
        assert not index.built
        assert index.n_samples == len(binary_mnist_data)
        assert index.dim == 784

    def test_binary_index_build(self, binary_mnist_data):
        """Test building binary index"""
        index = lorann.LorannBinaryIndex(
            data=binary_mnist_data[:10_000],
            n_clusters=128,
            global_dim=None,
            distance=lorann.HAMMING,
        )

        index.build(approximate=True, verbose=False)
        assert index.built

        # Test that we can't build twice
        with pytest.raises(RuntimeError):
            index.build()

    def test_binary_index_search(self, binary_mnist_data):
        """Test search functionality with binary index"""
        train_data = binary_mnist_data[:8000]
        test_data = binary_mnist_data[8000:8100]

        index = lorann.LorannBinaryIndex(
            data=train_data, n_clusters=64, global_dim=128, distance=lorann.HAMMING
        )
        index.build()

        k = 10
        # Search without distances
        results = index.search(test_data, k=k, clusters_to_search=10, points_to_rerank=50)
        assert results.shape == (100, k)

        # Search with distances
        results, distances = index.search(
            test_data, k=k, clusters_to_search=10, points_to_rerank=50, return_distances=True
        )
        assert results.shape == (100, k)
        assert distances.shape == (100, k)

    def test_binary_index_recall(self, binary_mnist_data):
        """Test that binary index achieves good recall"""
        data = binary_mnist_data[:59_000]
        queries = binary_mnist_data[59_000:]

        index = lorann.LorannBinaryIndex(
            data=data, n_clusters=256, global_dim=128, distance=lorann.HAMMING
        )
        index.build()

        k = 100
        approx = index.search(queries, k=k, clusters_to_search=16, points_to_rerank=400)
        exact = index.exact_search(queries, k=k)

        recall = lorann.compute_recall(approx, exact)
        assert recall >= 0.9

    def test_binary_index_save_load(self, binary_mnist_data, tmp_path):
        """Test saving and loading binary index"""
        index = lorann.LorannBinaryIndex(
            data=binary_mnist_data[:1000], n_clusters=10, global_dim=None
        )
        index.build()

        # Save index
        filepath = tmp_path / "binary_index.bin"
        index.save(str(filepath))

        # Load index
        loaded_index = lorann.LorannBinaryIndex.load(str(filepath))
        assert loaded_index.built
        assert loaded_index.n_samples == 1000
        assert loaded_index.n_clusters == 10


recall_parameters = []
for dist in [lorann.L2, lorann.IP]:
    for qb in [4, 8, None]:
        for r in [16, 32, 64]:
            for dtype in [np.float32, np.float16, np.uint16, np.uint8]:
                if dtype == np.uint8 and dist == lorann.IP:
                    continue
                recall_parameters.append((dist, qb, r, dtype))


class TestLorannIndex:
    """Test LorannIndex with different data types"""

    @pytest.mark.parametrize(
        "dtype,expected_exception", [(np.float32, None), (np.float16, "FP16"), (np.uint8, None)]
    )
    def test_lorann_index_dtypes(self, mnist_data, dtype, expected_exception):
        """Test LorannIndex with different data types"""
        data = mnist_data.astype(dtype)

        if expected_exception == "FP16" and not hasattr(lorann.lorannlib, "FP16LorannIndex"):
            with pytest.raises(RuntimeError, match="LoRANN not compiled with FP16 support"):
                lorann.LorannIndex(data=data, n_clusters=50, global_dim=128)
            return

        index = lorann.LorannIndex(
            data=data[:10_000],
            n_clusters=50,
            global_dim=128,
            quantization_bits=8,
            distance=lorann.L2,
        )

        assert index.dtype == dtype
        assert index.data_dim == 784

        index.build(approximate=True, verbose=False)
        assert index.built

    @pytest.mark.parametrize(
        "distance,quantization_bits,rank,dtype",
        recall_parameters,
        ids=lambda val: (
            f"{val[0]}_q{val[1]}_{val[2]}_{val[3]}" if isinstance(val, tuple) else str(val)
        ),
    )
    def test_lorann_index_recall(self, mnist_data, distance, quantization_bits, rank, dtype):
        """Test search recall"""
        if dtype == np.float16 and not hasattr(lorann.lorannlib, "FP16LorannIndex"):
            pytest.skip("float16 not supported in this lorann build")

        if dtype == np.uint16 and not hasattr(lorann.lorannlib, "BF16LorannIndex"):
            pytest.skip("bfloat16 not supported in this lorann build")

        data = mnist_data[:50_000].copy()
        queries = mnist_data[50_000:50_100].copy()

        if dtype != np.uint8:
            data /= np.linalg.norm(data, axis=1)[:, np.newaxis]
            queries /= np.linalg.norm(queries, axis=1)[:, np.newaxis]

        if dtype == np.uint16:
            data_f32 = ((data.view(np.uint32) + 0x8000) & 0xFFFF0000).view(np.float32)
            queries_f32 = ((queries.view(np.uint32) + 0x8000) & 0xFFFF0000).view(np.float32)
            data = np.right_shift(data_f32.view(np.uint32), 16).astype(np.uint16)
            queries = np.right_shift(queries_f32.view(np.uint32), 16).astype(np.uint16)
        else:
            data = data.astype(dtype)
            queries = queries.astype(dtype)

        index = lorann.LorannIndex(
            data=data,
            n_clusters=128,
            global_dim=128,
            rank=rank,
            quantization_bits=quantization_bits,
            distance=distance,
        )
        index.build()

        k = 100
        clusters_to_search = 8
        points_to_rerank = 400

        approx = index.search(
            queries, k=k, clusters_to_search=clusters_to_search, points_to_rerank=points_to_rerank
        )
        exact = index.exact_search(queries, k=k)

        recall = lorann.compute_recall(approx, exact)

        assert recall >= 0.9, (
            f"Recall {recall:.3f} < 0.9 for "
            f"dtype={dtype.__name__}, distance={distance}, "
            f"rank={rank}, quantization_bits={quantization_bits}"
        )

    def test_lorann_index_all_functions(self, mnist_data):
        """Test all exposed functions of LorannIndex"""
        data = mnist_data[:1000]
        index = lorann.LorannIndex(
            data=data,
            n_clusters=16,
            global_dim=None,
            quantization_bits=None,
            rank=32,
            train_size=5,
            distance=lorann.IP,
            balanced=False,
        )

        assert index.n_samples == 1000
        assert index.dim == 784

        training_queries = mnist_data[1000:2000]
        index.build(
            approximate=False, verbose=False, n_threads=2, training_queries=training_queries
        )

        assert index.n_clusters == 16

        # Test dissimilarity
        u = data[0]
        v = data[1]
        dissim = index.get_dissimilarity(u, v)
        assert isinstance(dissim, float)

        # Test single query search
        single_query = data[0]
        results = index.search(single_query, k=5, clusters_to_search=5, points_to_rerank=20)
        assert results.shape == (5,)

        # Test batch query search
        batch_queries = data[:10]
        results = index.search(
            batch_queries, k=5, clusters_to_search=5, points_to_rerank=20, n_threads=2
        )
        assert results.shape == (10, 5)

    def test_lorann_index_errors(self, mnist_data):
        """Test error conditions for LorannIndex"""
        # Test invalid data shape
        with pytest.raises(ValueError):
            lorann.LorannIndex(data=np.array([1, 2, 3]), n_clusters=10, global_dim=None)

        # Test NaN in data
        bad_data = mnist_data[:100].copy()
        bad_data[0, 0] = np.nan
        with pytest.raises(RuntimeError):
            index = lorann.LorannIndex(bad_data, n_clusters=10, global_dim=None)

        # Test dimension too small
        small_dim_data = np.random.rand(100, 16).astype(np.float32)
        with pytest.raises(AssertionError):
            lorann.LorannIndex(small_dim_data, n_clusters=10, global_dim=None)

        # Test invalid quantization bits
        with pytest.raises(ValueError):
            lorann.LorannIndex(
                mnist_data[:100], n_clusters=10, global_dim=None, quantization_bits=16
            )

        # Test invalid rank with quantization
        with pytest.raises(ValueError):
            lorann.LorannIndex(
                mnist_data[:100], n_clusters=10, global_dim=None, quantization_bits=8, rank=48
            )

        # Test querying before building
        index = lorann.LorannIndex(mnist_data[:100], n_clusters=10, global_dim=None)
        with pytest.raises(RuntimeError):
            index.search(mnist_data[0], k=5, clusters_to_search=5, points_to_rerank=10)

        # Test saving before building
        with pytest.raises(RuntimeError):
            index.save("test.bin")
