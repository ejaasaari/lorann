#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Dense>
#include <cstring>
#include <vector>

#include "lorann_base.h"
#include "utils.h"

#if defined(LORANN_USE_MKL)
#include "mkl.h"
#elif defined(LORANN_USE_OPENBLAS)
#include <cblas.h>
#endif

namespace Lorann {

class LorannFP : public LorannBase {
 public:
  /**
   * @brief Construct a new LorannFP object
   *
   * NOTE: The constructor does not build the actual index.
   *
   * @param data The data matrix as a float array of size $m \\times d$
   * @param m Number of points (rows) in the data matrix
   * @param d Number of dimensions (cols) in the data matrix
   * @param n_clusters Number of clusters. In general, for $m$ index points, a good starting point
   * is to set n_clusters as around $\\sqrt{m}$.
   * @param global_dim Globally reduced dimension ($s$). Must be either -1 or an integer that is a
   * multiple of 64. Higher values increase recall but also increase the query latency. In general,
   * a good starting point is to set global_dim = -1 if $d < 200$, global_dim = 128 if $200 \\leq d
   * \\leq 1000$, and global_dim = 256 if $d > 1000$.
   * @param rank Rank ($r$) of the parameter matrices. Defaults to 24. Higher ranks are mainly
   * useful if no exact re-ranking is performed in the query phase.
   * @param train_size Number of nearby clusters ($w$) used for training the reduced-rank regression
   * models. Defaults to 5, but lower values can be used if $m \\gtrsim 500 000$ to speed up the
   * index construction.
   * @param euclidean Whether to use Euclidean distance instead of (negative) inner product as the
   * dissimilarity measure. Defaults to false.
   * @param balanced Whether to use balanced clustering. Defaults to false.
   */
  explicit LorannFP(float *data, int m, int d, int n_clusters, int global_dim, int rank = 24,
                    int train_size = 5, bool euclidean = false, bool balanced = false)
      : LorannBase(data, m, d, n_clusters, global_dim, rank, train_size, euclidean, balanced) {}

  /**
   * @brief Query the index.
   *
   * @param data The query vector (dimensionality must match that of the index)
   * @param k The number of approximate nearest neighbors retrived
   * @param clusters_to_search Number of clusters to search
   * @param points_to_rerank Number of points for final (exact) re-ranking. If points_to_rerank is
   * set to 0, no re-ranking is performed and the original data does not need to be kept in memory.
   * In this case the final returned distances are approximate distances.
   * @param idx_out The index output array of length k
   * @param dist_out The (optional) distance output array of length k
   */
  void search(const float *data, const int k, const int clusters_to_search,
              const int points_to_rerank, int *idx_out, float *dist_out = nullptr) const override {
    Vector scaled_query;
    Vector transformed_query;

    Eigen::Map<const ColVector> data_vec(data, _dim);
    if (_euclidean) {
      scaled_query = -2. * data_vec;
    } else {
      scaled_query = -data_vec;
    }

    /* apply dimensionality reduction to the query */
    if (_global_dim < _dim) {
#if defined(LORANN_USE_MKL) || defined(LORANN_USE_OPENBLAS)
      transformed_query = Vector(_global_dim);
      cblas_sgemv(CblasRowMajor, CblasTrans, _global_transform.rows(), _global_transform.cols(), 1,
                  _global_transform.data(), _global_transform.cols(), scaled_query.data(), 1, 0,
                  transformed_query.data(), 1);
#else
      transformed_query = scaled_query * _global_transform;
#endif
    } else {
      transformed_query = scaled_query;
    }

    std::vector<int> I(clusters_to_search);
    select_nearest_clusters(transformed_query.data(), clusters_to_search, I.data());

    const int total_pts = _cluster_sizes(I).sum();
    Eigen::VectorXf all_distances(total_pts);
    Eigen::VectorXi all_idxs(total_pts);

#if defined(LORANN_USE_MKL) || defined(LORANN_USE_OPENBLAS)
    Vector tmp(_max_rank);
#endif

    int curr = 0;
    for (int i = 0; i < clusters_to_search; ++i) {
      const int cluster = I[i];
      const int sz = _cluster_sizes[cluster];
      if (sz == 0) continue;

      const RowMatrix &A = _A[cluster];
      const RowMatrix &B = _B[cluster];

#if defined(LORANN_USE_MKL) || defined(LORANN_USE_OPENBLAS)
      if (_euclidean) {
        std::memcpy(&all_distances[curr], _cluster_norms[cluster].data(), sizeof(float) * sz);
      } else {
        std::memset(&all_distances[curr], 0, sizeof(float) * sz);
      }

      cblas_sgemv(CblasRowMajor, CblasTrans, A.rows(), A.cols(), 1, A.data(), A.cols(),
                  transformed_query.data(), 1, 0, tmp.data(), 1);
      cblas_sgemv(CblasRowMajor, CblasTrans, B.rows(), B.cols(), 1, B.data(), B.cols(), tmp.data(),
                  1, 1, &all_distances[curr], 1);
#else
      Eigen::Map<Vector> resvec(&all_distances[curr], sz);

      if (_euclidean)
        resvec = (transformed_query * A) * B + _cluster_norms[cluster];
      else
        resvec = (transformed_query * A) * B;
#endif

      std::memcpy(&all_idxs[curr], _cluster_map[cluster].data(), sz * sizeof(int));
      curr += sz;
    }

    select_final(_euclidean ? data : scaled_query.data(), k, points_to_rerank, total_pts,
                 all_idxs.data(), all_distances.data(), idx_out, dist_out);
  }

  using LorannBase::build;

  /**
   * @brief Build the index.
   *
   * @param query_data A float array of training queries of size $n \\times d$ used to build the
   * index. Can be useful in the out-of-distribution setting where the training and query
   * distributions differ. Ideally there should be at least as many training query points as there
   * are index points.
   * @param query_n The number of training queries
   * @param approximate Whether to turn on various approximations during index construction.
   * Defaults to true. Setting approximate to false slows down the index construction but can
   * slightly increase the recall, especially if no exact re-ranking is used in the query phase.
   * @param num_threads Number of CPU threads to use (set to -1 to use all cores)
   */
  void build(const float *query_data, const int query_n, const bool approximate = true,
             int num_threads = -1) override {
    LORANN_ENSURE_POSITIVE(query_n);

#ifdef _OPENMP
    if (num_threads <= 0) {
      num_threads = omp_get_max_threads();
    }
#endif

    Eigen::Map<RowMatrix> train_mat(_data, _n_samples, _dim);
    Eigen::Map<const RowMatrix> query_mat(query_data, query_n, _dim);

    KMeans global_clustering(_n_clusters, KMEANS_ITERATIONS, _euclidean, _balanced,
                             KMEANS_MAX_BALANCE_DIFF, 0);

    std::vector<std::vector<int>> cluster_train_map;
    if (_global_dim < _dim) {
      RowMatrix query_sample = sample_rows(query_mat, GLOBAL_DIM_REDUCTION_SAMPLES);
      _global_transform =
          compute_principal_components(query_sample.transpose() * query_sample, _global_dim);
      RowMatrix reduced_train_mat = train_mat * _global_transform;

      if (query_mat.data() != train_mat.data()) {
        RowMatrix reduced_query_mat = query_mat * _global_transform;
        cluster_train_map = clustering(global_clustering, reduced_train_mat.data(),
                                       reduced_train_mat.rows(), reduced_query_mat.data(),
                                       reduced_query_mat.rows(), approximate, num_threads);
      } else {
        cluster_train_map = clustering(global_clustering, reduced_train_mat.data(),
                                       reduced_train_mat.rows(), reduced_train_mat.data(),
                                       reduced_train_mat.rows(), approximate, num_threads);
      }
    } else {
      cluster_train_map = clustering(global_clustering, train_mat.data(), train_mat.rows(),
                                     query_mat.data(), query_mat.rows(), approximate, num_threads);
    }

    _centroid_mat = global_clustering.get_centroids();

    if (_euclidean) {
      _global_centroid_norms = _centroid_mat.rowwise().squaredNorm();
      _data_norms = train_mat.rowwise().squaredNorm();
      _cluster_norms.resize(_n_clusters);
    }

    _A.resize(_n_clusters);
    _B.resize(_n_clusters);

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (int i = 0; i < _n_clusters; ++i) {
      if (_cluster_map[i].size() == 0) continue;

      if (_euclidean) {
        _cluster_norms[i] = _data_norms(_cluster_map[i]);
      }

      RowMatrix pts = train_mat(_cluster_map[i], Eigen::placeholders::all);
      RowMatrix Q;

      if (cluster_train_map[i].size() >= _cluster_map[i].size()) {
        Q = query_mat(cluster_train_map[i], Eigen::placeholders::all);
      } else {
        Q = pts;
      }

      /* compute reduced-rank regression solution */
      Eigen::MatrixXf beta_hat, Y_hat;
      if (_global_dim < _dim) {
        if (approximate) {
          beta_hat = (pts * _global_transform).transpose();
          Y_hat = (Q * _global_transform) * beta_hat;
        } else {
          Eigen::MatrixXf X = Q * _global_transform;
          beta_hat = X.colPivHouseholderQr().solve(Q * pts.transpose());
          Y_hat = X * beta_hat;
        }
      } else {
        beta_hat = pts.transpose();
        Y_hat = Q * pts.transpose();
      }

      Eigen::MatrixXf V = compute_V(Y_hat, _max_rank, approximate);
      _A[i] = beta_hat * V;
      _B[i] = V.transpose();
    }

    _cluster_sizes = Eigen::VectorXi(_n_clusters);
    for (int i = 0; i < _n_clusters; ++i) {
      _cluster_sizes(i) = static_cast<int>(_cluster_map[i].size());
    }
  }

 private:
  LorannFP() = default; /* default constructor should only be used for serialization */

  void select_nearest_clusters(const float *x, int k, int *out) const {
    Vector d(_n_clusters);

    const float *y = _centroid_mat.data();
    for (int i = 0; i < _n_clusters; ++i, y += _global_dim) {
      d[i] = dot_product(x, y, _global_dim);
    }

    if (_euclidean) d += _global_centroid_norms;

    select_k(k, out, d.size(), NULL, d.data());
  }

  friend class cereal::access;

  template <class Archive>
  void save(Archive &ar) const {
    ar(cereal::base_class<LorannBase>(this), _global_transform, _centroid_mat, _A, _B,
       _cluster_norms);
  }

  template <class Archive>
  void load(Archive &ar) {
    ar(cereal::base_class<LorannBase>(this), _global_transform, _centroid_mat, _A, _B,
       _cluster_norms);
  }

  RowMatrix _global_transform;
  RowMatrix _centroid_mat;

  std::vector<RowMatrix> _A;
  std::vector<RowMatrix> _B;
  std::vector<Vector> _cluster_norms;
};

}  // namespace Lorann

CEREAL_REGISTER_TYPE(Lorann::LorannFP)
CEREAL_REGISTER_POLYMORPHIC_RELATION(Lorann::LorannBase, Lorann::LorannFP)