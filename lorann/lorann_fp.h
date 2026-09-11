#pragma once

#define EIGEN_DONT_PARALLELIZE

#include <Eigen/Dense>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include "lorann_base.h"
#include "utils.h"

#if defined(LORANN_USE_MKL)
#include "mkl.h"
#elif defined(LORANN_USE_OPENBLAS)
#include <cblas.h>
#endif

namespace Lorann {

template <typename T>
class LorannFP final : public LorannBase<T> {
 public:
  using LorannBase<T>::build;

  /**
   * @brief Construct a new LorannFP object
   *
   * NOTE: The constructor does not build the actual index.
   *
   * @param data The data matrix as a T array of size $m \\times d$, where T is either float,
   * simsimd_f16_t, simsimd_bf16_t, uint8_t, or Lorann::BinaryType
   * @param m Number of points (rows) in the data matrix
   * @param d Number of dimensions (cols) in the data matrix
   * @param n_clusters Number of clusters. In general, for $m$ index points, a good starting point
   * is to set n_clusters as around $\\sqrt{m}$.
   * @param global_dim Globally reduced dimension ($s$). Must be either -1 or an integer that is a
   * multiple of 64. Higher values increase recall but also increase the query latency. In general,
   * a good starting point is to set global_dim = -1 if $d < 200$, global_dim = 128 if $200 \\leq d
   * \\leq 1000$, and global_dim = 256 if $d > 1000$.
   * @param rank Rank ($r$) of the parameter matrices. Must be less than d. Defaults to 24.
   * @param train_size Number of nearby clusters ($w$) used for training the reduced-rank regression
   * models. Defaults to 5, but lower values can be used if $m \\gtrsim 500 000$ to speed up the
   * index construction.
   * @param distance The distance measure to use. Either IP or L2. Defaults to IP.
   * @param balanced Whether to use balanced clustering. Defaults to false.
   * @param copy Whether to copy the input data. Defaults to false.
   */
  explicit LorannFP(T *data, int m, int d, int n_clusters, int global_dim, int rank = 24,
                    int train_size = 5, Distance distance = IP, bool balanced = false,
                    bool copy = false)
      : LorannBase<T>(data, m, d, n_clusters, global_dim, rank, train_size, distance, balanced,
                      copy) {
    if (rank >= d) {
      throw std::invalid_argument("rank must be less than the input dimensionality");
    }
  }

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
  void search(const T *data, const int k, const int clusters_to_search, const int points_to_rerank,
              int *idx_out, lorann_dist_t *dist_out = nullptr) const override {
    Vector scaled_query;
    Vector transformed_query;
    ColVector data_vec = detail::Traits<T>::to_float_vector(data, _dim);

    if (_distance == L2) {
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
      if (_distance == L2) {
        std::memcpy(&all_distances[curr], _cluster_norms[cluster].data(),
                    static_cast<std::size_t>(sz) * sizeof(float));
      } else {
        std::memset(&all_distances[curr], 0, static_cast<std::size_t>(sz) * sizeof(float));
      }

      cblas_sgemv(CblasRowMajor, CblasTrans, A.rows(), A.cols(), 1, A.data(), A.cols(),
                  transformed_query.data(), 1, 0, tmp.data(), 1);
      cblas_sgemv(CblasRowMajor, CblasTrans, B.rows(), B.cols(), 1, B.data(), B.cols(), tmp.data(),
                  1, 1, &all_distances[curr], 1);
#else
      Eigen::Map<Vector> resvec(&all_distances[curr], sz);

      if (_distance == L2)
        resvec = (transformed_query * A) * B + _cluster_norms[cluster];
      else
        resvec = (transformed_query * A) * B;
#endif

      std::memcpy(&all_idxs[curr], _cluster_map[cluster].data(),
                  static_cast<std::size_t>(sz) * sizeof(int));
      curr += sz;
    }

    select_final(data, _distance == L2 ? data_vec.data() : scaled_query.data(), k, points_to_rerank,
                 total_pts, all_idxs.data(), all_distances.data(), idx_out, dist_out);
  }

  /**
   * @brief Build the index.
   *
   * @param query_data An array of training queries of size $n \\times d$ used to build the
   * index. Can be useful in the out-of-distribution setting where the training and query
   * distributions differ. Ideally there should be at least as many training query points as there
   * are index points.
   * @param query_n The number of training queries
   * @param approximate Whether to turn on various approximations during index construction.
   * Defaults to true. Setting approximate to false slows down the index construction but can
   * slightly increase the recall, especially if no exact re-ranking is used in the query phase.
   * @param verbose Whether to use verbose output for index construction. Defaults to false.
   * @param num_threads Number of CPU threads to use (set to -1 to use all cores)
   */
  void build(const T *query_data, const int query_n, const bool approximate = true,
             const bool verbose = false, int num_threads = -1) override {
    auto build_context = this->prepare_build(query_data, query_n, num_threads);
    const auto &train_mat = build_context.train_mat.view;
    const auto &query_mat = build_context.query_mat.view;
    KMeans global_clustering = this->make_global_clustering(approximate);

    std::vector<std::vector<int>> cluster_train_map;
    std::optional<typename LorannBase<T>::ProjectedBuildData> projected_data;
    if (_global_dim < _dim) {
      RowMatrix query_sample = sample_rows(query_mat, GLOBAL_DIM_REDUCTION_SAMPLES);
      _global_transform = compute_principal_components_from_rows(query_sample, _global_dim);
      projected_data.emplace(build_context, _global_transform);
      cluster_train_map =
          this->cluster_reduced_data(global_clustering, build_context, projected_data->train_rows(),
                                     projected_data->query_rows(), verbose);
      projected_data->retain(
          this->projected_data_usage(build_context, cluster_train_map, approximate));
    } else {
      cluster_train_map = this->cluster_build_data(global_clustering, build_context, verbose);
    }

    _centroid_mat = std::move(global_clustering).get_centroids();
    this->initialize_distance_data(train_mat, _centroid_mat);

    _A.resize(_n_clusters);
    _B.resize(_n_clusters);

    this->build_cluster_models(
        build_context, cluster_train_map, projected_data ? &*projected_data : nullptr, approximate,
        verbose, _cluster_norms, [&](const int i, RowMatrix &&A, const Eigen::MatrixXf &V) {
          _A[i] = std::move(A);
          _B[i] = V.transpose();
        });
  }

 private:
  LorannFP() = default; /* default constructor should only be used for serialization */

  void select_nearest_clusters(const float *x, int k, int *out) const {
    Vector d(_n_clusters);

    const float *y = _centroid_mat.data();
    for (int i = 0; i < _n_clusters; ++i, y += _global_dim) {
      d[i] = detail::Traits<float>::dot_product(x, y, _global_dim);
    }

    if (_distance == L2) d += _global_centroid_norms;

    select_k<float>(k, out, d.size(), NULL, d.data());
  }

  friend class cereal::access;

  template <class Archive>
  void save(Archive &ar) const {
    ar(cereal::base_class<LorannBase<T>>(this), _global_transform, _centroid_mat, _A, _B,
       _cluster_norms);
  }

  template <class Archive>
  void load(Archive &ar) {
    ar(cereal::base_class<LorannBase<T>>(this), _global_transform, _centroid_mat, _A, _B,
       _cluster_norms);
  }

  RowMatrix _global_transform;
  RowMatrix _centroid_mat;

  std::vector<RowMatrix> _A;
  std::vector<RowMatrix> _B;
  std::vector<Vector> _cluster_norms;

  using LorannBase<T>::select_final;

  using LorannBase<T>::_data;
  using LorannBase<T>::_n_samples;
  using LorannBase<T>::_dim;
  using LorannBase<T>::_n_clusters;
  using LorannBase<T>::_global_dim;
  using LorannBase<T>::_max_rank;
  using LorannBase<T>::_train_size;
  using LorannBase<T>::_distance;
  using LorannBase<T>::_balanced;
  using LorannBase<T>::_copy;
  using LorannBase<T>::_cluster_map;
  using LorannBase<T>::_global_centroid_norms;
  using LorannBase<T>::_cluster_sizes;
  using LorannBase<T>::_data_norms;
};

}  // namespace Lorann

#define REGISTER_LORANNFP_TYPES(DataType)          \
  CEREAL_REGISTER_TYPE(Lorann::LorannFP<DataType>) \
  CEREAL_REGISTER_POLYMORPHIC_RELATION(Lorann::LorannBase<DataType>, Lorann::LorannFP<DataType>)

REGISTER_LORANNFP_TYPES(float)
#if SIMSIMD_NATIVE_F16
REGISTER_LORANNFP_TYPES(simsimd_f16_t)
#endif
#if SIMSIMD_NATIVE_BF16
REGISTER_LORANNFP_TYPES(simsimd_bf16_t)
#endif
REGISTER_LORANNFP_TYPES(uint8_t)
REGISTER_LORANNFP_TYPES(Lorann::BinaryType)
