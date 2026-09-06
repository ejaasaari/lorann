#pragma once

#define EIGEN_DONT_PARALLELIZE

#include <Eigen/Dense>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "lorann_base.h"
#include "quant.h"
#include "utils.h"

#if defined(LORANN_USE_MKL)
#include "mkl.h"
#elif defined(LORANN_USE_OPENBLAS)
#include <cblas.h>
#endif

namespace Lorann {

template <typename T, typename DataQuantizer = SQ8Quantizer, typename QueryQuantizer = SQ8Quantizer>
class Lorann final : public LorannBase<T> {
 public:
  using LorannBase<T>::build;

  /**
   * @brief Construct a new Lorann object
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
   * multiple of 32. If global_dim = -1, no dimensionality reduction is used, but the original
   * dimensionality must be a multiple of 32 in this case. Higher values increase recall but also
   * increase the query latency. In general, a good starting point is to set global_dim = -1 if
   * $d < 200$, global_dim = 128 if $200 \\leq d \\leq 1000$, and global_dim = 256 if $d > 1000$.
   * @param rank Rank ($r$) of the parameter matrices. Must be 16, 32, or 64. Defaults to 32. Rank =
   * 64 is mainly useful if no exact re-ranking is performed in the query phase.
   * @param train_size Number of nearby clusters ($w$) used for training the reduced-rank regression
   * models. Defaults to 5, but lower values can be used if $m \\gtrsim 500 000$ to speed up the
   * index construction.
   * @param distance The distance measure to use. Either IP or L2. Defaults to IP.
   * @param balanced Whether to use balanced clustering. Defaults to false.
   * @param copy Whether to copy the input data. Defaults to false.
   */
  explicit Lorann(T *data, int m, int d, int n_clusters, int global_dim, int rank = 32,
                  int train_size = 5, Distance distance = IP, bool balanced = false,
                  bool copy = false)
      : LorannBase<T>(data, m, d, n_clusters, global_dim, rank + 1, train_size, distance, balanced,
                      copy) {
    if (!(rank == 16 || rank == 32 || rank == 64)) {
      throw std::invalid_argument("rank must be 16, 32, or 64");
    }

    if (_global_dim % 32) {
      throw std::invalid_argument("global_dim must be a multiple of 32");
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
    ColVector scaled_query;
    ColVector transformed_query;
    ColVector data_vec = detail::Traits<T>::to_float_vector(data, _dim);

    if (_distance == L2) {
      scaled_query = -2 * data_vec;
    } else {
      scaled_query = -data_vec;
    }

    /* apply dimensionality reduction to the query */
#if defined(LORANN_USE_MKL) || defined(LORANN_USE_OPENBLAS)
    transformed_query = Vector(_global_dim);
    cblas_sgemv(CblasRowMajor, CblasTrans, _global_transform.rows(), _global_transform.cols(), 1,
                _global_transform.data(), _global_transform.cols(), scaled_query.data(), 1, 0,
                transformed_query.data(), 1);
#else
    transformed_query = _global_transform.transpose() * scaled_query;
#endif

    const float principal_axis = transformed_query[0];
    transformed_query[0] = 0; /* the first component is treated separately in fp32 precision */

    /* quantize the transformed query vector */
    VectorInt8 quantized_query(_global_dim);
    VectorInt8 quantized_query_doubled(_max_rank - 1);
    const float quantization_factor =
        quant_query.quantize_vector(transformed_query.data(), _global_dim, quantized_query.data());

    const float compensation = quantized_query.cast<float>().sum();
    const float compensation_data = compensation * quant_data.compensation_factor;
    const float compensation_query = compensation * quant_query.compensation_factor;

    std::vector<int> I(clusters_to_search);
    select_nearest_clusters(quantized_query, quantization_factor, principal_axis,
                            compensation_query, clusters_to_search, I.data());

    const int total_pts = _cluster_sizes(I).sum();
    ColVector all_distances(total_pts);
    ColVectorInt all_idxs(total_pts);
    ColVector tmp(_max_rank);

    int curr = 0;
    for (int i = 0; i < clusters_to_search; ++i) {
      const int cluster = I[i];
      const int sz = _cluster_sizes[cluster];
      if (sz == 0) continue;

      const ColMatrixUInt8 &A = _A[cluster];
      const ColMatrixUInt8 &B = _B[cluster];
      const Vector &A_correction = _A_corrections[cluster];
      const Vector &B_correction = _B_corrections[cluster];

      /* compute s = q^T A */
      quant_data.quantized_matvec_product_A(A, quantized_query, A_correction, quantization_factor,
                                            principal_axis, compensation_data, tmp.data());
      const float principal_axis_tmp = tmp[0];

      const float tmpfact = quant_query.quantize_vector(tmp.data() + 1, _max_rank - 1,
                                                        quantized_query_doubled.data());
      const float compensation_tmp =
          quantized_query_doubled.cast<float>().sum() * quant_data.compensation_factor;

      /* compute r = s^T B */
      quant_data.quantized_matvec_product_B(B, quantized_query_doubled, B_correction, tmpfact,
                                            principal_axis_tmp, compensation_tmp,
                                            &all_distances[curr]);

      if (_distance == L2)
        add_inplace(_cluster_norms[cluster].data(), &all_distances[curr],
                    _cluster_norms[cluster].size());

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

    /* compute dimensionality reduction matrix */
    RowMatrix query_sample = sample_rows(query_mat, GLOBAL_DIM_REDUCTION_SAMPLES);
    Eigen::MatrixXf global_dim_reduction =
        compute_principal_components_from_rows(query_sample, _global_dim);

    /* rotate the dimensionality reduction matrix beforehand so that we do not need to rotate
     * queries at query time */
    Eigen::MatrixXf sub_rotation = generate_rotation_matrix(_global_dim - 1);
    Eigen::MatrixXf rotation = Eigen::MatrixXf::Zero(_global_dim, _global_dim);
    rotation(0, 0) = 1;
    rotation.block(1, 1, _global_dim - 1, _global_dim - 1) = sub_rotation;
    _global_transform = global_dim_reduction * rotation;

    /* Project directly into the final rotated basis. Because rotation is orthogonal, clustering
     * in these coordinates has the same objective as clustering in the PCA coordinates. Let n be
     * the point count, d the input dimension, s the reduced dimension, q the number of distinct
     * training-query rows, and w their average cluster reuse. Caching replaces O((n + w*q) * d*s)
     * repeated projection work with O((n + q) * d*s) projection work and O((n + w*q) * s) row
     * gathering. Folding rotation into _global_transform also avoids a separate
     * O((n + q) * s^2) pass over the cached rows. */
    typename LorannBase<T>::ProjectedBuildData projected_data(build_context, _global_transform);

    KMeans global_clustering = this->make_global_clustering(approximate);
    std::vector<std::vector<int>> cluster_train_map =
        this->cluster_reduced_data(global_clustering, build_context, projected_data.train_rows(),
                                   projected_data.query_rows(), verbose);

    const auto projected_usage =
        this->projected_data_usage(build_context, cluster_train_map, approximate);
    projected_data.retain(projected_usage);

    /* The centroids are already expressed in the final rotated basis. */
    RowMatrix centroid_mat = std::move(global_clustering).get_centroids();
    ColMatrix centroid_mat_rotated = centroid_mat.transpose();
    Vector centroid_fix = centroid_mat_rotated.row(0);
    centroid_mat_rotated.row(0).array() *= 0;

    this->initialize_distance_data(train_mat, centroid_mat);

    /* quantize the cluster centroids */
    _centroids_quantized = ColMatrixUInt8(centroid_mat_rotated.rows(), centroid_mat_rotated.cols());
    _centroid_correction = Vector(_centroids_quantized.cols() * 2);
    quant_query.quantize_centroids_unsigned(centroid_mat_rotated, _centroids_quantized.data(),
                                           _centroid_correction.data());

    _centroid_correction(Eigen::seqN(_n_clusters, _n_clusters)) = centroid_fix;

    _A.resize(_n_clusters);
    _B.resize(_n_clusters);
    _A_corrections.resize(_n_clusters);
    _B_corrections.resize(_n_clusters);

    Eigen::MatrixXf rank_rotation = Eigen::MatrixXf::Zero(_max_rank, _max_rank);
    rank_rotation(0, 0) = 1;
    if (_max_rank > 1) {
      rank_rotation.block(1, 1, _max_rank - 1, _max_rank - 1) =
          generate_rotation_matrix(_max_rank - 1);
    }

    this->build_cluster_models(
        build_context, cluster_train_map, &projected_data, approximate, verbose, _cluster_norms,
        [&](const int i, const RowMatrix &A_before_rank_rotation, const Eigen::MatrixXf &V) {
          /* Apply the same rank-space rotation to every cluster. */
          ColMatrix A = A_before_rank_rotation * rank_rotation;
          ColMatrix B = rank_rotation.transpose() * V.transpose();

          /* quantize the A and B matrices */
          ColMatrixUInt8 A_quantized(A.rows() / quant_data.div_factor, A.cols());
          ColMatrixUInt8 B_quantized((B.rows() - 1) / quant_data.div_factor, B.cols());
          Vector A_correction(A.cols() * 2);
          Vector B_correction(B.cols() * 2);

          Vector A_fix = A.row(0);
          A.row(0).array() *= 0;
          Vector B_fix = B.row(0);

          A_correction(Eigen::seqN(A.cols(), A.cols())) = A_fix;
          B_correction(Eigen::seqN(B.cols(), B.cols())) = B_fix;

          quant_data.quantize_matrix_A_unsigned(A, A_quantized.data(), A_correction.data());
          quant_data.quantize_matrix_B_unsigned(B, B_quantized.data(), B_correction.data());

          _A[i] = std::move(A_quantized);
          _B[i] = std::move(B_quantized);
          _A_corrections[i] = std::move(A_correction);
          _B_corrections[i] = std::move(B_correction);
        });
  }

 private:
  Lorann() = default; /* default constructor should only be used for serialization */

  void select_nearest_clusters(const VectorInt8 &query_quantized, const float quantization_factor,
                               const float correction, const float compensation, int k,
                               int *out) const {
    ColVector dists(_centroids_quantized.cols());
    quant_query.quantized_matvec_product_A(_centroids_quantized, query_quantized,
                                           _centroid_correction, quantization_factor, correction,
                                           compensation, dists.data());
    if (_distance == L2)
      add_inplace(_global_centroid_norms.data(), dists.data(), _global_centroid_norms.size());
    select_k<float>(k, out, _centroids_quantized.cols(), NULL, dists.data());
  }

  friend class cereal::access;

  template <class Archive>
  void save(Archive &ar) const {
    ar(cereal::base_class<LorannBase<T>>(this), _global_transform, _centroids_quantized,
       _centroid_correction, _A, _B, _A_corrections, _B_corrections, _cluster_norms);
  }

  template <class Archive>
  void load(Archive &ar) {
    ar(cereal::base_class<LorannBase<T>>(this), _global_transform, _centroids_quantized,
       _centroid_correction, _A, _B, _A_corrections, _B_corrections, _cluster_norms);
  }

  DataQuantizer quant_data;
  QueryQuantizer quant_query;

  RowMatrix _global_transform;
  ColMatrixUInt8 _centroids_quantized;
  Vector _centroid_correction;

  std::vector<ColMatrixUInt8> _A;
  std::vector<ColMatrixUInt8> _B;
  std::vector<Vector> _A_corrections;
  std::vector<Vector> _B_corrections;
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

#define REGISTER_LORANN_TYPES(DataType, TypePrefix)                                       \
  typedef Lorann::Lorann<DataType, Lorann::SQ4Quantizer, Lorann::SQ4Quantizer>            \
      TypePrefix##_sq4sq4;                                                                \
  typedef Lorann::Lorann<DataType, Lorann::SQ4Quantizer, Lorann::SQ8Quantizer>            \
      TypePrefix##_sq4sq8;                                                                \
  typedef Lorann::Lorann<DataType, Lorann::SQ8Quantizer, Lorann::SQ4Quantizer>            \
      TypePrefix##_sq8sq4;                                                                \
  typedef Lorann::Lorann<DataType, Lorann::SQ8Quantizer, Lorann::SQ8Quantizer>            \
      TypePrefix##_sq8sq8;                                                                \
                                                                                          \
  CEREAL_REGISTER_TYPE(TypePrefix##_sq4sq4)                                               \
  CEREAL_REGISTER_POLYMORPHIC_RELATION(Lorann::LorannBase<DataType>, TypePrefix##_sq4sq4) \
                                                                                          \
  CEREAL_REGISTER_TYPE(TypePrefix##_sq4sq8)                                               \
  CEREAL_REGISTER_POLYMORPHIC_RELATION(Lorann::LorannBase<DataType>, TypePrefix##_sq4sq8) \
                                                                                          \
  CEREAL_REGISTER_TYPE(TypePrefix##_sq8sq4)                                               \
  CEREAL_REGISTER_POLYMORPHIC_RELATION(Lorann::LorannBase<DataType>, TypePrefix##_sq8sq4) \
                                                                                          \
  CEREAL_REGISTER_TYPE(TypePrefix##_sq8sq8)                                               \
  CEREAL_REGISTER_POLYMORPHIC_RELATION(Lorann::LorannBase<DataType>, TypePrefix##_sq8sq8)

REGISTER_LORANN_TYPES(float, lorann_fp32)
#if SIMSIMD_NATIVE_F16
REGISTER_LORANN_TYPES(simsimd_f16_t, lorann_fp16)
#endif
#if SIMSIMD_NATIVE_BF16
REGISTER_LORANN_TYPES(simsimd_bf16_t, lorann_bf16)
#endif
REGISTER_LORANN_TYPES(uint8_t, lorann_u8)
REGISTER_LORANN_TYPES(Lorann::BinaryType, lorann_b)
