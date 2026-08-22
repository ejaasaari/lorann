#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <simsimd/simsimd.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "clustering.h"
#include "detail/detail.h"
#include "serialization.h"
#include "utils.h"

#define KMEANS_ITERATIONS 10
#define BALANCED_KMEANS_MAX_DIFF 32
#define BALANCED_KMEANS_PENALTY 1.4
#define SAMPLED_POINTS_PER_CLUSTER 256
#define GLOBAL_DIM_REDUCTION_SAMPLES 16384

namespace Lorann {

template <typename T>
class LorannBase {
 public:
  LorannBase(T *data, int m, int d, int n_clusters, int global_dim, int rank, int train_size,
             Distance distance, bool balanced, bool copy)
      : _data(nullptr, [](T *) { /* will be set properly below */ }),
        _n_samples(m),
        _dim(d),
        _n_clusters(n_clusters),
        _global_dim(global_dim <= 0 ? d : std::min(global_dim, d)),
        _max_rank(std::min(rank, d)),
        _train_size(train_size),
        _distance(distance),
        _balanced(balanced),
        _copy(copy) {
    if (d < 64) {
      throw std::invalid_argument(
          "LoRANN is meant for high-dimensional data: the dimensionality should be at least 64.");
    }

    LORANN_ENSURE_POSITIVE(m);
    LORANN_ENSURE_POSITIVE(n_clusters);
    LORANN_ENSURE_POSITIVE(rank);
    LORANN_ENSURE_POSITIVE(train_size);

    const std::size_t width = static_cast<std::size_t>(d) / detail::Traits<T>::dim_divisor;
    check_for_nan(data, m, width);

    if (!copy) {
      _data = std::unique_ptr<T[], void (*)(T *)>(data, [](T *) { /* no-op for external data */ });
    } else {
      const std::size_t count = static_cast<std::size_t>(m) * width;
      _data = make_aligned_array<T>(count);
      std::memcpy(_data.get(), data, count * sizeof(T));
    }
  }

  /**
   * @brief Get the number of samples in the index.
   *
   * @return int
   */
  inline int get_n_samples() const { return _n_samples; }

  /**
   * @brief Get the dimensionality of the vectors in the index.
   *
   * @return int
   */
  inline int get_dim() const { return _dim; }

  /**
   * @brief Get the number of clusters.
   *
   * @return int
   */
  inline int get_n_clusters() const { return _n_clusters; }

  /**
   * @brief Compute the dissimilarity between two vectors.
   *
   * @param u First vector
   * @param v Second vector

   * @return float The dissimilarity
   */
  inline lorann_dist_t get_dissimilarity(const T *u, const T *v) {
    const int width = _dim / detail::Traits<T>::dim_divisor;

    if (_distance == L2) {
      return detail::Traits<T>::squared_euclidean(u, v, width);
    } else {
      return -detail::Traits<T>::dot_product(u, v, width);
    }
  }

  inline int get_type_marker() { return detail::Traits<T>::type_marker; }

  /**
   * @brief Build the index.
   *
   * @param approximate Whether to turn on various approximations during index construction.
   * Defaults to true. Setting approximate to false slows down the index construction but can
   * slightly increase the recall, especially if no exact re-ranking is used in the query phase.
   * @param verbose Whether to use verbose output for index construction. Defaults to false.
   * @param num_threads Number of CPU threads to use (set to -1 to use all cores)
   */
  void build(const bool approximate = true, const bool verbose = false, int num_threads = -1) {
    build(_data.get(), _n_samples, approximate, verbose, num_threads);
  }

  virtual void build(const T *query_data, const int query_n, const bool approximate,
                     const bool verbose, int num_threads) {}

  virtual void search(const T *data, const int k, const int clusters_to_search,
                      const int points_to_rerank, int *idx_out,
                      lorann_dist_t *dist_out = nullptr) const {}

  virtual ~LorannBase() {}

  /**
   * @brief Perform exact k-nn search using the index.
   *
   * @param q The query vector (dimension must match the index data dimension)
   * @param k The number of nearest neighbors
   * @param out The index output array of length k
   * @param dist_out The (optional) distance output array of length k
   */
  void exact_search(const T *q, int k, int *out, lorann_dist_t *dist_out = nullptr) const {
    DistVector dist(_n_samples);

    const T *data_ptr = _data.get();
    const std::size_t width = static_cast<std::size_t>(_dim) / detail::Traits<T>::dim_divisor;

    if (_distance == L2) {
      for (int i = 0; i < _n_samples; ++i) {
        const std::size_t offset = static_cast<std::size_t>(i) * width;
        dist[i] = detail::Traits<T>::squared_euclidean(q, data_ptr + offset, width);
      }
    } else {
      for (int i = 0; i < _n_samples; ++i) {
        const std::size_t offset = static_cast<std::size_t>(i) * width;
        dist[i] = -detail::Traits<T>::dot_product(q, data_ptr + offset, width);
      }
    }

    /* optimization for the special case k = 1 */
    if (k == 1) {
      Eigen::MatrixXf::Index index;
      dist.minCoeff(&index);
      out[0] = index;
      if (dist_out) dist_out[0] = dist[index];
      return;
    }

    const int final_k = k;
    if (k > _n_samples) {
      k = _n_samples;
    }

    select_k<lorann_dist_t>(k, out, _n_samples, NULL, dist.data(), dist_out, true);
    for (int i = k; i < final_k; ++i) {
      out[i] = -1;
      if (dist_out) dist_out[i] = std::numeric_limits<lorann_dist_t>::infinity();
    }
  }

 protected:
  /* default constructor should only be used for serialization */
  LorannBase() : _data(nullptr, [](T *) {}) {}

  struct BuildContext {
    MappedMatrix train_mat;
    MappedMatrix query_mat;
    int num_threads;
    bool queries_are_train;
  };

  struct ProjectedBuildData {
    template <typename Transform>
    ProjectedBuildData(const BuildContext &build_context,
                       const Eigen::MatrixBase<Transform> &transform)
        : queries_are_train_(build_context.queries_are_train) {
      auto project = [&](const auto &source, RowMatrix &destination) {
        destination.resize(source.rows(), transform.cols());
#ifdef _OPENMP
        if (build_context.num_threads > 1 && source.rows() >= build_context.num_threads * 64) {
#pragma omp parallel num_threads(build_context.num_threads)
          {
            const Eigen::Index thread = omp_get_thread_num();
            const Eigen::Index threads = omp_get_num_threads();
            const Eigen::Index begin = source.rows() * thread / threads;
            const Eigen::Index end = source.rows() * (thread + 1) / threads;
            destination.middleRows(begin, end - begin).noalias() =
                source.middleRows(begin, end - begin) * transform;
          }
          return;
        }
#endif
        destination.noalias() = source * transform;
      };

      project(build_context.train_mat.view, train_mat_);
      if (!queries_are_train_) {
        project(build_context.query_mat.view, query_mat_);
      }
    }

    const RowMatrix &train_rows() const { return train_mat_; }

    const RowMatrix &query_rows() const { return queries_are_train_ ? train_mat_ : query_mat_; }

    void retain(const std::pair<bool, bool> &usage) {
      uses_train_ = usage.first;
      uses_queries_ = usage.second;
      if (queries_are_train_) {
        if (!uses_train_ && !uses_queries_) train_mat_.resize(0, 0);
      } else {
        if (!uses_train_) train_mat_.resize(0, 0);
        if (!uses_queries_) query_mat_.resize(0, 0);
      }
    }

    const RowMatrix *used_train_rows() const { return uses_train_ ? &train_mat_ : nullptr; }

    const RowMatrix *used_query_rows() const {
      if (!uses_queries_) return nullptr;
      return queries_are_train_ ? &train_mat_ : &query_mat_;
    }

   private:
    RowMatrix train_mat_;
    RowMatrix query_mat_;
    bool queries_are_train_;
    bool uses_train_ = true;
    bool uses_queries_ = true;
  };

  BuildContext prepare_build(const T *query_data, const int query_n, int num_threads) const {
    LORANN_ENSURE_POSITIVE(query_n);

#ifdef _OPENMP
    if (num_threads <= 0) {
      num_threads = omp_get_max_threads();
    }
#endif

    MappedMatrix train_mat = detail::Traits<T>::to_float_matrix(_data.get(), _n_samples, _dim);
    const bool queries_are_train = query_data == _data.get() && query_n == _n_samples;
    if (queries_are_train) {
      return {train_mat, train_mat, num_threads, true};
    }

    return {std::move(train_mat), detail::Traits<T>::to_float_matrix(query_data, query_n, _dim),
            num_threads, false};
  }

  KMeans make_global_clustering(const bool approximate) const {
    int to_sample = SAMPLED_POINTS_PER_CLUSTER;
    if (_balanced || !approximate ||
        static_cast<std::int64_t>(to_sample) * static_cast<std::int64_t>(_n_clusters) >
            0.5 * static_cast<double>(_n_samples)) {
      to_sample = -1;
    }

    return KMeans(_n_clusters, KMEANS_ITERATIONS, to_sample, _distance, _balanced,
                  BALANCED_KMEANS_MAX_DIFF, BALANCED_KMEANS_PENALTY);
  }

  std::vector<std::vector<int>> cluster_build_data(KMeans &global_clustering,
                                                   const BuildContext &build_context,
                                                   const bool verbose) {
    const auto &train_mat = build_context.train_mat.view;
    const auto &query_mat = build_context.query_mat.view;
    return clustering(global_clustering, train_mat.data(), train_mat.rows(), query_mat.data(),
                      query_mat.rows(), verbose, build_context.num_threads);
  }

  std::vector<std::vector<int>> cluster_reduced_data(KMeans &global_clustering,
                                                     const BuildContext &build_context,
                                                     const RowMatrix &reduced_train_mat,
                                                     const RowMatrix &reduced_query_mat,
                                                     const bool verbose) {
    return clustering(global_clustering, reduced_train_mat.data(), reduced_train_mat.rows(),
                      reduced_query_mat.data(), reduced_query_mat.rows(), verbose,
                      build_context.num_threads);
  }

  bool uses_point_rows_as_queries(const BuildContext &build_context,
                                  const std::vector<std::vector<int>> &cluster_train_map,
                                  const int cluster) const {
    const bool has_enough_training_queries =
        cluster_train_map[cluster].size() >= _cluster_map[cluster].size();
    return !has_enough_training_queries || (_train_size == 1 && build_context.queries_are_train);
  }

  std::pair<bool, bool> projected_data_usage(const BuildContext &build_context,
                                             const std::vector<std::vector<int>> &cluster_train_map,
                                             const bool approximate) const {
    bool uses_projected_train = false;
    bool uses_projected_queries = false;
    for (int i = 0; i < _n_clusters; ++i) {
      if (_cluster_map[i].empty()) continue;

      const bool queries_are_points =
          uses_point_rows_as_queries(build_context, cluster_train_map, i);
      uses_projected_train |= approximate || queries_are_points;
      uses_projected_queries |= !queries_are_points;
    }

    return {uses_projected_train, uses_projected_queries};
  }

  void initialize_distance_data(const Eigen::Map<const RowMatrix> &train_mat,
                                const RowMatrix &centroid_mat) {
    if (_distance == L2) {
      _global_centroid_norms = centroid_mat.rowwise().squaredNorm();
      _data_norms = train_mat.rowwise().squaredNorm();
    }
  }

  template <typename ModelBuilder>
  void build_cluster_models(const BuildContext &build_context,
                            const std::vector<std::vector<int>> &cluster_train_map,
                            const ProjectedBuildData *projected_data, const bool approximate,
                            const bool verbose, std::vector<Vector> &cluster_norms,
                            ModelBuilder model_builder) {
    const auto &train_mat = build_context.train_mat.view;
    const auto &query_mat = build_context.query_mat.view;
#ifdef _OPENMP
    const int num_threads = build_context.num_threads;
#endif

    if (_distance == L2) {
      cluster_norms.resize(_n_clusters);
    }

    int completed_clusters = 0;
    auto report_progress = [&]() {
      if (!verbose) return;
#ifdef _OPENMP
#pragma omp critical(lorann_model_build_progress)
#endif
      {
        ++completed_clusters;
        if (completed_clusters % 100 == 0 || completed_clusters == _n_clusters) {
          std::cout << "Cluster model build progress: " << completed_clusters << "/" << _n_clusters
                    << std::endl;
        }
      }
    };

#ifdef _OPENMP
#pragma omp parallel for schedule(guided) num_threads(num_threads)
#endif
    for (int i = 0; i < _n_clusters; ++i) {
      if (_cluster_map[i].empty()) {
        report_progress();
        continue;
      }

      if (_distance == L2) {
        cluster_norms[i] = _data_norms(_cluster_map[i]);
      }

      const bool queries_are_points =
          uses_point_rows_as_queries(build_context, cluster_train_map, i);
      const bool needs_original_data = projected_data == nullptr || !approximate;

      RowMatrix pts;
      RowMatrix training_queries;
      const RowMatrix *Q = nullptr;
      if (needs_original_data) {
        pts = train_mat(_cluster_map[i], Eigen::placeholders::all);
        if (queries_are_points) {
          Q = &pts;
        } else {
          training_queries = query_mat(cluster_train_map[i], Eigen::placeholders::all);
          Q = &training_queries;
        }
      }

      RowMatrix A;
      Eigen::MatrixXf V;

      /* The three model-building cases have the following direct formulations, where G is the
       * global projection, q is the number of training-query rows, m is the number of point rows,
       * s is the model feature dimension, k is the number of nonzero QR pivots, and r is the model
       * rank:
       *
       *   1. Transformed, approximate:
       *        beta_hat = (pts * G).transpose();
       *        Y_hat = (Q * G) * beta_hat;
       *
       *   2. Transformed, exact:
       *        X = Q * G;
       *        beta_hat = X.colPivHouseholderQr().solve(Q * pts.transpose());
       *        Y_hat = X * beta_hat;
       *
       *   3. No transform:
       *        beta_hat = pts.transpose();
       *        Y_hat = Q * pts.transpose();
       *
       * Only A = beta_hat * V and B = V.transpose() are needed by the finished model, where V
       * contains the leading right singular vectors of Y_hat. The code below therefore computes
       * A and V directly:
       *
       *   1. Let T = Q * G and F = pts * G, so Y_hat = T * F.transpose(). If T = U * R, the
       *      smaller R * F.transpose() has the same right singular vectors. Compute V from that
       *      product and form A = F.transpose() * V without materializing beta_hat or Y_hat. For
       *      tall T and F, forming and compressing the product costs O((q + m) * s^2), versus
       *      O(q * m * s) to form Y_hat, and randomized SVD works on s-by-m instead of q-by-m.
       *
       *   2. For the pivoted QR X * P = U * R, form
       *      S = U_k.transpose() * (Q * pts.transpose()), where U_k contains the nonzero columns
       *      of U. S has the same right singular vectors as Y_hat. Let R11 be the leading k-by-k
       *      nonsingular upper-triangular block of R, with k equal to the number of nonzero QR
       *      pivots. Solve R11 against only S * V (rank right-hand sides), then place each solved
       *      row into its original feature row of A according to the column permutation P; rows
       *      without a nonzero pivot remain zero. The triangular solve costs O(k^2 * r), versus
       *      O(k^2 * m) for all columns of beta_hat, and Y_hat is not formed.
       *
       *   3. Use the same product optimization as case 1 with T = Q and F = pts.
       */
      auto build_product_model = [&](const RowMatrix &projected_queries,
                                     const RowMatrix &projected_points) {
        const Eigen::Index feature_dim = projected_queries.cols();
        Eigen::MatrixXf singular_vector_input;

        if (projected_queries.rows() > feature_dim && projected_points.rows() > feature_dim) {
          /* If projected_queries = U * R, then projected_queries * projected_points^T and
           * R * projected_points^T have identical right singular vectors. Form R through the
           * smaller Gram matrix; fall back to Householder QR if it is ill-conditioned. */
          auto factor_projected_queries = [&]() {
            Eigen::MatrixXf query_gram = Eigen::MatrixXf::Zero(feature_dim, feature_dim);
            query_gram.selfadjointView<Eigen::Lower>().rankUpdate(projected_queries.transpose());
            Eigen::LLT<Eigen::MatrixXf, Eigen::Lower> llt(query_gram);
            if (llt.info() == Eigen::Success) {
              return Eigen::MatrixXf(llt.matrixL().transpose());
            }

            Eigen::HouseholderQR<Eigen::MatrixXf> qr(projected_queries);
            return Eigen::MatrixXf(
                qr.matrixQR().topRows(feature_dim).template triangularView<Eigen::Upper>());
          };

          Eigen::MatrixXf query_factor = factor_projected_queries();
          singular_vector_input.noalias() = query_factor * projected_points.transpose();
        } else {
          singular_vector_input.noalias() = projected_queries * projected_points.transpose();
        }

        V = compute_V(singular_vector_input, _max_rank);
        A.noalias() = projected_points.transpose() * V;
      };

      if (projected_data != nullptr && !approximate) {
        /* For X = Q * G, column-pivoted QR gives X * P = U * R. Eigen's
         * least-squares solution is beta_hat = P [R11^-1 U_k^T (Q * pts^T); 0]. Therefore
         * Y_hat has the same right singular vectors as
         *
         *   S = U_k^T (Q * pts^T),
         *
         * and A can be obtained by solving R11 against only S * V (rank RHS columns). */
        const RowMatrix *projected_query_source = queries_are_points
                                                      ? projected_data->used_train_rows()
                                                      : projected_data->used_query_rows();
        assert(projected_query_source != nullptr);
        const std::vector<int> &projected_query_indices =
            queries_are_points ? _cluster_map[i] : cluster_train_map[i];
        Eigen::MatrixXf X =
            (*projected_query_source)(projected_query_indices, Eigen::placeholders::all);
        Eigen::ColPivHouseholderQR<Eigen::MatrixXf> qr(X);
        const Eigen::Index nonzero_pivots = qr.nonzeroPivots();

        A = RowMatrix::Zero(X.cols(), _max_rank);
        V = Eigen::MatrixXf::Zero(pts.rows(), _max_rank);

        if (nonzero_pivots > 0) {
          const long double project_queries_work =
              static_cast<long double>(nonzero_pivots) * Q->cols() * (Q->rows() + pts.rows());
          const long double form_response_work =
              static_cast<long double>(Q->rows()) * pts.rows() * (Q->cols() + nonzero_pivots);

          Eigen::MatrixXf singular_vector_input;
          if (project_queries_work <= form_response_work) {
            Eigen::MatrixXf projected_queries = *Q;
            auto householder_q = qr.householderQ();
            householder_q.setLength(nonzero_pivots);
            projected_queries.applyOnTheLeft(householder_q.adjoint());
            singular_vector_input.noalias() =
                projected_queries.topRows(nonzero_pivots) * pts.transpose();
          } else {
            Eigen::MatrixXf response = (*Q) * pts.transpose();
            auto householder_q = qr.householderQ();
            householder_q.setLength(nonzero_pivots);
            response.applyOnTheLeft(householder_q.adjoint());
            singular_vector_input = response.topRows(nonzero_pivots);
          }

          V = compute_V(singular_vector_input, _max_rank);
          Eigen::MatrixXf solved = singular_vector_input * V;
          qr.matrixR()
              .topLeftCorner(nonzero_pivots, nonzero_pivots)
              .template triangularView<Eigen::Upper>()
              .solveInPlace(solved);

          for (Eigen::Index j = 0; j < nonzero_pivots; ++j) {
            A.row(qr.colsPermutation().indices().coeff(j)) = solved.row(j);
          }
        }
      } else if (projected_data != nullptr) {
        const RowMatrix *projected_train_rows = projected_data->used_train_rows();
        assert(projected_train_rows != nullptr);
        RowMatrix projected_points =
            (*projected_train_rows)(_cluster_map[i], Eigen::placeholders::all);
        if (queries_are_points) {
          build_product_model(projected_points, projected_points);
        } else {
          const RowMatrix *projected_query_rows = projected_data->used_query_rows();
          assert(projected_query_rows != nullptr);
          RowMatrix projected_queries =
              (*projected_query_rows)(cluster_train_map[i], Eigen::placeholders::all);
          build_product_model(projected_queries, projected_points);
        }
      } else {
        build_product_model(*Q, pts);
      }

      model_builder(i, std::move(A), V);
      report_progress();
    }

    update_cluster_sizes();
  }

  void select_final(const T *orig, const float *x, const int k, const int points_to_rerank,
                    const int s, const int *all_idxs, const float *all_distances, int *idx_out,
                    lorann_dist_t *dist_out) const {
    const int n_selected = std::min(std::max(k, points_to_rerank), s);

    if (points_to_rerank == 0) {
      select_k<float, lorann_dist_t>(n_selected, idx_out, s, all_idxs, all_distances, dist_out,
                                     true);

      if (dist_out && _distance == L2) {
        lorann_dist_t query_norm = detail::Traits<float>::dot_product(x, x, _dim);
        for (int i = 0; i < n_selected; ++i) {
          dist_out[i] += query_norm;
        }
      }

      for (int i = n_selected; i < k; ++i) {
        idx_out[i] = -1;
        if (dist_out) dist_out[i] = std::numeric_limits<lorann_dist_t>::infinity();
      }

      return;
    }

    std::vector<int> final_select(n_selected);
    select_k<float>(n_selected, final_select.data(), s, all_idxs, all_distances);
    reorder_exact(orig, k, final_select, idx_out, dist_out);
  }

  void reorder_exact(const T *q, int k, const std::vector<int> &in, int *out,
                     lorann_dist_t *dist_out = nullptr) const {
    const int n = in.size();
    DistVector dist(n);

    const T *data_ptr = _data.get();
    const std::size_t width = static_cast<std::size_t>(_dim) / detail::Traits<T>::dim_divisor;

    if constexpr (std::is_same_v<T, float>) {
      detail::compute_one_to_many(
          q, data_ptr, width, in.data(), static_cast<std::size_t>(n),
          _distance == L2 ? detail::OneToManyMetric::L2 : detail::OneToManyMetric::IP, dist.data());
    } else if (_distance == L2) {
      for (int i = 0; i < n; ++i) {
        const std::size_t offset = static_cast<std::size_t>(in[i]) * width;
        dist[i] = detail::Traits<T>::squared_euclidean(q, data_ptr + offset, width);
      }
    } else {
      for (int i = 0; i < n; ++i) {
        const std::size_t offset = static_cast<std::size_t>(in[i]) * width;
        dist[i] = -detail::Traits<T>::dot_product(q, data_ptr + offset, width);
      }
    }

    /* optimization for the special case k = 1 */
    if (k == 1) {
      Eigen::MatrixXf::Index index;
      dist.minCoeff(&index);
      out[0] = in[index];
      if (dist_out) dist_out[0] = dist[index];
      return;
    }

    const int final_k = k;
    if (k > n) {
      k = n;
    }

    select_k<lorann_dist_t>(k, out, in.size(), in.data(), dist.data(), dist_out, true);
    for (int i = k; i < final_k; ++i) {
      out[i] = -1;
      if (dist_out) dist_out[i] = std::numeric_limits<lorann_dist_t>::infinity();
    }
  }

  std::vector<std::vector<int>> clustering(KMeans &global_clustering, const float *data,
                                           const int n, const float *train_data, const int train_n,
                                           const bool verbose, int num_threads) {
    const bool combine_final_assignments =
        _train_size > 1 && train_data == data && train_n == n && !global_clustering.is_balanced();
    _cluster_map = global_clustering.train(data, n, _global_dim, verbose, num_threads,
                                           combine_final_assignments ? _train_size : 1);
    if (_train_size > 1) {
      if (combine_final_assignments) {
        return global_clustering.take_final_assignments();
      }
      return global_clustering.assign(train_data, train_n, _train_size, num_threads);
    } else {
      return _cluster_map;
    }
  }

  void update_cluster_sizes() {
    _cluster_sizes = Eigen::VectorXi(_n_clusters);
    for (int i = 0; i < _n_clusters; ++i) {
      _cluster_sizes(i) = static_cast<int>(_cluster_map[i].size());
    }
  }

  friend class cereal::access;

  template <class Archive>
  void save(Archive &ar) const {
    const std::size_t width = static_cast<std::size_t>(_dim) / detail::Traits<T>::dim_divisor;
    const std::size_t count = static_cast<std::size_t>(_n_samples) * width;

    ar(_n_samples);
    ar(_dim);
    ar(cereal::binary_data(_data.get(), count * sizeof(T)), _n_clusters, _global_dim, _max_rank,
       _train_size, static_cast<int>(_distance), _balanced, _cluster_map, _global_centroid_norms,
       _data_norms);
  }

  template <class Archive>
  void load(Archive &ar) {
    ar(_n_samples);
    ar(_dim);

    const std::size_t width = static_cast<std::size_t>(_dim) / detail::Traits<T>::dim_divisor;
    const std::size_t count = static_cast<std::size_t>(_n_samples) * width;
    _data = make_aligned_array<T>(count);

    int distance_tmp;
    ar(cereal::binary_data(_data.get(), count * sizeof(T)), _n_clusters, _global_dim, _max_rank,
       _train_size, distance_tmp, _balanced, _cluster_map, _global_centroid_norms, _data_norms);

    _distance = static_cast<Distance>(distance_tmp);
    update_cluster_sizes();
  }

  std::unique_ptr<T[], void (*)(T *)> _data;

  int _n_samples;
  int _dim;
  int _n_clusters;
  int _global_dim;
  int _max_rank; /* max rank (r) for the RRR parameter matrices */
  int _train_size;
  Distance _distance;
  bool _balanced;
  bool _copy;

  /* vector of points assigned to a cluster, for each cluster */
  std::vector<std::vector<int>> _cluster_map;

  Eigen::VectorXf _global_centroid_norms;
  Eigen::VectorXi _cluster_sizes;
  Vector _data_norms;
};

}  // namespace Lorann
