#pragma once

#include <simsimd/simsimd.h>

#include <cstring>
#include <memory>
#include <stdexcept>
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

    const int width = d / detail::Traits<T>::dim_divisor;
    check_for_nan(data, m, width);

    if (!copy) {
      _data = std::unique_ptr<T[], void (*)(T *)>(data, [](T *) { /* no-op for external data */ });
    } else {
      _data = make_aligned_array<T>(m * width);
      std::memcpy(_data.get(), data, m * width * sizeof(T));
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
    const int width = _dim / detail::Traits<T>::dim_divisor;

    if (_distance == L2) {
      for (int i = 0; i < _n_samples; ++i) {
        dist[i] = detail::Traits<T>::squared_euclidean(q, data_ptr + i * width, width);
      }
    } else {
      for (int i = 0; i < _n_samples; ++i) {
        dist[i] = -detail::Traits<T>::dot_product(q, data_ptr + i * width, width);
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
    const int width = _dim / detail::Traits<T>::dim_divisor;

    if (_distance == L2) {
      for (int i = 0; i < n; ++i) {
        dist[i] = detail::Traits<T>::squared_euclidean(q, data_ptr + in[i] * width, width);
      }
    } else {
      for (int i = 0; i < n; ++i) {
        dist[i] = -detail::Traits<T>::dot_product(q, data_ptr + in[i] * width, width);
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
                                           const bool approximate, const bool verbose,
                                           int num_threads) {
    const int to_sample = SAMPLED_POINTS_PER_CLUSTER * _n_clusters;
    if (!_balanced && approximate && to_sample < 0.5f * n) {
      /* sample points for k-means */
      const RowMatrix sampled =
          sample_rows(Eigen::Map<const RowMatrix>(data, n, _global_dim), to_sample);
      (void)global_clustering.train(sampled.data(), sampled.rows(), sampled.cols(), verbose,
                                    num_threads);
      _cluster_map = global_clustering.assign(data, n, 1);
    } else {
      _cluster_map = global_clustering.train(data, n, _global_dim, verbose, num_threads);
    }

    return global_clustering.assign(train_data, train_n, _train_size);
  }

  friend class cereal::access;

  template <class Archive>
  void save(Archive &ar) const {
    const int width = _dim / detail::Traits<T>::dim_divisor;

    ar(_n_samples);
    ar(_dim);
    ar(cereal::binary_data(_data.get(), sizeof(T) * _n_samples * width), _n_clusters, _global_dim,
       _max_rank, _train_size, static_cast<int>(_distance), _balanced, _cluster_map,
       _global_centroid_norms, _data_norms);
  }

  template <class Archive>
  void load(Archive &ar) {
    ar(_n_samples);
    ar(_dim);

    const int width = _dim / detail::Traits<T>::dim_divisor;
    _data = make_aligned_array<T>(_n_samples * width);

    int distance_tmp;
    ar(cereal::binary_data(_data.get(), sizeof(T) * _n_samples * width), _n_clusters, _global_dim,
       _max_rank, _train_size, distance_tmp, _balanced, _cluster_map, _global_centroid_norms,
       _data_norms);

    _distance = static_cast<Distance>(distance_tmp);
    _cluster_sizes = Eigen::VectorXi(_n_clusters);

    for (int i = 0; i < _n_clusters; ++i) {
      _cluster_sizes(i) = static_cast<int>(_cluster_map[i].size());
    }
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