#pragma once

#include <stdexcept>
#include <vector>

#include "clustering.h"
#include "serialization.h"
#include "utils.h"

#define KMEANS_ITERATIONS 10
#define KMEANS_MAX_BALANCE_DIFF 16
#define SAMPLED_POINTS_PER_CLUSTER 256
#define GLOBAL_DIM_REDUCTION_SAMPLES 16384

namespace Lorann {

class LorannBase {
 public:
  LorannBase(float *data, int m, int d, int n_clusters, int global_dim, int rank, int train_size,
             bool euclidean, bool balanced)
      : _data(data),
        _n_samples(m),
        _dim(d),
        _n_clusters(n_clusters),
        _global_dim(global_dim <= 0 ? d : std::min(global_dim, d)),
        _max_rank(std::min(rank, d)),
        _train_size(train_size),
        _euclidean(euclidean),
        _balanced(balanced) {
    if (d < 64) {
      throw std::invalid_argument(
          "LoRANN is meant for high-dimensional data: the dimensionality should be at least 64.");
    }

    LORANN_ENSURE_POSITIVE(m);
    LORANN_ENSURE_POSITIVE(n_clusters);
    LORANN_ENSURE_POSITIVE(rank);
    LORANN_ENSURE_POSITIVE(train_size);
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
   * @brief Get whether the index uses the Euclidean distance as the dissimilarity measure.
   *
   * @return bool
   */
  inline bool get_euclidean() const { return _euclidean; }

  /**
   * @brief Get a pointer to a vector in the index.
   *
   * @param idx The index to the vector.
   * @param out The output buffer.
   */
  inline void get_vector(const int idx, float *out) {
    if (idx < 0 || idx >= _n_samples) {
      throw std::invalid_argument("Invalid index");
    }

    std::memcpy(out, _data + idx * _dim, _dim);
  }

  /**
   * @brief Compute the dissimilarity between two vectors.
   *
   * @param u First vector
   * @param v Second vector

   * @return float The dissimilarity
   */
  inline float get_dissimilarity(const float *u, const float *v) {
    if (_euclidean) {
      return squared_euclidean(u, v, _dim);
    } else {
      return -dot_product(u, v, _dim);
    }
  }

  /**
   * @brief Build the index.
   *
   * @param approximate Whether to turn on various approximations during index construction.
   * Defaults to true. Setting approximate to false slows down the index construction but can
   * slightly increase the recall, especially if no exact re-ranking is used in the query phase.
   * @param num_threads Number of CPU threads to use (set to -1 to use all cores)
   */
  void build(const bool approximate = true, int num_threads = -1) {
    build(_data, _n_samples, approximate, num_threads);
  }

  virtual void build(const float *query_data, const int query_n, const bool approximate,
                     int num_threads) {}

  virtual void search(const float *data, const int k, const int clusters_to_search,
                      const int points_to_rerank, int *idx_out, float *dist_out = nullptr) const {}

  virtual ~LorannBase() {}

  /**
   * @brief Perform exact k-nn search using the index.
   *
   * @param q The query vector (dimension must match the index data dimension)
   * @param k The number of nearest neighbors
   * @param out The index output array of length k
   * @param dist_out The (optional) distance output array of length k
   */
  void exact_search(const float *q, int k, int *out, float *dist_out = nullptr) const {
    Vector dist(_n_samples);

    const float *data_ptr = _data;
    if (_euclidean) {
      for (int i = 0; i < _n_samples; ++i) {
        dist[i] = squared_euclidean(q, data_ptr + i * _dim, _dim);
      }
    } else {
      for (int i = 0; i < _n_samples; ++i) {
        dist[i] = -dot_product(q, data_ptr + i * _dim, _dim);
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

    select_k(k, out, _n_samples, NULL, dist.data(), dist_out, true);
    for (int i = k; i < final_k; ++i) {
      out[i] = -1;
      if (dist_out) dist_out[i] = std::numeric_limits<float>::infinity();
    }
  }

 protected:
  /* default constructor should only be used for serialization */
  LorannBase() = default;

  void select_final(const float *x, const int k, const int points_to_rerank, const int s,
                    const int *all_idxs, const float *all_distances, int *idx_out,
                    float *dist_out) const {
    const int n_selected = std::min(std::max(k, points_to_rerank), s);

    if (points_to_rerank == 0) {
      select_k(n_selected, idx_out, s, all_idxs, all_distances, dist_out, true);

      if (dist_out && _euclidean) {
        float query_norm = 0;
        for (int i = 0; i < _dim; ++i) {
          query_norm += x[i] * x[i];
        }
        for (int i = 0; i < n_selected; ++i) {
          dist_out[i] += query_norm;
        }
      }

      for (int i = n_selected; i < k; ++i) {
        idx_out[i] = -1;
        if (dist_out) dist_out[i] = std::numeric_limits<float>::infinity();
      }

      return;
    }

    std::vector<int> final_select(n_selected);
    select_k(n_selected, final_select.data(), s, all_idxs, all_distances);
    reorder_exact(x, k, final_select, idx_out, dist_out);
  }

  void reorder_exact(const float *q, int k, const std::vector<int> &in, int *out,
                     float *dist_out = nullptr) const {
    const int n = in.size();
    Vector dist(n);

    const float *data_ptr = _data;
    if (_euclidean) {
      for (int i = 0; i < n; ++i) {
        dist[i] = squared_euclidean(q, data_ptr + in[i] * _dim, _dim);
      }
    } else {
      for (int i = 0; i < n; ++i) {
        dist[i] = dot_product(q, data_ptr + in[i] * _dim, _dim);
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

    select_k(k, out, in.size(), in.data(), dist.data(), dist_out, true);
    for (int i = k; i < final_k; ++i) {
      out[i] = -1;
      if (dist_out) dist_out[i] = std::numeric_limits<float>::infinity();
    }
  }

  std::vector<std::vector<int>> clustering(KMeans &global_clustering, const float *data,
                                           const int n, const float *train_data, const int train_n,
                                           const bool approximate, int num_threads) {
    const int to_sample = SAMPLED_POINTS_PER_CLUSTER * _n_clusters;
    if (!_balanced && approximate && to_sample < 0.5f * n) {
      /* sample points for k-means */
      const RowMatrix sampled =
          sample_rows(Eigen::Map<const RowMatrix>(data, n, _global_dim), to_sample);
      (void)global_clustering.train(sampled.data(), sampled.rows(), sampled.cols(), num_threads);
      _cluster_map = global_clustering.assign(data, n, 1);
    } else {
      _cluster_map = global_clustering.train(data, n, _global_dim, num_threads);
    }

    return global_clustering.assign(train_data, train_n, _train_size);
  }

  friend class cereal::access;

  template <class Archive>
  void save(Archive &ar) const {
    ar(_n_samples);
    ar(_dim);
    ar(cereal::binary_data(_data, sizeof(float) * _n_samples * _dim), _n_clusters, _global_dim,
       _max_rank, _train_size, _euclidean, _balanced, _cluster_map, _global_centroid_norms);
  }

  template <class Archive>
  void load(Archive &ar) {
    ar(_n_samples);
    ar(_dim);

    _owned_data = Vector(_n_samples * _dim);
    _data = _owned_data.data();

    ar(cereal::binary_data(_data, sizeof(float) * _n_samples * _dim), _n_clusters, _global_dim,
       _max_rank, _train_size, _euclidean, _balanced, _cluster_map, _global_centroid_norms);

    _cluster_sizes = Eigen::VectorXi(_n_clusters);

    for (int i = 0; i < _n_clusters; ++i) {
      _cluster_sizes(i) = static_cast<int>(_cluster_map[i].size());
    }

    if (_euclidean) {
      Eigen::Map<RowMatrix> train_mat(_data, _n_samples, _dim);
      _data_norms = train_mat.rowwise().squaredNorm();
    }
  }

  float *_data;
  Vector _owned_data;

  int _n_samples;
  int _dim;
  int _n_clusters;
  int _global_dim;
  int _max_rank; /* max rank (r) for the RRR parameter matrices */
  int _train_size;
  bool _euclidean;
  bool _balanced;

  /* vector of points assigned to a cluster, for each cluster */
  std::vector<std::vector<int>> _cluster_map;

  Eigen::VectorXf _global_centroid_norms;
  Eigen::VectorXi _cluster_sizes;
  Vector _data_norms;
};

}  // namespace Lorann