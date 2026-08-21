#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "utils.h"

#define EPS (1 / 1024.)

namespace Lorann {

class KMeans {
 public:
  /**
   * @brief Construct a new KMeans object.
   *
   * NOTE: The constructor does not perform the actual clustering.
   *
   * @param n_clusters The number of clusters (k)
   * @param iters The number of k-means iterations to perform. Defaults to 25.
   * @param samples_per_cluster Number of points to sample per cluster for k-means iterations
   * (total sample size = samples_per_cluster * n_clusters). K-means iterations use this subset
   * for speed, while final assignments use the full dataset. Set to <= 0 to disable sampling.
   * @param distance The distance measure to use. Either IP or L2. Defaults to IP.
   * @param balanced Whether to ensure clusters are balanced using an efficient balanced k-means
   * algorithm. Defaults to false.
   * @param max_balance_diff The maximum allowed difference in cluster sizes for balanced
   * clustering. Used only if balanced = true. Defaults to 16.
   * @param penalty_factor Penalty factor for balanced clustering. Higher values can be used
   * for faster clustering at the cost of clustering quality. Used only if balanced = True.
   * Defaults to 1.4.
   */
  KMeans(int n_clusters, int iters = 25, int samples_per_cluster = 256, Distance distance = IP,
         bool balanced = false, int max_balance_diff = 16, float penalty_factor = 1.4)
      : _iters(iters),
        _n_clusters(n_clusters),
        _samples_per_cluster(samples_per_cluster),
        _distance(distance),
        _balanced(balanced),
        _max_balance_diff(max_balance_diff),
        _penalty_factor(penalty_factor),
        _trained(false) {
    LORANN_ENSURE_POSITIVE(n_clusters);
    LORANN_ENSURE_POSITIVE(iters);
    LORANN_ENSURE_POSITIVE(max_balance_diff);
  }

  /**
   * @brief Performs the clustering on the provided data.
   *
   * @param data The data matrix
   * @param n Number of points (rows) in the data matrix
   * @param m Number of dimensions (cols) in the data matrix
   * @param verbose Whether to use verbose output. Defaults to false.
   * @param num_threads Number of CPU threads to use (set to -1 to use all cores)
   * @return std::vector<std::vector<int>> Clustering assignments as a vector of
   * vectors where each vector contains the ids of the points assigned to the
   * corresponding cluster
   */
  std::vector<std::vector<int>> train(const float *data, const int n, const int m,
                                      const bool verbose = false, int num_threads = -1,
                                      const int final_assignment_k = 1) {
    LORANN_ENSURE_POSITIVE(n);
    LORANN_ENSURE_POSITIVE(m);
    LORANN_ENSURE_POSITIVE(final_assignment_k);

    if (_trained) {
      throw std::runtime_error("The clustering has already been trained");
    }

    if (n < _n_clusters) {
      throw std::runtime_error(
          "The number of points should be at least as large as the number of clusters");
    }

    check_for_nan(data, n, m);

#ifdef _OPENMP
    if (num_threads <= 0) {
      num_threads = omp_get_max_threads();
    }
#endif

    Eigen::Map<const RowMatrix> train_mat = Eigen::Map<const RowMatrix>(data, n, m);

    _assignments = std::vector<int>(train_mat.rows());
    _cluster_sizes = Vector(_n_clusters);
    Vector data_norms;
    if (_distance == L2 && _balanced) {
      data_norms = train_mat.rowwise().squaredNorm();
    }

    _centroids = sample_rows(train_mat, _n_clusters);
    postprocess_centroids();

    RowMatrix sampled_training_mat;
    const float *iter_data_ptr = data;
    int iter_n = n;

    if (_samples_per_cluster > 0) {
      const std::int64_t sample_size =
          static_cast<std::int64_t>(_samples_per_cluster) * static_cast<std::int64_t>(_n_clusters);
      if (sample_size < n) {
        sampled_training_mat = sample_rows(train_mat, static_cast<int>(sample_size));
        iter_data_ptr = sampled_training_mat.data();
        iter_n = sampled_training_mat.rows();
        _assignments.resize(iter_n);
      }
    }

    Eigen::Map<const RowMatrix> iter_mat(iter_data_ptr, iter_n, m);

    if (verbose) {
      std::cout << "Clustering..." << std::endl;
    }

    for (int i = 0; i < _iters; ++i) {
      assign_clusters(iter_mat, num_threads);
      update_centroids(iter_mat);
      split_clusters(iter_mat);
      postprocess_centroids();
      if (verbose)
        std::cout << "Iteration " << i + 1 << "/" << _iters << " | Objective: " << cost(iter_mat)
                  << std::endl;
    }

    // After iterations, assign clusters using the full original data
    _assignments.resize(n);
    if (final_assignment_k > 1 && !_balanced) {
      _final_assignments =
          assign_multiple(train_mat, n, final_assignment_k, num_threads, _assignments.data());
      std::memset(_cluster_sizes.data(), 0, static_cast<std::size_t>(_n_clusters) * sizeof(float));
      for (const int assignment : _assignments) {
        _cluster_sizes[assignment] += 1;
      }
    } else {
      _final_assignments.clear();
      assign_clusters(train_mat, num_threads);
    }

    if (_balanced) {
      if (verbose) std::cout << "Balancing clusters..." << std::endl;
      balance(train_mat, data_norms, verbose);
    }

    std::vector<std::vector<int>> res(_n_clusters);
    for (int i = 0; i < _n_clusters; ++i) {
      res[i].reserve(static_cast<std::size_t>(_cluster_sizes[i]));
    }
    for (int i = 0; i < n; ++i) {
      res[_assignments[i]].push_back(i);
    }

    std::vector<int>().swap(_assignments);

    _trained = true;
    return res;
  }

  std::vector<std::vector<int>> take_final_assignments() { return std::move(_final_assignments); }

  /**
   * @brief Assign given data points to their k nearest clusters.
   *
   * NOTE: The dimensionality of the data should match the dimensionality of the
   * data that the clustering was trained on.
   *
   * @param data The data matrix
   * @param n The number of data points (rows) in the data matrix
   * @param k The number of clusters each point is assigned to
   * @param num_threads Number of CPU threads to use (set to -1 to use all cores)
   * @return std::vector<std::vector<int>> Clustering assignments as a vector of
   * vectors where each vector contains the ids of the points assigned to the
   * corresponding cluster
   */
  std::vector<std::vector<int>> assign(const float *data, int n, int k,
                                       int num_threads = -1) const {
    LORANN_ENSURE_POSITIVE(n);
    LORANN_ENSURE_POSITIVE(k);

    if (!_trained) {
      throw std::runtime_error("The clustering has not been trained");
    }

#ifdef _OPENMP
    if (num_threads <= 0) {
      num_threads = omp_get_max_threads();
    }
#endif

    Eigen::Map<const RowMatrix> data_mat(data, n, _centroids.cols());
    return assign_multiple(data_mat, n, k, num_threads, nullptr);
  }

  /**
   * @brief Get the number of clusters
   *
   * @return int
   */
  inline int get_n_clusters() const { return _n_clusters; }

  /**
   * @brief Get the number of k-means iterations
   *
   * @return int
   */
  inline int get_iters() const { return _iters; }

  /**
   * @brief Get whether balanced k-means is used
   *
   * @return bool
   */
  inline bool is_balanced() const { return _balanced; }

  /**
   * @brief Get the centroids
   *
   * @return float* Centroids as a float array of size n_clusters * dim
   */
  inline RowMatrix get_centroids() const & {
    if (!_trained) {
      throw std::runtime_error("The clustering has not been trained");
    }

    return _centroids;
  }

  inline RowMatrix get_centroids() && {
    if (!_trained) {
      throw std::runtime_error("The clustering has not been trained");
    }

    return std::move(_centroids);
  }

 private:
  std::vector<std::vector<int>> assign_multiple(const Eigen::Map<const RowMatrix> &data_mat,
                                                const int n, const int k, const int num_threads,
                                                int *primary_assignments) const {
    Vector centroid_norms;
    if (_distance == L2) {
      centroid_norms = _centroids.rowwise().squaredNorm();
    }

    const std::size_t idx_count = static_cast<std::size_t>(n) * static_cast<std::size_t>(k);
    std::vector<int> idxs(idx_count);
    const int block_rows = similarity_block_rows();
#ifdef _OPENMP
#pragma omp parallel for if (n > block_rows) num_threads(num_threads)
#endif
    for (int begin = 0; begin < n; begin += block_rows) {
      const int rows = begin + block_rows <= n ? block_rows : n - begin;
      RowMatrix distances = data_mat.middleRows(begin, rows) * _centroids.transpose();
      if (_distance == L2) {
        distances *= -2.0f;
        distances.rowwise() += centroid_norms;
      } else {
        distances *= -1.0f;
      }

      for (int row = 0; row < rows; ++row) {
        const std::size_t idx_offset =
            static_cast<std::size_t>(begin + row) * static_cast<std::size_t>(k);
        select_k(k, &idxs[idx_offset], _n_clusters, NULL, distances.row(row).data());
        if (primary_assignments != nullptr) {
          Eigen::Index primary;
          distances.row(row).minCoeff(&primary);
          primary_assignments[begin + row] = static_cast<int>(primary);
        }
      }
    }

    std::vector<std::size_t> cluster_sizes(static_cast<std::size_t>(_n_clusters));
    for (const int cluster : idxs) {
      ++cluster_sizes[static_cast<std::size_t>(cluster)];
    }

    std::vector<std::vector<int>> res(_n_clusters);
    for (int cluster = 0; cluster < _n_clusters; ++cluster) {
      res[cluster].reserve(cluster_sizes[static_cast<std::size_t>(cluster)]);
    }
    for (int i = 0; i < n; ++i) {
      const std::size_t idx_offset = static_cast<std::size_t>(i) * static_cast<std::size_t>(k);
      for (int j = 0; j < k; ++j) {
        res[idxs[idx_offset + static_cast<std::size_t>(j)]].push_back(i);
      }
    }

    return res;
  }

  int similarity_block_rows() const {
    constexpr std::size_t target_similarity_bytes = 1 << 20;
    constexpr int min_block_rows = 128;
    constexpr int max_block_rows = 1024;
    constexpr int block_granularity = 64;
    const std::size_t similarity_row_bytes = sizeof(float) * static_cast<std::size_t>(_n_clusters);
    int block_rows = static_cast<int>(target_similarity_bytes / similarity_row_bytes);
    block_rows = std::clamp(block_rows, min_block_rows, max_block_rows);
    return (block_rows / block_granularity) * block_granularity;
  }

  /* Assign each data point to its nearest cluster */
  void assign_clusters(const Eigen::Map<const RowMatrix> &train_mat, const int num_threads) {
    const int block_rows = similarity_block_rows();

    if (_distance == L2) {
      Eigen::VectorXf centroid_norms = _centroids.rowwise().squaredNorm();
#ifdef _OPENMP
#pragma omp parallel for schedule(guided) num_threads(num_threads)
#endif
      for (int begin = 0; begin < train_mat.rows(); begin += block_rows) {
        const int rows =
            begin + block_rows <= train_mat.rows() ? block_rows : train_mat.rows() - begin;
        RowMatrix dot_products = train_mat.middleRows(begin, rows) * _centroids.transpose();
        for (int row = 0; row < rows; ++row) {
          float min_dist = std::numeric_limits<float>::max();
          for (int j = 0; j < _n_clusters; ++j) {
            const float dist = -2 * dot_products(row, j) + centroid_norms(j);
            if (dist < min_dist) {
              min_dist = dist;
              _assignments[begin + row] = j;
            }
          }
        }
      }
    } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(guided) num_threads(num_threads)
#endif
      for (int begin = 0; begin < train_mat.rows(); begin += block_rows) {
        const int rows =
            begin + block_rows <= train_mat.rows() ? block_rows : train_mat.rows() - begin;
        RowMatrix dot_products = train_mat.middleRows(begin, rows) * _centroids.transpose();
        for (int row = 0; row < rows; ++row) {
          Eigen::Index max_index;
          dot_products.row(row).maxCoeff(&max_index);
          _assignments[begin + row] = static_cast<int>(max_index);
        }
      }
    }

    std::memset(_cluster_sizes.data(), 0, static_cast<std::size_t>(_n_clusters) * sizeof(float));
    for (int i = 0; i < train_mat.rows(); ++i) {
      _cluster_sizes[_assignments[i]] += 1;
    }
  }

  /* Re-compute cluster centroids */
  void update_centroids(const Eigen::Map<const RowMatrix> &train_mat) {
    _centroids = RowMatrix::Zero(_n_clusters, train_mat.cols());

    for (int i = 0; i < train_mat.rows(); ++i) {
      _centroids.row(_assignments[i]) += train_mat.row(i);
    }

    for (int j = 0; j < _n_clusters; ++j) {
      if (_cluster_sizes(j) > 0) {
        _centroids.row(j).array() /= _cluster_sizes(j);
      }
    }
  }

  void postprocess_centroids() {
    /* normalize centroids if using spherical k-means */
    if (_distance == IP) {
      for (int j = 0; j < _n_clusters; ++j) {
        _centroids.row(j).array() /= _centroids.row(j).norm();
      }
    }
  }

  /**
   * Handle empty clusters by splitting large clusters into two.
   *
   * Based on the Faiss implementation:
   * https://github.com/facebookresearch/faiss/blob/main/faiss/Clustering.cpp
   */
  void split_clusters(const Eigen::Map<const RowMatrix> &train_mat) {
    std::default_random_engine generator;

    for (int i = 0; i < _n_clusters; ++i) {
      if (_cluster_sizes[i] == 0) {
        int j;
        for (j = 0; true; j = (j + 1) % _n_clusters) {
          /* probability to pick this empty cluster for splitting */
          float p = (_cluster_sizes[j] - 1.0) / (float)(train_mat.rows() - _n_clusters);
          float r = std::uniform_real_distribution<float>(0, 1)(generator);
          if (r < p) {
            break;
          }
        }

        _centroids.row(i) = _centroids.row(j);

        /* small symmetric perturbation */
        for (int k = 0; k < train_mat.cols(); ++k) {
          if (k % 2 == 0) {
            _centroids(i, k) *= 1 + EPS;
            _centroids(j, k) *= 1 - EPS;
          } else {
            _centroids(i, k) *= 1 - EPS;
            _centroids(j, k) *= 1 + EPS;
          }
        }

        /* split evenly */
        _cluster_sizes[i] = _cluster_sizes[j] / 2;
        _cluster_sizes[j] -= _cluster_sizes[i];
      }
    }
  }

  float cost(const Eigen::Map<const RowMatrix> &train_mat) {
    float total_cost = 0;

    if (_distance == L2) {
      for (int i = 0; i < train_mat.rows(); ++i) {
        total_cost += (train_mat.row(i) - _centroids.row(_assignments[i])).squaredNorm();
      }
    } else {
      for (int i = 0; i < train_mat.rows(); ++i) {
        total_cost += train_mat.row(i).dot(_centroids.row(_assignments[i]));
      }
    }

    return total_cost / train_mat.rows();
  }

  /**
   * Balanced k-means algorithm
   *
   * A straightforward implementation of Algorithm 1 from the paper
   * Rieke de Maeyer, Sami Sieranoja, and Pasi Fränti. Balanced k-means
   * revisited. Applied Computing and Intelligence, 3(2):145–179, 2023.
   */
  void balance(const Eigen::Map<const RowMatrix> &train_mat, const Vector &data_norms,
               const bool verbose = false) {
    RowMatrix unnormalized_centroids = RowMatrix::Zero(_n_clusters, train_mat.cols());
    Vector centroid_norms = _centroids.rowwise().squaredNorm();

    for (int i = 0; i < train_mat.rows(); ++i) {
      unnormalized_centroids.row(_assignments[i]) += train_mat.row(i);
    }

    float n_min = _cluster_sizes.minCoeff();
    float n_max = _cluster_sizes.maxCoeff();

    int iters = 0;
    float p_now = 0;
    float p_next = std::numeric_limits<float>::max();

    float penalty_factor = _penalty_factor;

    while (n_max - n_min > 0.5 + _max_balance_diff) {
      for (int i = 0; i < train_mat.rows(); ++i) {
        int old = _assignments[i];
        float n_old = _cluster_sizes[old];
        unnormalized_centroids.row(old) -= train_mat.row(i);

        if (n_old > 0) {
          _centroids.row(old) = unnormalized_centroids.row(old).array() / (n_old - 1);
          if (_distance == L2) {
            centroid_norms(old) = _centroids.row(old).squaredNorm();
          } else {
            _centroids.row(old).array() /= _centroids.row(old).norm();
          }
        }

        _cluster_sizes[old] -= 1;

        Vector dists;
        if (_distance == L2) {
          Vector dots = train_mat.row(i) * _centroids.transpose();
          dists = (centroid_norms - 2 * dots).array() + data_norms(i);
        } else {
          dists = -train_mat.row(i) * _centroids.transpose();
        }

        Vector costs = dists + p_now * _cluster_sizes;
        Eigen::Index minIndex;
        costs.minCoeff(&minIndex);
        Vector penalties_1 = dists.array() - dists(old);
        Vector penalties_2 = _cluster_sizes[old] - _cluster_sizes.array();
        Vector penalties = penalties_1.array() / penalties_2.array();
        float min_p_value = std::numeric_limits<float>::max();

        for (int p = 0; p < _n_clusters; ++p) {
          if (_cluster_sizes[old] > _cluster_sizes[p] && penalties[p] < min_p_value) {
            min_p_value = penalties[p];
          }
        }

        if (p_now < min_p_value && min_p_value < p_next) {
          p_next = min_p_value;
        }

        _cluster_sizes[minIndex] += 1;

        unnormalized_centroids.row(minIndex) += train_mat.row(i);
        _centroids.row(minIndex) =
            unnormalized_centroids.row(minIndex).array() / _cluster_sizes[minIndex];

        if (_distance == L2) {
          centroid_norms(minIndex) = _centroids.row(minIndex).squaredNorm();
        } else {
          _centroids.row(minIndex).array() /= _centroids.row(minIndex).norm();
        }

        _assignments[i] = minIndex;
      }

      n_min = _cluster_sizes.minCoeff();
      n_max = _cluster_sizes.maxCoeff();

      p_now = penalty_factor * p_next;
      p_next = std::numeric_limits<float>::max();

      ++iters;

      if (verbose) {
        std::cout << "Iteration " << iters << " | Objective: " << cost(train_mat)
                  << " | Max diff: " << n_max - n_min << std::endl;
      }
    }
  }

  RowMatrix _centroids;
  std::vector<int> _assignments;
  std::vector<std::vector<int>> _final_assignments;
  Vector _cluster_sizes;

  const int _iters;
  const int _n_clusters;
  const int _samples_per_cluster;
  const Distance _distance;
  const bool _balanced;
  const int _max_balance_diff;
  const float _penalty_factor;
  bool _trained;
};

}  // namespace Lorann
