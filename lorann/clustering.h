#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "utils.h"

#define EPS (1 / 1024.)
#define PARTLY_REMAINING_FACTOR 0.15
#define PENALTY_FACTOR 2.5

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
   * @param euclidean Whether to use Euclidean distance instead of (negative) inner product as the
   * dissimilarity measure. Defaults to false.
   * @param balanced Whether to ensure clusters are balanced using an efficient balanced k-means
   * algorithm. Defaults to false.
   * @param max_balance_diff The maximum allowed difference in cluster sizes for balanced
   * clustering. Used only if balanced = true. Defaults to 16.
   * @param verbose Whether to enable verbose output. Defaults to false.
   */
  KMeans(int n_clusters, int iters = 25, bool euclidean = false, bool balanced = false,
         int max_balance_diff = 16, bool verbose = false)
      : _iters(iters),
        _n_clusters(n_clusters),
        _euclidean(euclidean),
        _balanced(balanced),
        _max_balance_diff(max_balance_diff),
        _verbose(verbose),
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
   * @param num_threads Number of CPU threads to use (set to -1 to use all cores)
   * @return std::vector<std::vector<int>> Clustering assignments as a vector of
   * vectors where each vector contains the ids of the points assigned to the
   * corresponding cluster
   */
  std::vector<std::vector<int>> train(const float *data, int n, int m, int num_threads = -1) {
    LORANN_ENSURE_POSITIVE(n);
    LORANN_ENSURE_POSITIVE(m);

    if (_trained) {
      throw std::runtime_error("The clustering has already been trained");
    }

    if (n < _n_clusters) {
      throw std::runtime_error(
          "The number of points should be at least as large as the number of clusters");
    }

#ifdef _OPENMP
    if (num_threads <= 0) {
      num_threads = omp_get_max_threads();
    }
#endif

    Eigen::Map<const RowMatrix> train_mat = Eigen::Map<const RowMatrix>(data, n, m);

    _assignments = std::vector<int>(n);
    _cluster_sizes = Vector(_n_clusters);
    Vector data_norms = train_mat.rowwise().squaredNorm();

    _centroids = sample_rows(train_mat, _n_clusters);
    postprocess_centroids();

    for (int i = 0; i < _iters; ++i) {
      assign_clusters(train_mat, data_norms, num_threads);
      update_centroids(train_mat);
      split_clusters(train_mat);
      postprocess_centroids();
      if (_verbose)
        std::cout << "Iteration " << i + 1 << "/" << _iters << " | Cost : " << cost(train_mat)
                  << std::endl;
    }

    assign_clusters(train_mat, data_norms, num_threads);

    if (_balanced) {
      if (_verbose) std::cout << "Balancing" << std::endl;
      balance(train_mat, data_norms);
    }

    std::vector<std::vector<int>> res(_n_clusters);
    for (int i = 0; i < n; ++i) {
      res[_assignments[i]].push_back(i);
    }

    _trained = true;

    std::vector<int>().swap(_assignments);
    return res;
  }

  /**
   * @brief Assign given data points to their k nearest clusters.
   *
   * NOTE: The dimensionality of the data should match the dimensionality of the
   * data that the clustering was trained on.
   *
   * @param data The data matrix
   * @param n The number of data points (rows) in the data matrix
   * @param k The number of clusters each point is assigned to
   * @return std::vector<std::vector<int>> Clustering assignments as a vector of
   * vectors where each vector contains the ids of the points assigned to the
   * corresponding cluster
   */
  std::vector<std::vector<int>> assign(const float *data, int n, int k) const {
    LORANN_ENSURE_POSITIVE(n);
    LORANN_ENSURE_POSITIVE(k);

    if (!_trained) {
      throw std::runtime_error("The clustering has not been trained");
    }

    Vector centroid_norms = _centroids.rowwise().squaredNorm();

    std::vector<int> idxs = std::vector<int>(n * k);
    for (int i = 0; i < n; ++i) {
      Eigen::Map<const Vector> q =
          Eigen::Map<const Vector>(data + i * _centroids.cols(), _centroids.cols());
      Vector dots = _centroids * q.transpose();
      Vector dists;
      if (_euclidean) {
        dists = -2. * dots + centroid_norms;
      } else {
        dists = -dots;
      }

      select_k(k, &idxs[i * k], dists.size(), NULL, dists.data());
    }

    std::vector<std::vector<int>> res(_n_clusters);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < k; ++j) {
        res[idxs[i * k + j]].push_back(i);
      }
    }

    return res;
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
   * @brief Get whether Euclidean distance is used as the dissimilarity measure
   *
   * @return bool
   */
  inline bool get_euclidean() const { return _euclidean; }

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
  inline RowMatrix get_centroids() const {
    if (!_trained) {
      throw std::runtime_error("The clustering has not been trained");
    }

    return _centroids;
  }

 private:
  /* Assign each data point to its nearest cluster */
  void assign_clusters(const Eigen::Map<const RowMatrix> &train_mat, const Vector &data_norms,
                       const int num_threads) {
    if (_euclidean) {
      Eigen::VectorXf centroid_norms = _centroids.rowwise().squaredNorm();
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
      for (int i = 0; i < train_mat.rows(); ++i) {
        Eigen::VectorXf dot_products = train_mat.row(i) * _centroids.transpose();
        float min_dist = std::numeric_limits<float>::max();
        for (int j = 0; j < _n_clusters; ++j) {
          float dist = data_norms(i) - 2 * dot_products(j) + centroid_norms(j);
          if (dist < min_dist) {
            min_dist = dist;
            _assignments[i] = j;
          }
        }
      }
    } else {
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
      for (int i = 0; i < train_mat.rows(); ++i) {
        Eigen::VectorXf dot_products = train_mat.row(i) * _centroids.transpose();
        float max_similarity = std::numeric_limits<float>::min();
        for (int j = 0; j < _n_clusters; ++j) {
          float similarity = dot_products(j);
          if (similarity > max_similarity) {
            max_similarity = similarity;
            _assignments[i] = j;
          }
        }
      }
    }

    std::memset(_cluster_sizes.data(), 0, _n_clusters * sizeof(float));
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
    if (!_euclidean) {
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

    if (_euclidean) {
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
  void balance(const Eigen::Map<const RowMatrix> &train_mat, const Vector &data_norms) {
    RowMatrix unnormalized_centroids = RowMatrix::Zero(_n_clusters, train_mat.cols());
    Vector centroid_norms = _centroids.rowwise().squaredNorm();

    for (int i = 0; i < train_mat.rows(); ++i) {
      unnormalized_centroids.row(_assignments[i]) += train_mat.row(i);
    }

    float n_min = _cluster_sizes.minCoeff();
    float n_max = _cluster_sizes.maxCoeff();

    constexpr float partly_remaining_factor = PARTLY_REMAINING_FACTOR;
    constexpr float penalty_factor = PENALTY_FACTOR;

    int iters = 0;
    float p_now = 0;
    float p_next = std::numeric_limits<float>::max();

    while (n_max - n_min > 0.5 + _max_balance_diff) {
      for (int i = 0; i < train_mat.rows(); ++i) {
        int old = _assignments[i];
        float n_old = _cluster_sizes[old];
        unnormalized_centroids.row(old) -= train_mat.row(i);

        if (n_old > 0) {
          _centroids.row(old) = unnormalized_centroids.row(old).array() / (n_old - 1);
          if (_euclidean) {
            centroid_norms(old) = _centroids.row(old).squaredNorm();
          } else {
            _centroids.row(old).array() /= _centroids.row(old).norm();
          }
        }

        _cluster_sizes[old] = _cluster_sizes[old] - 1 + partly_remaining_factor;

        Vector dists;
        if (_euclidean) {
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

        if (_euclidean) {
          centroid_norms(minIndex) = _centroids.row(minIndex).squaredNorm();
        } else {
          _centroids.row(minIndex).array() /= _centroids.row(minIndex).norm();
        }

        _cluster_sizes[old] -= partly_remaining_factor;
        _assignments[i] = minIndex;
      }

      n_min = _cluster_sizes.minCoeff();
      n_max = _cluster_sizes.maxCoeff();
      p_now = penalty_factor * p_next;
      p_next = std::numeric_limits<float>::max();

      ++iters;

      if (_verbose) {
        std::cout << "Iteration " << iters << " | Cost : " << cost(train_mat)
                  << " | Max diff : " << n_max - n_min << std::endl;
      }
    }
  }

  RowMatrix _centroids;
  std::vector<int> _assignments;
  Vector _cluster_sizes;

  const int _iters;
  const int _n_clusters;
  const bool _euclidean;
  const bool _balanced;
  const int _max_balance_diff;
  const bool _verbose;
  bool _trained;
};

}  // namespace Lorann