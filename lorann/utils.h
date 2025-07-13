#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <cstring>
#include <random>
#include <rsvd/Constants.hpp>
#include <rsvd/RandomizedSvd.hpp>
#include <unordered_set>
#include <vector>

#include "miniselect/pdqselect.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#else
#if defined(_MSC_VER) || defined(__MINGW32__)
#include <intrin.h>
#else
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__) || \
    defined(__SSE3__)
#if !defined(__riscv)
#include <immintrin.h>
#endif
#endif
#endif
#endif

#define RSVD_OVERSAMPLES 10
#define RSVD_N_ITER 4

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define LORANN_ENSURE_POSITIVE(x)                               \
  if ((x) <= 0) {                                               \
    throw std::invalid_argument("Value must be positive: " #x); \
  }

namespace Lorann {

enum Distance { 
  IP = 0,
  L2 = 1,

  HAMMING = L2
};

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ColMatrix;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrix;
typedef Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ColMatrixInt8;
typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ColMatrixUInt8;

typedef Eigen::VectorXf ColVector;
typedef Eigen::VectorXi ColVectorInt;
typedef Eigen::RowVectorXf Vector;
typedef Eigen::Matrix<int32_t, 1, Eigen::Dynamic> VectorInt;
typedef Eigen::Matrix<int8_t, 1, Eigen::Dynamic> VectorInt8;
typedef Eigen::Matrix<uint8_t, 1, Eigen::Dynamic> VectorUInt8;

struct MappedMatrix {
  Eigen::Map<const RowMatrix> view;
  std::shared_ptr<float[]> owner;

  MappedMatrix(const float *p, const Eigen::Index n, const Eigen::Index d,
               std::shared_ptr<float[]> own = {})
      : view{p, n, d}, owner{std::move(own)} {}
};

struct ArgsortComparator {
  const float *vals;
  bool operator()(const int a, const int b) const { return vals[a] < vals[b]; }
};

#if defined(__AVX2__)
static inline int32_t horizontal_add(__m128i const a) {
  const __m128i sum1 = _mm_hadd_epi32(a, a);
  const __m128i sum2 = _mm_hadd_epi32(sum1, sum1);
  return _mm_cvtsi128_si32(sum2);
}

static inline int32_t horizontal_add(__m256i const a) {
  const __m128i sum1 = _mm_add_epi32(_mm256_extracti128_si256(a, 1), _mm256_castsi256_si128(a));
  const __m128i sum2 = _mm_add_epi32(sum1, _mm_unpackhi_epi64(sum1, sum1));
  const __m128i sum3 = _mm_add_epi32(sum2, _mm_shuffle_epi32(sum2, 1));
  return (int32_t)_mm_cvtsi128_si32(sum3);
}
#endif

#if defined(__AVX2__)
static inline void add_inplace(const float *__restrict__ v, float *__restrict__ r, const size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v_vec = _mm256_loadu_ps(&v[i]);
    __m256 r_vec = _mm256_loadu_ps(&r[i]);
    __m256 result_vec = _mm256_add_ps(r_vec, v_vec);
    _mm256_storeu_ps(&r[i], result_vec);
  }

  for (; i < n; ++i) {
    r[i] += v[i];
  }
}
#else
static inline void add_inplace(const float *v, float *r, const size_t n) {
  for (size_t i = 0; i < n; ++i) {
    r[i] += v[i];
  }
}
#endif

static inline int nearest_int(const float fval) {
  float val = fval + 12582912.f;
  int i;
  memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

static inline float compute_quantization_factor(const float *v, const int len, const int bits) {
  /* compute the absmax of vector v */
  float absmax = 0.0f;
  for (int i = 0; i < len; ++i) {
    if (std::abs(v[i]) > absmax) {
      absmax = std::abs(v[i]);
    }
  }

  /* (2^(bits - 1) - 1) / absmax */
  return absmax > 0 ? ((1 << (bits - 1)) - 1) / absmax : 0;
}

static void select_k(const int k, int *labels, const int k_base, const int *base_labels,
                     const float *base_distances, float *distances = nullptr, bool sorted = false) {
  if (k >= k_base) {
    if (base_labels != NULL) {
      for (int i = 0; i < k_base; ++i) {
        labels[i] = base_labels[i];
      }
    } else {
      for (int i = 0; i < k_base; ++i) {
        labels[i] = i;
      }
    }

    if (distances) {
      for (int i = 0; i < k_base; ++i) {
        distances[i] = base_distances[i];
      }
    }

    return;
  }

  std::vector<int> perm(k_base);
  for (int i = 0; i < k_base; ++i) {
    perm[i] = i;
  }

  ArgsortComparator comp = {base_distances};

  if (sorted) {
    miniselect::pdqpartial_sort_branchless(perm.begin(), perm.begin() + k, perm.end(), comp);
  } else {
    miniselect::pdqselect_branchless(perm.begin(), perm.begin() + k, perm.end(), comp);
  }

  if (base_labels != NULL) {
    for (int i = 0; i < k; ++i) {
      labels[i] = base_labels[perm[i]];
    }
  } else {
    for (int i = 0; i < k; ++i) {
      labels[i] = perm[i];
    }
  }

  if (distances) {
    for (int i = 0; i < k; ++i) {
      distances[i] = base_distances[perm[i]];
    }
  }
}

/* Samples n random rows from the matrix X using reservoir sampling */
static RowMatrix sample_rows(const Eigen::Map<const RowMatrix> &X, const int sample_size) {
  if (sample_size >= X.rows()) {
    return X;
  }

  std::unordered_set<int> sample;
  std::mt19937_64 generator;

  int upper_bound = X.rows() - 1;
  for (int d = upper_bound - sample_size; d < upper_bound; d++) {
    int t = std::uniform_int_distribution<>(0, d)(generator);
    if (sample.find(t) == sample.end())
      sample.insert(t);
    else
      sample.insert(d);
  }

  RowMatrix ret(sample_size, X.cols());
  int i = 0;
  for (auto idx : sample) {
    ret.row(i++) = X.row(idx);
  }

  return ret;
}

/* Generates a standard normal random matrix of size nxn */
static inline Eigen::MatrixXf generate_random_normal_matrix(const int n) {
  std::mt19937_64 generator;
  std::normal_distribution<float> randn_distribution(0.0, 1.0);
  auto normal = [&](float) { return randn_distribution(generator); };

  Eigen::MatrixXf random_normal_matrix = Eigen::MatrixXf::NullaryExpr(n, n, normal);
  return random_normal_matrix;
}

/* Generates a random rotation matrix of size nxn */
static inline Eigen::MatrixXf generate_rotation_matrix(const int n) {
  /* the random rotation matrix is obtained as Q from the QR decomposition A = QR, where A is a
   * standard normal random matrix */
  Eigen::MatrixXf random_normal_matrix = generate_random_normal_matrix(n);
  return random_normal_matrix.fullPivHouseholderQr().matrixQ();
}

static inline Eigen::MatrixXf compute_principal_components(const Eigen::MatrixXf &X,
                                                           const int n_columns) {
  /* assumes X is a symmetric matrix */
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(X);
  Eigen::MatrixXf principal_components =
      es.eigenvectors()(Eigen::placeholders::all, Eigen::placeholders::lastN(n_columns));
  return principal_components.rowwise().reverse();
}

/* Computes V_r, the first r right singular vectors of X */
static inline Eigen::MatrixXf compute_V(const Eigen::MatrixXf &X, const int rank) {
  /* randomized (approximate) SVD */
  std::mt19937_64 randomEngine{};
  Rsvd::RandomizedSvd<Eigen::MatrixXf, std::mt19937_64, Rsvd::SubspaceIterationConditioner::Lu>
      rsvd(randomEngine);
  rsvd.compute(X, std::min(static_cast<long>(X.cols()), static_cast<long>(rank)), RSVD_OVERSAMPLES,
               RSVD_N_ITER);

  Eigen::MatrixXf V = Eigen::MatrixXf::Zero(X.cols(), rank);
  const long rows = std::min(static_cast<long>(X.cols()), static_cast<long>(rsvd.matrixV().rows()));
  const long cols = std::min(static_cast<long>(rank), static_cast<long>(rsvd.matrixV().cols()));
  V.topLeftCorner(rows, cols) = rsvd.matrixV().topLeftCorner(rows, cols);

  return V;
}

}  // namespace Lorann
