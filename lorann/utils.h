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

struct ArgsortComparator {
  const float *vals;
  bool operator()(const int a, const int b) const { return vals[a] < vals[b]; }
};

#if defined(__AVX512F__)
static inline float dot_product(const float *x1, const float *x2, size_t length) {
  __m512 sum = _mm512_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= length; i += 16) {
    __m512 v1 = _mm512_loadu_ps(x1 + i);
    __m512 v2 = _mm512_loadu_ps(x2 + i);
    sum = _mm512_fmadd_ps(v1, v2, sum);
  }
  if (i < length) {
    __m512 v1 = _mm512_maskz_loadu_ps((1 << (length - i)) - 1, x1 + i);
    __m512 v2 = _mm512_maskz_loadu_ps((1 << (length - i)) - 1, x2 + i);
    sum = _mm512_fmadd_ps(v1, v2, sum);
  }

  auto sumh = _mm256_add_ps(_mm512_castps512_ps256(sum), _mm512_extractf32x8_ps(sum, 1));
  auto sumhh = _mm_add_ps(_mm256_castps256_ps128(sumh), _mm256_extractf128_ps(sumh, 1));
  auto tmp1 = _mm_add_ps(sumhh, _mm_movehl_ps(sumhh, sumhh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}
#elif defined(__FMA__)
static inline float dot_product(const float *x1, const float *x2, size_t length) {
  __m256 sum = _mm256_setzero_ps();

  size_t i;
  for (i = 0; i + 7 < length; i += 8) {
    __m256 v1 = _mm256_load_ps(x1 + i);
    __m256 v2 = _mm256_load_ps(x2 + i);
    sum = _mm256_fmadd_ps(v1, v2, sum);
  }

  __attribute__((aligned(32))) float temp[8];
  _mm256_store_ps(temp, sum);
  float result = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

  for (; i < length; ++i) {
    result += x1[i] * x2[i];
  }

  return result;
}
#elif defined(__AVX2__)
static inline float dot_product(const float *x1, const float *x2, size_t length) {
  __m256 sum = _mm256_setzero_ps();

  size_t i;
  for (i = 0; i + 7 < length; i += 8) {
    __m256 v1 = _mm256_load_ps(x1 + i);
    __m256 v2 = _mm256_load_ps(x2 + i);
    __m256 prod = _mm256_mul_ps(v1, v2);
    sum = _mm256_add_ps(sum, prod);
  }

  __attribute__((aligned(32))) float temp[8];
  _mm256_store_ps(temp, sum);
  float result = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

  for (; i < length; ++i) {
    result += x1[i] * x2[i];
  }

  return result;
}
#elif defined(__ARM_FEATURE_SVE)
static inline float dot_product(const float *x1, const float *x2, size_t length) {
  int64_t i = 0;
  svfloat32_t sum = svdup_n_f32(0);
  while (i + svcntw() <= length) {
    svfloat32_t in1 = svld1_f32(svptrue_b32(), x1 + i);
    svfloat32_t in2 = svld1_f32(svptrue_b32(), x2 + i);
    sum = svmad_f32_m(svptrue_b32(), in1, in2, sum);
    i += svcntw();
  }
  svbool_t while_mask = svwhilelt_b32(i, length);
  do {
    svfloat32_t in1 = svld1_f32(while_mask, x1 + i);
    svfloat32_t in2 = svld1_f32(while_mask, x2 + i);
    sum = svmad_f32_m(svptrue_b32(), in1, in2, sum);
    i += svcntw();
    while_mask = svwhilelt_b32(i, length);
  } while (svptest_any(svptrue_b32(), while_mask));

  return svaddv_f32(svptrue_b32(), sum);
}
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
static inline float dot_product(const float *x1, const float *x2, size_t length) {
  float32x4_t ab_vec = vdupq_n_f32(0);
  size_t i = 0;
  for (; i + 4 <= length; i += 4) {
    float32x4_t a_vec = vld1q_f32(x1 + i);
    float32x4_t b_vec = vld1q_f32(x2 + i);
    ab_vec = vfmaq_f32(ab_vec, a_vec, b_vec);
  }
  float ab = vaddvq_f32(ab_vec);
  for (; i < length; ++i) {
    ab += x1[i] * x2[i];
  }
  return ab;
}
#else
static inline float dot_product(const float *x1, const float *x2, size_t length) {
  float sum = 0;
  for (size_t i = 0; i < length; ++i) {
    sum += x1[i] * x2[i];
  }
  return sum;
}
#endif

#if defined(__AVX512F__)
static inline float squared_euclidean(const float *x1, const float *x2, size_t length) {
  __m512 sum = _mm512_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= length; i += 16) {
    __m512 v1 = _mm512_loadu_ps(x1 + i);
    __m512 v2 = _mm512_loadu_ps(x2 + i);
    __m512 diff = _mm512_sub_ps(v1, v2);
    sum = _mm512_fmadd_ps(diff, diff, sum);
  }
  if (i < length) {
    __m512 v1 = _mm512_maskz_loadu_ps((1 << (length - i)) - 1, x1 + i);
    __m512 v2 = _mm512_maskz_loadu_ps((1 << (length - i)) - 1, x2 + i);
    __m512 diff = _mm512_sub_ps(v1, v2);
    sum = _mm512_fmadd_ps(diff, diff, sum);
  }

  __m256 sumh = _mm256_add_ps(_mm512_castps512_ps256(sum), _mm512_extractf32x8_ps(sum, 1));
  __m128 sumhh = _mm_add_ps(_mm256_castps256_ps128(sumh), _mm256_extractf128_ps(sumh, 1));
  __m128 tmp1 = _mm_add_ps(sumhh, _mm_movehl_ps(sumhh, sumhh));
  __m128 tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}
#elif defined(__FMA__)
static inline float squared_euclidean(const float *x1, const float *x2, size_t length) {
  __m256 sum = _mm256_setzero_ps();

  size_t i;
  for (i = 0; i + 7 < length; i += 8) {
    __m256 v1 = _mm256_load_ps(x1 + i);
    __m256 v2 = _mm256_load_ps(x2 + i);
    __m256 diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_fmadd_ps(diff, diff, sum);
  }

  __attribute__((aligned(32))) float temp[8];
  _mm256_store_ps(temp, sum);
  float result = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

  for (; i < length; ++i) {
    float diff = x1[i] - x2[i];
    result += diff * diff;
  }

  return result;
}
#elif defined(__AVX2__)
static inline float squared_euclidean(const float *x1, const float *x2, size_t length) {
  __m256 sum = _mm256_setzero_ps();

  size_t i;
  for (i = 0; i + 7 < length; i += 8) {
    __m256 v1 = _mm256_load_ps(x1 + i);
    __m256 v2 = _mm256_load_ps(x2 + i);
    __m256 diff = _mm256_sub_ps(v1, v2);
    __m256 squared = _mm256_mul_ps(diff, diff);
    sum = _mm256_add_ps(sum, squared);
  }

  __attribute__((aligned(32))) float temp[8];
  _mm256_store_ps(temp, sum);
  float result = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

  for (; i < length; ++i) {
    float diff = x1[i] - x2[i];
    result += diff * diff;
  }

  return result;
}
#elif defined(__ARM_FEATURE_SVE)
static inline float dot_product(const float *x1, const float *x2, size_t length) {
  int64_t i = 0;
  svfloat32_t sum = svdup_n_f32(0);
  while (i + svcntw() <= length) {
    svfloat32_t in1 = svld1_f32(svptrue_b32(), x1 + i);
    svfloat32_t in2 = svld1_f32(svptrue_b32(), x2 + i);
    svfloat32_t diff = svsub_f32_m(svptrue_b32(), in1, in2);
    sum = svmla_f32_m(svptrue_b32(), sum, diff, diff);
    i += svcntw();
  }
  svbool_t while_mask = svwhilelt_b32(i, length);
  do {
    svfloat32_t in1 = svld1_f32(while_mask, x1 + i);
    svfloat32_t in2 = svld1_f32(while_mask, x2 + i);
    svfloat32_t diff = svsub_f32_m(while_mask, in1, in2);
    sum = svmla_f32_m(while_mask, sum, diff, diff);
    i += svcntw();
    while_mask = svwhilelt_b32(i, length);
  } while (svptest_any(svptrue_b32(), while_mask));

  return svaddv_f32(svptrue_b32(), sum);
}
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
static inline float squared_euclidean(const float *x1, const float *x2, size_t length) {
  float32x4_t diff_sum = vdupq_n_f32(0);
  size_t i = 0;
  for (; i + 4 <= length; i += 4) {
    float32x4_t a_vec = vld1q_f32(x1 + i);
    float32x4_t b_vec = vld1q_f32(x2 + i);
    float32x4_t diff_vec = vsubq_f32(a_vec, b_vec);
    diff_sum = vfmaq_f32(diff_sum, diff_vec, diff_vec);
  }
  float sqr_dist = vaddvq_f32(diff_sum);
  for (; i < length; ++i) {
    float diff = x1[i] - x2[i];
    sqr_dist += diff * diff;
  }
  return sqr_dist;
}
#else
static inline float squared_euclidean(const float *x1, const float *x2, size_t length) {
  float sqr_dist = 0;
  for (size_t i = 0; i < length; ++i) {
    float diff = x1[i] - x2[i];
    sqr_dist += diff * diff;
  }
  return sqr_dist;
}
#endif

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
static inline Eigen::MatrixXf compute_V(const Eigen::MatrixXf &X, const int rank,
                                        const bool approximate) {
  if (approximate) {
    /* randomized (approximate) SVD */
    std::mt19937_64 randomEngine{};
    Rsvd::RandomizedSvd<Eigen::MatrixXf, std::mt19937_64, Rsvd::SubspaceIterationConditioner::Lu>
        rsvd(randomEngine);
    rsvd.compute(X, std::min(static_cast<long>(X.cols()), static_cast<long>(rank)),
                 RSVD_OVERSAMPLES, RSVD_N_ITER);

    Eigen::MatrixXf V = Eigen::MatrixXf::Zero(X.cols(), rank);
    const long rows =
        std::min(static_cast<long>(X.cols()), static_cast<long>(rsvd.matrixV().rows()));
    const long cols = std::min(static_cast<long>(rank), static_cast<long>(rsvd.matrixV().cols()));
    V.topLeftCorner(rows, cols) = rsvd.matrixV().topLeftCorner(rows, cols);

    return V;
  } else {
    /* exact SVD */
    Eigen::BDCSVD<Eigen::MatrixXf, Eigen::ComputeFullV> svd(X);

    Eigen::MatrixXf V = Eigen::MatrixXf::Zero(X.cols(), rank);
    const long rows =
        std::min(static_cast<long>(X.cols()), static_cast<long>(svd.matrixV().rows()));
    const long cols = std::min(static_cast<long>(rank), static_cast<long>(svd.matrixV().cols()));
    V.topLeftCorner(rows, cols) = svd.matrixV().topLeftCorner(rows, cols);

    return V;
  }
}

}  // namespace Lorann
