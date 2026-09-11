#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <rsvd/Constants.hpp>
#include <rsvd/RandomizedSvd.hpp>
#include <unordered_set>
#include <vector>

#ifdef _MSC_VER
#include <malloc.h>
#endif

#include "miniselect/ipnselect.h"
#include "miniselect/pdqselect.h"

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

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

#define RSVD_OVERSAMPLES 5
#define RSVD_N_ITER 4

#define LORANN_MIN(a, b) ((a) < (b) ? (a) : (b))

#define LORANN_ENSURE_POSITIVE(x)                               \
  if ((x) <= 0) {                                               \
    throw std::invalid_argument("Value must be positive: " #x); \
  }

#ifndef LORANN_RESTRICT
#if defined(__GNUC__) || defined(__clang__)
#define LORANN_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define LORANN_RESTRICT __restrict
#elif defined(__INTEL_COMPILER)
#define LORANN_RESTRICT __restrict__
#else
#define LORANN_RESTRICT
#endif
#endif

#ifndef LORANN_ALWAYS_INLINE
#if __has_cpp_attribute(gnu::always_inline)
#define LORANN_ALWAYS_INLINE [[gnu::always_inline]]
#elif defined(__GNUC__) || defined(__clang__)
#define LORANN_ALWAYS_INLINE __attribute__((always_inline))
#elif defined(_MSC_VER)
#define LORANN_ALWAYS_INLINE __forceinline
#else
#define LORANN_ALWAYS_INLINE
#endif
#endif

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
typedef Eigen::Matrix<lorann_dist_t, 1, Eigen::Dynamic> DistVector;

struct MappedMatrix {
  Eigen::Map<const RowMatrix> view;
  std::shared_ptr<float[]> owner;

  MappedMatrix(const float *p, const Eigen::Index n, const Eigen::Index d,
               std::shared_ptr<float[]> own = {})
      : view{p, n, d}, owner{std::move(own)} {}
};

template <typename T>
struct ArgsortComparator {
  const T *vals;
  bool operator()(const int a, const int b) const { return vals[a] < vals[b]; }
};

template <typename T>
inline void check_for_nan(const T *data, const std::size_t n, const std::size_t dim) {
  if constexpr (std::is_floating_point<T>::value) {
    const std::size_t count = static_cast<std::size_t>(n) * static_cast<std::size_t>(dim);
    bool has_nan = false;
    for (std::size_t i = 0; i < count; ++i) {
      has_nan |= std::isnan(data[i]);
    }
    if (has_nan) {
      throw std::invalid_argument("Data matrix contains NaN");
    }
  }
}

template <typename T>
inline std::unique_ptr<T[], void (*)(T *)> make_aligned_array(std::size_t count,
                                                              std::size_t alignment = 64) {
  std::size_t bytes = count * sizeof(T);
  std::size_t aligned_bytes = ((bytes + alignment - 1) / alignment) * alignment;

#ifdef _MSC_VER
  T *aligned_ptr = static_cast<T *>(_aligned_malloc(aligned_bytes, alignment));
  if (!aligned_ptr) {
    throw std::bad_alloc();
  }
  return {aligned_ptr, [](T *ptr) { _aligned_free(ptr); }};
#else
  T *aligned_ptr = static_cast<T *>(std::aligned_alloc(alignment, aligned_bytes));
  if (!aligned_ptr) {
    throw std::bad_alloc();
  }
  return {aligned_ptr, [](T *ptr) { std::free(ptr); }};
#endif
}

template <typename T>
inline std::shared_ptr<T[]> make_aligned_shared_array(std::size_t count,
                                                      std::size_t alignment = 64) {
  auto unique_ptr = make_aligned_array<T>(count, alignment);
  T *raw_ptr = unique_ptr.release();
  return {raw_ptr, [](T *ptr) { std::free(ptr); }};
}

#if defined(__AVX2__)
LORANN_ALWAYS_INLINE static inline int32_t horizontal_add(__m128i const a) {
  const __m128i sum1 = _mm_hadd_epi32(a, a);
  const __m128i sum2 = _mm_hadd_epi32(sum1, sum1);
  return _mm_cvtsi128_si32(sum2);
}

LORANN_ALWAYS_INLINE static inline int32_t horizontal_add(__m256i const a) {
  const __m128i sum1 = _mm_add_epi32(_mm256_extracti128_si256(a, 1), _mm256_castsi256_si128(a));
  const __m128i sum2 = _mm_add_epi32(sum1, _mm_unpackhi_epi64(sum1, sum1));
  const __m128i sum3 = _mm_add_epi32(sum2, _mm_shuffle_epi32(sum2, 1));
  return (int32_t)_mm_cvtsi128_si32(sum3);
}
#endif

#if defined(__AVX2__)
static inline void add_inplace(const float *LORANN_RESTRICT v, float *LORANN_RESTRICT r,
                               const size_t n) {
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
#elif defined(__ARM_FEATURE_SVE)
static inline void add_inplace(const float *LORANN_RESTRICT v, float *LORANN_RESTRICT r,
                               const size_t n) {
  size_t i = 0;
  const size_t lanes = svcntw();

  while (i < n) {
    const svbool_t active = svwhilelt_b32(static_cast<uint64_t>(i), static_cast<uint64_t>(n));
    const svfloat32_t v_vec = svld1_f32(active, v + i);
    const svfloat32_t r_vec = svld1_f32(active, r + i);
    svst1_f32(active, r + i, svadd_f32_x(active, r_vec, v_vec));
    i += lanes;
  }
}
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
static inline void add_inplace(const float *LORANN_RESTRICT v, float *LORANN_RESTRICT r,
                               const size_t n) {
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t rv = vld1q_f32(v + i);
    float32x4_t rr = vld1q_f32(r + i);
    rr = vaddq_f32(rr, rv);
    vst1q_f32(r + i, rr);
  }

  for (; i < n; ++i) {
    r[i] += v[i];
  }
}
#else
static inline void add_inplace(const float *LORANN_RESTRICT v, float *LORANN_RESTRICT r,
                               const size_t n) {
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

template <typename BaseDist, typename OutDist = BaseDist>
static void select_k(const int k, int *labels, const int k_base, const int *base_labels,
                     const BaseDist *base_distances, OutDist *distances = nullptr,
                     bool sorted = false) {
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

  ArgsortComparator<BaseDist> comp = {base_distances};

  if (sorted) {
    miniselect::pdqpartial_sort_branchless(perm.begin(), perm.begin() + k, perm.end(), comp);
  } else if (k_base >= 32768) {
    miniselect::ipnselect_branchless(perm.data(), perm.data() + k, perm.data() + k_base, comp);
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

#if defined(__AVX512F__)
inline void store_selected_pairs(float *scores, int *ids, __m512 values, __m512i labels,
                                 __mmask16 mask, int count) {
  const __mmask16 output_mask = (1u << count) - 1u;
  // Zen 4 handles register compression plus masked stores faster than compress-store.
  _mm512_mask_storeu_ps(scores, output_mask, _mm512_maskz_compress_ps(mask, values));
  _mm512_mask_storeu_epi32(ids, output_mask, _mm512_maskz_compress_epi32(mask, labels));
}

// Hold both end blocks in registers so compacted stores cannot overwrite unread input.
template <bool IncludeEqual>
inline int partition_candidates(float *scores, int *ids, int begin, int end, float pivot) {
  const __m512 cut = _mm512_set1_ps(pivot);
  int read_left = begin + 16, read_right = end - 16;
  int write_left = begin, write_right = end;
  const __m512 first = _mm512_loadu_ps(scores + begin);
  const __m512 last = _mm512_loadu_ps(scores + read_right);
  const __m512i first_ids = _mm512_loadu_si512(ids + begin);
  const __m512i last_ids = _mm512_loadu_si512(ids + read_right);
  auto emit = [&](const __m512 values, const __m512i labels, const __mmask16 valid) {
    const __mmask16 lower =
        valid & _mm512_cmp_ps_mask(values, cut, IncludeEqual ? _CMP_LE_OQ : _CMP_LT_OQ);
    const __mmask16 upper = valid & ~lower;
    const int lower_count = _mm_popcnt_u32(lower), upper_count = _mm_popcnt_u32(upper);
    write_right -= upper_count;
    store_selected_pairs(scores + write_left, ids + write_left, values, labels, lower, lower_count);
    store_selected_pairs(scores + write_right, ids + write_right, values, labels, upper,
                         upper_count);
    write_left += lower_count;
  };
  while (read_right - read_left >= 16) {
    int position;
    if (read_left - write_left < write_right - read_right) {
      position = read_left;
      read_left += 16;
    } else {
      read_right -= 16;
      position = read_right;
    }
    emit(_mm512_loadu_ps(scores + position), _mm512_loadu_si512(ids + position), 0xffff);
  }
  const __mmask16 tail_mask = (1u << (read_right - read_left)) - 1u;
  const __m512 tail = _mm512_maskz_loadu_ps(tail_mask, scores + read_left);
  const __m512i tail_ids = _mm512_maskz_loadu_epi32(tail_mask, ids + read_left);
  emit(first, first_ids, 0xffff);
  emit(last, last_ids, 0xffff);
  emit(tail, tail_ids, tail_mask);
  return write_left;
}

// On failure, scores remain paired with their IDs so generic selection can finish the job.
inline bool partition_best_candidates(float *scores, int *ids, int count, int k) {
  int begin = 0, end = count;
  for (int iteration = 0; end - begin >= 32; ++iteration) {
    if (iteration == 24) return false;
    float sample[17];
    for (int i = 0; i < 17; ++i)
      sample[i] = scores[begin + static_cast<std::int64_t>(end - begin - 1) * i / 16];
    const int rank =
        std::clamp(int(static_cast<std::int64_t>(k - begin) * 17 / (end - begin)), 1, 15);
    std::nth_element(sample, sample + rank, sample + 17);
    const float pivot = sample[rank];
    int split = partition_candidates<false>(scores, ids, begin, end, pivot);
    if (split == k) return true;
    if (split == begin) {
      split = partition_candidates<true>(scores, ids, begin, end, pivot);
      if (split >= k) return true;
      if (split == begin) return false;
    }
    if (split > k)
      end = split;
    else
      begin = split;
  }
  for (int i = begin + 1; i < end; ++i) {
    const float score = scores[i];
    const int id = ids[i];
    int j = i;
    while (j > begin && score < scores[j - 1]) {
      scores[j] = scores[j - 1];
      ids[j] = ids[j - 1];
      --j;
    }
    scores[j] = score;
    ids[j] = id;
  }
  return true;
}
#endif

// Select k IDs for exact reranking. Scores and IDs may be permuted together.
inline void select_candidates(int k, int *selected, int count, int *ids, float *scores) {
#if defined(__AVX512F__)
  if (count >= 4096 && k > 0 && k < count && partition_best_candidates(scores, ids, count, k)) {
    std::copy_n(ids, k, selected);
    return;
  }
#endif
  select_k<float>(k, selected, count, ids, scores);
}

/* Samples n random rows from the matrix X using reservoir sampling */
static RowMatrix sample_rows(const Eigen::Map<const RowMatrix> &X, const int sample_size) {
  if (sample_size >= X.rows()) {
    return X;
  }

  // A local seeded stream makes benchmark builds reproducible across threads.
  const char *seed = std::getenv("LORANN_SEED");
  std::mt19937_64 generator(seed ? std::stoull(seed) : std::random_device{}());

  std::vector<int> reservoir(sample_size);
  std::iota(reservoir.begin(), reservoir.end(), 0);

  for (int i = sample_size; i < X.rows(); ++i) {
    int j = std::uniform_int_distribution<>(0, i)(generator);
    if (j < sample_size) {
      reservoir[j] = i;
    }
  }

  RowMatrix ret(sample_size, X.cols());
  for (int i = 0; i < sample_size; ++i) {
    ret.row(i) = X.row(reservoir[i]);
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

static inline Eigen::MatrixXf compute_principal_components_from_rows(const RowMatrix &X,
                                                                     const int n_columns) {
  Eigen::MatrixXf gram = Eigen::MatrixXf::Zero(X.cols(), X.cols());
  gram.selfadjointView<Eigen::Lower>().rankUpdate(X.transpose());
  return compute_principal_components(gram, n_columns);
}

/* Computes V_r, the first r right singular vectors of X */
static inline Eigen::MatrixXf compute_V(const Eigen::MatrixXf &X, const int rank) {
  Eigen::MatrixXf V = Eigen::MatrixXf::Zero(X.cols(), rank);
  const long effective_rank =
      std::min({static_cast<long>(X.rows()), static_cast<long>(X.cols()), static_cast<long>(rank)});
  if (effective_rank == 0) return V;

  /* randomized (approximate) SVD */
  std::mt19937_64 randomEngine{};
  Rsvd::RandomizedSvd<Eigen::MatrixXf, std::mt19937_64, Rsvd::SubspaceIterationConditioner::Lu>
      rsvd(randomEngine);
  Eigen::MatrixXf right_singular_vectors;
  if (X.rows() < X.cols()) {
    /* RandomizedSvd expects a tall matrix. The left singular vectors of X^T are the right
     * singular vectors of X, so transposing also handles compressed, wide model inputs safely. */
    Eigen::MatrixXf X_transposed = X.transpose();
    rsvd.compute(X_transposed, effective_rank, RSVD_OVERSAMPLES, RSVD_N_ITER);
    right_singular_vectors = rsvd.matrixU();
  } else {
    rsvd.compute(X, effective_rank, RSVD_OVERSAMPLES, RSVD_N_ITER);
    right_singular_vectors = rsvd.matrixV();
  }

  const long rows =
      std::min(static_cast<long>(X.cols()), static_cast<long>(right_singular_vectors.rows()));
  const long cols =
      std::min(static_cast<long>(rank), static_cast<long>(right_singular_vectors.cols()));
  V.topLeftCorner(rows, cols) = right_singular_vectors.topLeftCorner(rows, cols);

  return V;
}

}  // namespace Lorann
