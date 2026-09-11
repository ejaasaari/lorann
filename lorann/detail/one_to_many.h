#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#elif defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(_MSC_VER)
#define LORANN_OTM_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define LORANN_OTM_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define LORANN_OTM_ALWAYS_INLINE inline
#endif

#define LORANN_OTM_REPEAT_4(operation) operation(0) operation(1) operation(2) operation(3)
#define LORANN_OTM_REPEAT_8(operation) \
  LORANN_OTM_REPEAT_4(operation) operation(4) operation(5) operation(6) operation(7)
#define LORANN_OTM_REPEAT_16(operation)                                                           \
  LORANN_OTM_REPEAT_8(operation)                                                                  \
  operation(8) operation(9) operation(10) operation(11) operation(12) operation(13) operation(14) \
      operation(15)

namespace Lorann {
namespace detail {

enum class OneToManyMetric { IP, L2 };

struct FallbackOneToMany {};
struct NeonOneToMany {};
struct SveOneToMany {};
struct Avx2OneToMany {};
struct Avx512OneToMany {};

template <typename Architecture>
struct OneToManyKernel;

template <>
struct OneToManyKernel<FallbackOneToMany> {
  template <OneToManyMetric metric, typename Output>
  static LORANN_OTM_ALWAYS_INLINE void run(const float *query, const float *data,
                                           const std::size_t dim, const int *indices,
                                           const std::size_t count, Output *output) {
    std::size_t candidate = 0;
    for (; count - candidate >= 4; candidate += 4) {
#define LORANN_OTM_FALLBACK_ROW(i) \
  const float *const row##i = data + static_cast<std::size_t>(indices[candidate + i]) * dim;
      LORANN_OTM_REPEAT_4(LORANN_OTM_FALLBACK_ROW)
#undef LORANN_OTM_FALLBACK_ROW

#define LORANN_OTM_FALLBACK_SUM(i) float sum##i = 0.0f;
      LORANN_OTM_REPEAT_4(LORANN_OTM_FALLBACK_SUM)
#undef LORANN_OTM_FALLBACK_SUM

      if constexpr (metric == OneToManyMetric::L2) {
        for (std::size_t j = 0; j < dim; ++j) {
          const float query_value = query[j];
#define LORANN_OTM_FALLBACK_L2(i)                \
  const float diff##i = query_value - row##i[j]; \
  sum##i += diff##i * diff##i;
          LORANN_OTM_REPEAT_4(LORANN_OTM_FALLBACK_L2)
#undef LORANN_OTM_FALLBACK_L2
        }
#define LORANN_OTM_FALLBACK_STORE_L2(i) output[candidate + i] = static_cast<Output>(sum##i);
        LORANN_OTM_REPEAT_4(LORANN_OTM_FALLBACK_STORE_L2)
#undef LORANN_OTM_FALLBACK_STORE_L2
      } else {
        for (std::size_t j = 0; j < dim; ++j) {
          const float query_value = query[j];
#define LORANN_OTM_FALLBACK_IP(i) sum##i += query_value * row##i[j];
          LORANN_OTM_REPEAT_4(LORANN_OTM_FALLBACK_IP)
#undef LORANN_OTM_FALLBACK_IP
        }
#define LORANN_OTM_FALLBACK_STORE_IP(i) output[candidate + i] = static_cast<Output>(-sum##i);
        LORANN_OTM_REPEAT_4(LORANN_OTM_FALLBACK_STORE_IP)
#undef LORANN_OTM_FALLBACK_STORE_IP
      }
    }

    for (; candidate < count; ++candidate) {
      const float *const row =
          data + static_cast<std::size_t>(indices[candidate]) * static_cast<std::size_t>(dim);
      float sum = 0.0f;
      if constexpr (metric == OneToManyMetric::L2) {
        for (std::size_t j = 0; j < dim; ++j) {
          const float diff = query[j] - row[j];
          sum += diff * diff;
        }
        output[candidate] = static_cast<Output>(sum);
      } else {
        for (std::size_t j = 0; j < dim; ++j) sum += query[j] * row[j];
        output[candidate] = static_cast<Output>(-sum);
      }
    }
  }
};

#if defined(__ARM_FEATURE_SVE)

template <>
struct OneToManyKernel<SveOneToMany> {
 private:
  template <OneToManyMetric metric>
  static LORANN_OTM_ALWAYS_INLINE float run_one(const float *query, const float *row,
                                                const std::size_t dim) {
    svfloat32_t sum = svdup_n_f32(0.0f);
    for (std::size_t j = 0; j < dim; j += svcntw()) {
      const svbool_t predicate =
          svwhilelt_b32_u64(static_cast<std::uint64_t>(j), static_cast<std::uint64_t>(dim));
      const svfloat32_t query_vector = svld1_f32(predicate, query + j);
      if constexpr (metric == OneToManyMetric::L2) {
        const svfloat32_t diff =
            svsub_f32_x(predicate, query_vector, svld1_f32(predicate, row + j));
        sum = svmla_f32_m(predicate, sum, diff, diff);
      } else {
        sum = svmla_f32_m(predicate, sum, query_vector, svld1_f32(predicate, row + j));
      }
    }
    const float scalar = svaddv_f32(svptrue_b32(), sum);
    if constexpr (metric == OneToManyMetric::L2) {
      return scalar;
    } else {
      return -scalar;
    }
  }

 public:
  template <OneToManyMetric metric, typename Output>
  static LORANN_OTM_ALWAYS_INLINE void run(const float *query, const float *data,
                                           const std::size_t dim, const int *indices,
                                           const std::size_t count, Output *output) {
    std::size_t candidate = 0;
    for (; count - candidate >= 16; candidate += 16) {
#define LORANN_OTM_SVE_ROW(i) \
  const float *const row##i = data + static_cast<std::size_t>(indices[candidate + i]) * dim;
      LORANN_OTM_REPEAT_16(LORANN_OTM_SVE_ROW)
#undef LORANN_OTM_SVE_ROW

#define LORANN_OTM_SVE_SUM(i) svfloat32_t sum##i = svdup_n_f32(0.0f);
      LORANN_OTM_REPEAT_16(LORANN_OTM_SVE_SUM)
#undef LORANN_OTM_SVE_SUM

      for (std::size_t j = 0; j < dim; j += svcntw()) {
        const svbool_t predicate =
            svwhilelt_b32_u64(static_cast<std::uint64_t>(j), static_cast<std::uint64_t>(dim));
        const svfloat32_t query_vector = svld1_f32(predicate, query + j);
        if constexpr (metric == OneToManyMetric::L2) {
#define LORANN_OTM_SVE_L2(i)                                                  \
  const svfloat32_t diff##i =                                                 \
      svsub_f32_x(predicate, query_vector, svld1_f32(predicate, row##i + j)); \
  sum##i = svmla_f32_m(predicate, sum##i, diff##i, diff##i);
          LORANN_OTM_REPEAT_16(LORANN_OTM_SVE_L2)
#undef LORANN_OTM_SVE_L2
        } else {
#define LORANN_OTM_SVE_IP(i) \
  sum##i = svmla_f32_m(predicate, sum##i, query_vector, svld1_f32(predicate, row##i + j));
          LORANN_OTM_REPEAT_16(LORANN_OTM_SVE_IP)
#undef LORANN_OTM_SVE_IP
        }
      }

      const svbool_t all = svptrue_b32();
#define LORANN_OTM_SVE_REDUCE(i) const float scalar##i = svaddv_f32(all, sum##i);
      LORANN_OTM_REPEAT_16(LORANN_OTM_SVE_REDUCE)
#undef LORANN_OTM_SVE_REDUCE

      if constexpr (metric == OneToManyMetric::L2) {
#define LORANN_OTM_SVE_STORE_L2(i) output[candidate + i] = static_cast<Output>(scalar##i);
        LORANN_OTM_REPEAT_16(LORANN_OTM_SVE_STORE_L2)
#undef LORANN_OTM_SVE_STORE_L2
      } else {
#define LORANN_OTM_SVE_STORE_IP(i) output[candidate + i] = static_cast<Output>(-scalar##i);
        LORANN_OTM_REPEAT_16(LORANN_OTM_SVE_STORE_IP)
#undef LORANN_OTM_SVE_STORE_IP
      }
    }

    for (; candidate < count; ++candidate) {
      const float *const row = data + static_cast<std::size_t>(indices[candidate]) * dim;
      output[candidate] = static_cast<Output>(run_one<metric>(query, row, dim));
    }
  }
};

#elif defined(__ARM_NEON) || defined(__ARM_NEON__)

template <>
struct OneToManyKernel<NeonOneToMany> {
 private:
  static LORANN_OTM_ALWAYS_INLINE float32x4_t multiply_add(const float32x4_t sum,
                                                           const float32x4_t left,
                                                           const float32x4_t right) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return vfmaq_f32(sum, left, right);
#else
    return vmlaq_f32(sum, left, right);
#endif
  }

  static LORANN_OTM_ALWAYS_INLINE float horizontal_sum(const float32x4_t value) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return vaddvq_f32(value);
#else
    const float32x2_t pair = vadd_f32(vget_low_f32(value), vget_high_f32(value));
    return vget_lane_f32(vpadd_f32(pair, pair), 0);
#endif
  }

  template <OneToManyMetric metric>
  static LORANN_OTM_ALWAYS_INLINE float run_one(const float *query, const float *row,
                                                const std::size_t dim) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    std::size_t j = 0;
    if constexpr (metric == OneToManyMetric::L2) {
      for (; j + 4 <= dim; j += 4) {
        const float32x4_t query_vector = vld1q_f32(query + j);
        const float32x4_t diff = vsubq_f32(query_vector, vld1q_f32(row + j));
        sum = multiply_add(sum, diff, diff);
      }
    } else {
      for (; j + 4 <= dim; j += 4) {
        sum = multiply_add(sum, vld1q_f32(query + j), vld1q_f32(row + j));
      }
    }

    float scalar = horizontal_sum(sum);
    if constexpr (metric == OneToManyMetric::L2) {
      for (; j < dim; ++j) {
        const float diff = query[j] - row[j];
        scalar += diff * diff;
      }
      return scalar;
    } else {
      for (; j < dim; ++j) scalar += query[j] * row[j];
      return -scalar;
    }
  }

 public:
  template <OneToManyMetric metric, typename Output>
  static LORANN_OTM_ALWAYS_INLINE void run(const float *query, const float *data,
                                           const std::size_t dim, const int *indices,
                                           const std::size_t count, Output *output) {
    std::size_t candidate = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
    for (; count - candidate >= 16; candidate += 16) {
#define LORANN_OTM_NEON_ROW(i) \
  const float *const row##i = data + static_cast<std::size_t>(indices[candidate + i]) * dim;
      LORANN_OTM_REPEAT_16(LORANN_OTM_NEON_ROW)
#undef LORANN_OTM_NEON_ROW

#define LORANN_OTM_NEON_SUM(i) float32x4_t sum##i = vdupq_n_f32(0.0f);
      LORANN_OTM_REPEAT_16(LORANN_OTM_NEON_SUM)
#undef LORANN_OTM_NEON_SUM

      std::size_t j = 0;
      if constexpr (metric == OneToManyMetric::L2) {
        for (; j + 4 <= dim; j += 4) {
          const float32x4_t query_vector = vld1q_f32(query + j);
#define LORANN_OTM_NEON_L2(i)                                                 \
  const float32x4_t diff##i = vsubq_f32(query_vector, vld1q_f32(row##i + j)); \
  sum##i = multiply_add(sum##i, diff##i, diff##i);
          LORANN_OTM_REPEAT_16(LORANN_OTM_NEON_L2)
#undef LORANN_OTM_NEON_L2
        }
      } else {
        for (; j + 4 <= dim; j += 4) {
          const float32x4_t query_vector = vld1q_f32(query + j);
#define LORANN_OTM_NEON_IP(i) sum##i = multiply_add(sum##i, query_vector, vld1q_f32(row##i + j));
          LORANN_OTM_REPEAT_16(LORANN_OTM_NEON_IP)
#undef LORANN_OTM_NEON_IP
        }
      }

#define LORANN_OTM_NEON_REDUCE(i) float scalar##i = horizontal_sum(sum##i);
      LORANN_OTM_REPEAT_16(LORANN_OTM_NEON_REDUCE)
#undef LORANN_OTM_NEON_REDUCE

      if constexpr (metric == OneToManyMetric::L2) {
        for (; j < dim; ++j) {
          const float query_value = query[j];
#define LORANN_OTM_NEON_TAIL_L2(i)               \
  const float diff##i = query_value - row##i[j]; \
  scalar##i += diff##i * diff##i;
          LORANN_OTM_REPEAT_16(LORANN_OTM_NEON_TAIL_L2)
#undef LORANN_OTM_NEON_TAIL_L2
        }
#define LORANN_OTM_NEON_STORE_L2(i) output[candidate + i] = static_cast<Output>(scalar##i);
        LORANN_OTM_REPEAT_16(LORANN_OTM_NEON_STORE_L2)
#undef LORANN_OTM_NEON_STORE_L2
      } else {
        for (; j < dim; ++j) {
          const float query_value = query[j];
#define LORANN_OTM_NEON_TAIL_IP(i) scalar##i += query_value * row##i[j];
          LORANN_OTM_REPEAT_16(LORANN_OTM_NEON_TAIL_IP)
#undef LORANN_OTM_NEON_TAIL_IP
        }
#define LORANN_OTM_NEON_STORE_IP(i) output[candidate + i] = static_cast<Output>(-scalar##i);
        LORANN_OTM_REPEAT_16(LORANN_OTM_NEON_STORE_IP)
#undef LORANN_OTM_NEON_STORE_IP
      }
    }
#endif

    for (; candidate < count; ++candidate) {
      const float *const row = data + static_cast<std::size_t>(indices[candidate]) * dim;
      output[candidate] = static_cast<Output>(run_one<metric>(query, row, dim));
    }
  }
};

#elif defined(__AVX512F__)

template <>
struct OneToManyKernel<Avx512OneToMany> {
 private:
  static LORANN_OTM_ALWAYS_INLINE __m512 multiply_add(const __m512 sum, const __m512 left,
                                                      const __m512 right) {
#if defined(__FMA__)
    return _mm512_fmadd_ps(left, right, sum);
#else
    return _mm512_add_ps(sum, _mm512_mul_ps(left, right));
#endif
  }

  template <OneToManyMetric metric>
  static LORANN_OTM_ALWAYS_INLINE float run_one(const float *query, const float *row,
                                                const std::size_t dim) {
    __m512 sum = _mm512_setzero_ps();
    std::size_t j = 0;
    if constexpr (metric == OneToManyMetric::L2) {
      for (; j + 16 <= dim; j += 16) {
        const __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(query + j), _mm512_loadu_ps(row + j));
        sum = multiply_add(sum, diff, diff);
      }
    } else {
      for (; j + 16 <= dim; j += 16) {
        sum = multiply_add(sum, _mm512_loadu_ps(query + j), _mm512_loadu_ps(row + j));
      }
    }

    float scalar = _mm512_reduce_add_ps(sum);
    if constexpr (metric == OneToManyMetric::L2) {
      for (; j < dim; ++j) {
        const float diff = query[j] - row[j];
        scalar += diff * diff;
      }
      return scalar;
    } else {
      for (; j < dim; ++j) scalar += query[j] * row[j];
      return -scalar;
    }
  }

 public:
  template <OneToManyMetric metric, typename Output>
  static LORANN_OTM_ALWAYS_INLINE void run(const float *query, const float *data,
                                           const std::size_t dim, const int *indices,
                                           const std::size_t count, Output *output) {
    std::size_t candidate = 0;
    for (; count - candidate >= 8; candidate += 8) {
#define LORANN_OTM_AVX512_ROW(i) \
  const float *const row##i = data + static_cast<std::size_t>(indices[candidate + i]) * dim;
      LORANN_OTM_REPEAT_8(LORANN_OTM_AVX512_ROW)
#undef LORANN_OTM_AVX512_ROW

#define LORANN_OTM_AVX512_SUM(i) __m512 sum##i = _mm512_setzero_ps();
      LORANN_OTM_REPEAT_8(LORANN_OTM_AVX512_SUM)
#undef LORANN_OTM_AVX512_SUM

      std::size_t j = 0;
      if constexpr (metric == OneToManyMetric::L2) {
        for (; j + 16 <= dim; j += 16) {
          const __m512 query_vector = _mm512_loadu_ps(query + j);
#define LORANN_OTM_AVX512_L2(i)                                                    \
  const __m512 diff##i = _mm512_sub_ps(query_vector, _mm512_loadu_ps(row##i + j)); \
  sum##i = multiply_add(sum##i, diff##i, diff##i);
          LORANN_OTM_REPEAT_8(LORANN_OTM_AVX512_L2)
#undef LORANN_OTM_AVX512_L2
        }
      } else {
        for (; j + 16 <= dim; j += 16) {
          const __m512 query_vector = _mm512_loadu_ps(query + j);
#define LORANN_OTM_AVX512_IP(i) \
  sum##i = multiply_add(sum##i, query_vector, _mm512_loadu_ps(row##i + j));
          LORANN_OTM_REPEAT_8(LORANN_OTM_AVX512_IP)
#undef LORANN_OTM_AVX512_IP
        }
      }

#define LORANN_OTM_AVX512_REDUCE(i) float scalar##i = _mm512_reduce_add_ps(sum##i);
      LORANN_OTM_REPEAT_8(LORANN_OTM_AVX512_REDUCE)
#undef LORANN_OTM_AVX512_REDUCE

      if constexpr (metric == OneToManyMetric::L2) {
        for (; j < dim; ++j) {
          const float query_value = query[j];
#define LORANN_OTM_AVX512_TAIL_L2(i)             \
  const float diff##i = query_value - row##i[j]; \
  scalar##i += diff##i * diff##i;
          LORANN_OTM_REPEAT_8(LORANN_OTM_AVX512_TAIL_L2)
#undef LORANN_OTM_AVX512_TAIL_L2
        }
#define LORANN_OTM_AVX512_STORE_L2(i) output[candidate + i] = static_cast<Output>(scalar##i);
        LORANN_OTM_REPEAT_8(LORANN_OTM_AVX512_STORE_L2)
#undef LORANN_OTM_AVX512_STORE_L2
      } else {
        for (; j < dim; ++j) {
          const float query_value = query[j];
#define LORANN_OTM_AVX512_TAIL_IP(i) scalar##i += query_value * row##i[j];
          LORANN_OTM_REPEAT_8(LORANN_OTM_AVX512_TAIL_IP)
#undef LORANN_OTM_AVX512_TAIL_IP
        }
#define LORANN_OTM_AVX512_STORE_IP(i) output[candidate + i] = static_cast<Output>(-scalar##i);
        LORANN_OTM_REPEAT_8(LORANN_OTM_AVX512_STORE_IP)
#undef LORANN_OTM_AVX512_STORE_IP
      }
    }

    for (; candidate < count; ++candidate) {
      const float *const row = data + static_cast<std::size_t>(indices[candidate]) * dim;
      output[candidate] = static_cast<Output>(run_one<metric>(query, row, dim));
    }
  }
};

#elif defined(__AVX2__)

template <>
struct OneToManyKernel<Avx2OneToMany> {
 private:
  static LORANN_OTM_ALWAYS_INLINE __m256 multiply_add(const __m256 sum, const __m256 left,
                                                      const __m256 right) {
#if defined(__FMA__)
    return _mm256_fmadd_ps(left, right, sum);
#else
    return _mm256_add_ps(sum, _mm256_mul_ps(left, right));
#endif
  }

  static LORANN_OTM_ALWAYS_INLINE float horizontal_sum(const __m256 value) {
    const __m128 halves =
        _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    const __m128 pairs = _mm_add_ps(halves, _mm_movehl_ps(halves, halves));
    return _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps(pairs, pairs, 0x55)));
  }

  template <OneToManyMetric metric>
  static LORANN_OTM_ALWAYS_INLINE float run_one(const float *query, const float *row,
                                                const std::size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    std::size_t j = 0;
    if constexpr (metric == OneToManyMetric::L2) {
      for (; j + 8 <= dim; j += 8) {
        const __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(query + j), _mm256_loadu_ps(row + j));
        sum = multiply_add(sum, diff, diff);
      }
    } else {
      for (; j + 8 <= dim; j += 8) {
        sum = multiply_add(sum, _mm256_loadu_ps(query + j), _mm256_loadu_ps(row + j));
      }
    }

    float scalar = horizontal_sum(sum);
    if constexpr (metric == OneToManyMetric::L2) {
      for (; j < dim; ++j) {
        const float diff = query[j] - row[j];
        scalar += diff * diff;
      }
      return scalar;
    } else {
      for (; j < dim; ++j) scalar += query[j] * row[j];
      return -scalar;
    }
  }

 public:
  template <OneToManyMetric metric, typename Output>
  static LORANN_OTM_ALWAYS_INLINE void run(const float *query, const float *data,
                                           const std::size_t dim, const int *indices,
                                           const std::size_t count, Output *output) {
    std::size_t candidate = 0;
    for (; count - candidate >= 8; candidate += 8) {
#define LORANN_OTM_AVX2_ROW(i) \
  const float *const row##i = data + static_cast<std::size_t>(indices[candidate + i]) * dim;
      LORANN_OTM_REPEAT_8(LORANN_OTM_AVX2_ROW)
#undef LORANN_OTM_AVX2_ROW

#define LORANN_OTM_AVX2_SUM(i) __m256 sum##i = _mm256_setzero_ps();
      LORANN_OTM_REPEAT_8(LORANN_OTM_AVX2_SUM)
#undef LORANN_OTM_AVX2_SUM

      std::size_t j = 0;
      if constexpr (metric == OneToManyMetric::L2) {
        for (; j + 8 <= dim; j += 8) {
          const __m256 query_vector = _mm256_loadu_ps(query + j);
#define LORANN_OTM_AVX2_L2(i)                                                      \
  const __m256 diff##i = _mm256_sub_ps(query_vector, _mm256_loadu_ps(row##i + j)); \
  sum##i = multiply_add(sum##i, diff##i, diff##i);
          LORANN_OTM_REPEAT_8(LORANN_OTM_AVX2_L2)
#undef LORANN_OTM_AVX2_L2
        }
      } else {
        for (; j + 8 <= dim; j += 8) {
          const __m256 query_vector = _mm256_loadu_ps(query + j);
#define LORANN_OTM_AVX2_IP(i) \
  sum##i = multiply_add(sum##i, query_vector, _mm256_loadu_ps(row##i + j));
          LORANN_OTM_REPEAT_8(LORANN_OTM_AVX2_IP)
#undef LORANN_OTM_AVX2_IP
        }
      }

#define LORANN_OTM_AVX2_REDUCE(i) float scalar##i = horizontal_sum(sum##i);
      LORANN_OTM_REPEAT_8(LORANN_OTM_AVX2_REDUCE)
#undef LORANN_OTM_AVX2_REDUCE

      if constexpr (metric == OneToManyMetric::L2) {
        for (; j < dim; ++j) {
          const float query_value = query[j];
#define LORANN_OTM_AVX2_TAIL_L2(i)               \
  const float diff##i = query_value - row##i[j]; \
  scalar##i += diff##i * diff##i;
          LORANN_OTM_REPEAT_8(LORANN_OTM_AVX2_TAIL_L2)
#undef LORANN_OTM_AVX2_TAIL_L2
        }
#define LORANN_OTM_AVX2_STORE_L2(i) output[candidate + i] = static_cast<Output>(scalar##i);
        LORANN_OTM_REPEAT_8(LORANN_OTM_AVX2_STORE_L2)
#undef LORANN_OTM_AVX2_STORE_L2
      } else {
        for (; j < dim; ++j) {
          const float query_value = query[j];
#define LORANN_OTM_AVX2_TAIL_IP(i) scalar##i += query_value * row##i[j];
          LORANN_OTM_REPEAT_8(LORANN_OTM_AVX2_TAIL_IP)
#undef LORANN_OTM_AVX2_TAIL_IP
        }
#define LORANN_OTM_AVX2_STORE_IP(i) output[candidate + i] = static_cast<Output>(-scalar##i);
        LORANN_OTM_REPEAT_8(LORANN_OTM_AVX2_STORE_IP)
#undef LORANN_OTM_AVX2_STORE_IP
      }
    }

    for (; candidate < count; ++candidate) {
      const float *const row = data + static_cast<std::size_t>(indices[candidate]) * dim;
      output[candidate] = static_cast<Output>(run_one<metric>(query, row, dim));
    }
  }
};

#endif

#if defined(__ARM_FEATURE_SVE)
using NativeOneToMany = SveOneToMany;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
using NativeOneToMany = NeonOneToMany;
#elif defined(__AVX512F__)
using NativeOneToMany = Avx512OneToMany;
#elif defined(__AVX2__)
using NativeOneToMany = Avx2OneToMany;
#else
using NativeOneToMany = FallbackOneToMany;
#endif

template <typename Output>
LORANN_OTM_ALWAYS_INLINE void compute_one_to_many(const float *query, const float *data,
                                                  const std::size_t dim, const int *indices,
                                                  const std::size_t count,
                                                  const OneToManyMetric metric, Output *output) {
  static_assert(std::is_floating_point<Output>::value,
                "one-to-many distance outputs must be floating point");
  if (count == 0) return;
  if (metric == OneToManyMetric::L2) {
    OneToManyKernel<NativeOneToMany>::template run<OneToManyMetric::L2>(query, data, dim, indices,
                                                                        count, output);
  } else {
    OneToManyKernel<NativeOneToMany>::template run<OneToManyMetric::IP>(query, data, dim, indices,
                                                                        count, output);
  }
}

}  // namespace detail
}  // namespace Lorann

#undef LORANN_OTM_REPEAT_16
#undef LORANN_OTM_REPEAT_8
#undef LORANN_OTM_REPEAT_4
#undef LORANN_OTM_ALWAYS_INLINE
