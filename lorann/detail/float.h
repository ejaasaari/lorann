#pragma once

#include <cstddef>

#include "traits.h"
#include "utils.h"

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

namespace Lorann {

namespace detail {

template <>
struct Traits<float> {
  static constexpr TypeMarker type_marker = FLOAT32;
  static constexpr int dim_divisor = 1;

  static inline Lorann::ColVector to_float_vector(const float *x, int d) {
    return Eigen::Map<const Lorann::ColVector>(x, d);
  }

  static inline Lorann::MappedMatrix to_float_matrix(const float *data, const int n, const int d) {
    return {data, static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(d)};
  }

#if defined(__AVX512F__)
  static inline float dot_product(const float *x1, const float *x2, const size_t length) {
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
  static inline float dot_product(const float *x1, const float *x2, const size_t length) {
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
  static inline float dot_product(const float *x1, const float *x2, const size_t length) {
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
  static inline float dot_product(const float *x1, const float *x2, const size_t length) {
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
  static inline float dot_product(const float *x1, const float *x2, const size_t length) {
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
  static inline float dot_product(const float *x1, const float *x2, const size_t length) {
    float sum = 0;
    for (size_t i = 0; i < length; ++i) {
      sum += x1[i] * x2[i];
    }
    return sum;
  }
#endif

#if defined(__AVX512F__)
  static inline float squared_euclidean(const float *x1, const float *x2, const size_t length) {
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
  static inline float squared_euclidean(const float *x1, const float *x2, const size_t length) {
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
  static inline float squared_euclidean(const float *x1, const float *x2, const size_t length) {
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
  static inline float dot_product(const float *x1, const float *x2, const size_t length) {
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
  static inline float squared_euclidean(const float *x1, const float *x2, const size_t length) {
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
  static inline float squared_euclidean(const float *x1, const float *x2, const size_t length) {
    float sqr_dist = 0;
    for (size_t i = 0; i < length; ++i) {
      float diff = x1[i] - x2[i];
      sqr_dist += diff * diff;
    }
    return sqr_dist;
  }
#endif
};

}  // namespace detail

}  // namespace Lorann