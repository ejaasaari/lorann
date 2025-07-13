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
struct Traits<_Float16> {
  static constexpr TypeMarker type_marker = FLOAT16;
  static constexpr size_t dim_divisor = 1;

#if defined(__AVX512FP16__)
  void convert_f16_to_f32(const _Float16 *src, float *dst, const size_t count) {
    size_t i = 0;

    for (; i + 16 <= count; i += 16) {
      __m256i f16_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i));
      __m512 f32_vec = _mm512_cvtph_ps(f16_vec);
      _mm512_storeu_ps(dst + i, f32_vec);
    }

    for (; i + 8 <= count; i += 8) {
      __m128i f16_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));
      __m256 f32_vec = _mm256_cvtph_ps(f16_vec);
      _mm256_storeu_ps(dst + i, f32_vec);
    }

    for (; i < count; i++) {
      dst[i] = _mm_cvtsh_ss(src[i]);
    }
  }
#elif defined(__ARM_FEATURE_FP16_FML) || defined(__ARM_FEATURE_FP16FML)
  __attribute__((target("+fullfp16,+fp16fml"))) static void convert_f16_to_f32(const _Float16 *src,
                                                                               float *dst,
                                                                               const size_t count) {
    size_t i = 0;

    for (; i + 8 <= count; i += 8) {
      float16x8_t f16_vec = vld1q_f16(reinterpret_cast<const float16_t *>(src + i));

      float32x4_t f32_low = vcvt_f32_f16(vget_low_f16(f16_vec));
      float32x4_t f32_high = vcvt_f32_f16(vget_high_f16(f16_vec));

      vst1q_f32(dst + i, f32_low);
      vst1q_f32(dst + i + 4, f32_high);
    }

    for (; i < count; i++) {
      dst[i] = static_cast<float>(src[i]);
    }
  }
#else
  static void convert_f16_to_f32(const _Float16 *src, float *dst, const size_t count) {
    for (size_t i = 0; i < count; ++i) {
      dst[i] = static_cast<float>(src[i]);
    }
  }
#endif

  static inline Lorann::ColVector to_float_vector(const _Float16 *x, int d) {
    Lorann::ColVector vec(d);
    convert_f16_to_f32(x, vec.data(), d);
    return vec;
  }

  static inline Lorann::MappedMatrix to_float_matrix(const _Float16 *data, const int n,
                                                     const int d) {
    auto buf = std::shared_ptr<float[]>{new float[n * d], std::default_delete<float[]>()};
    convert_f16_to_f32(data, buf.get(), n * d);
    return {buf.get(), n, d, buf};
  }

#if defined(__AVX512FP16__)
  static inline float dot_product(const _Float16 *restrict a, const _Float16 *restrict b,
                                  const size_t dim) {
    __m512 acc32 = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 31 < dim; i += 32) {
      __m512h va = _mm512_loadu_ph(a + i);
      __m512h vb = _mm512_loadu_ph(b + i);

      __m512 va32 = _mm512_cvtph_ps(va);
      __m512 vb32 = _mm512_cvtph_ps(vb);
      acc32 = _mm512_fmadd_ps(va32, vb32, acc32);
    }

    float sum = _mm512_reduce_add_ps(acc32);

    for (; i < dim; ++i) {
      sum += (float)a[i] * (float)b[i];
    }

    return sum;
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  __attribute__((target("+fullfp16,+fp16fml"))) static inline float dot_product(const _Float16 *a,
                                                                                const _Float16 *b,
                                                                                const size_t dim) {
    const float16_t *pa = (const float16_t *)a;
    const float16_t *pb = (const float16_t *)b;

    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);

    size_t i = 0;

#if defined(__ARM_FEATURE_FP16_FML) || defined(__ARM_FEATURE_FP16FML)
    for (; i + 7 < dim; i += 8) {
      float16x8_t va = vld1q_f16(pa + i);
      float16x8_t vb = vld1q_f16(pb + i);

      acc0 = vfmlalq_low_f16(acc0, va, vb);
      acc1 = vfmlalq_high_f16(acc1, va, vb);
    }
#else
    for (; i + 7 < dim; i += 8) {
      float16x8_t va = vld1q_f16(pa + i);
      float16x8_t vb = vld1q_f16(pb + i);

      float32x4_t lo = vcvt_f32_f16(vget_low_f16(va));
      float32x4_t hi = vcvt_f32_f16(vget_high_f16(va));

      float32x4_t lo_b = vcvt_f32_f16(vget_low_f16(vb));
      float32x4_t hi_b = vcvt_f32_f16(vget_high_f16(vb));

      acc0 = vfmaq_f32(acc0, lo, lo_b);
      acc1 = vfmaq_f32(acc1, hi, hi_b);
    }
#endif

    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < dim; ++i) {
      sum += (float)pa[i] * (float)pb[i];
    }

    return sum;
  }
#else
  static inline float dot_product(const _Float16 *a, const _Float16 *b, const size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
      sum += float(a[i]) * float(b[i]);
    }
    return sum;
  }
#endif

#if defined(__AVX512FP16__)
  static inline float squared_euclidean(const _Float16 *restrict a, const _Float16 *restrict b,
                                        const size_t dim) {
    __m512h vacc16 = _mm512_setzero_ph();

    size_t i = 0;
    for (; i + 31 < dim; i += 32) {
      __m512h va = _mm512_loadu_ph(a + i);
      __m512h vb = _mm512_loadu_ph(b + i);
      __m512h vdiff = _mm512_sub_ph(va, vb);
      vacc16 = _mm512_fmadd_ph(vdiff, vdiff, vacc16);
    }

    __m512 vacc32 = _mm512_cvtph_ps(vacc16);
    float sum = _mm512_reduce_add_ps(vacc32);

    for (; i < dim; ++i) {
      float diff = (float)a[i] - (float)b[i];
      sum += diff * diff;
    }

    return sum;
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  __attribute__((target("+fullfp16,+fp16fml"))) static inline float squared_euclidean(
      const _Float16 *a, const _Float16 *b, const size_t dim) {
    const float16_t *pa = (const float16_t *)a;
    const float16_t *pb = (const float16_t *)b;

    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);

    size_t i = 0;

#if defined(__ARM_FEATURE_FP16_FML) || defined(__ARM_FEATURE_FP16FML)
    for (; i + 7 < dim; i += 8) {
      float16x8_t diff = vsubq_f16(vld1q_f16(pa + i), vld1q_f16(pb + i));

      acc0 = vfmlalq_low_f16(acc0, diff, diff);
      acc1 = vfmlalq_high_f16(acc1, diff, diff);
    }

#else
    for (; i + 7 < dim; i += 8) {
      float16x8_t diff = vsubq_f16(vld1q_f16(pa + i), vld1q_f16(pb + i));

      float32x4_t lo = vcvt_f32_f16(vget_low_f16(diff));
      float32x4_t hi = vcvt_f32_f16(vget_high_f16(diff));

      acc0 = vfmaq_f32(acc0, lo, lo);
      acc1 = vfmaq_f32(acc1, hi, hi);
    }
#endif

    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));

    for (; i < dim; ++i) {
      float d = (float)pa[i] - (float)pb[i];
      sum += d * d;
    }
    return sum;
  }
#else
  static inline float squared_euclidean(const _Float16 *a, const _Float16 *b, const size_t dim) {
    float s = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
      const float diff = float(a[i]) - float(b[i]);
      s += diff * diff;
    }
    return s;
  }
#endif
};

}  // namespace detail

}  // namespace Lorann