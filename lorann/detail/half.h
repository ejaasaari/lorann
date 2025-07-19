#pragma once

#include <cstddef>

#include <simsimd/simsimd.h>

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
struct Traits<simsimd_f16_t> {
  static constexpr TypeMarker type_marker = FLOAT16;
  static constexpr size_t dim_divisor = 1;

#if defined(__AVX512FP16__)
  void convert_f16_to_f32(const simsimd_f16_t *src, float *dst, const size_t count) {
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
  __attribute__((target("+fullfp16,+fp16fml"))) static void convert_f16_to_f32(const simsimd_f16_t *src,
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
  static void convert_f16_to_f32(const simsimd_f16_t *src, float *dst, const size_t count) {
    for (size_t i = 0; i < count; ++i) {
      dst[i] = static_cast<float>(src[i]);
    }
  }
#endif

  static inline Lorann::ColVector to_float_vector(const simsimd_f16_t *x, int d) {
    Lorann::ColVector vec(d);
    convert_f16_to_f32(x, vec.data(), d);
    return vec;
  }

  static inline Lorann::MappedMatrix to_float_matrix(const simsimd_f16_t *data, const int n,
                                                     const int d) {
    auto buf = std::shared_ptr<float[]>{new float[n * d], std::default_delete<float[]>()};
    convert_f16_to_f32(data, buf.get(), n * d);
    return {buf.get(), n, d, buf};
  }

  static inline float dot_product(const simsimd_f16_t *a, const simsimd_f16_t *b,
                                  const size_t dim) {
    simsimd_distance_t distance;
    simsimd_dot_f16(a, b, dim, &distance);
    return distance;
  }

  static inline float squared_euclidean(const simsimd_f16_t *a, const simsimd_f16_t *b,
                                        const size_t dim) {
    simsimd_distance_t distance;
    simsimd_l2sq_f16(a, b, dim, &distance);
    return distance;
  }
};

}  // namespace detail

}  // namespace Lorann
