#pragma once

#include <cstddef>
#include <cstdint>

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

#if defined(__GNUC__) || defined(__clang__)
#define POPCOUNT(x) __builtin_popcount(x)
#elif defined(_MSC_VER) || defined(__MINGW32__)
#define POPCOUNT(x) __popcnt(x)
#else
int popcount_fallback(unsigned int n) {
  int count = 0;
  while (n > 0) {
    n &= (n - 1);
    count++;
  }
  return count;
}
#define POPCOUNT(x) popcount_fallback(x)
#endif

namespace Lorann {

struct BinaryType {
  uint8_t v;
  constexpr BinaryType() : v{} {}
  constexpr explicit BinaryType(uint8_t b) : v{b} {}

  constexpr operator uint8_t() const noexcept { return v; }
};

namespace detail {

template <>
struct Traits<BinaryType> {
  static constexpr TypeMarker type_marker = BINARY;
  static constexpr int dim_divisor = 8;

#if defined(__AVX2__)
  static inline void unpack(const uint8_t *in, float *out, std::size_t d) {
    const std::size_t nbytes = d >> 3;
    const __m256i one_i = _mm256_set1_epi32(1);
    const __m256i shifts = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    std::size_t i = 0;

    for (; i + 4 <= nbytes; i += 4) {
      __m256i v0 = _mm256_set1_epi32(in[i]);
      __m256i v1 = _mm256_set1_epi32(in[i + 1]);
      __m256i v2 = _mm256_set1_epi32(in[i + 2]);
      __m256i v3 = _mm256_set1_epi32(in[i + 3]);

      __m256i bits0 = _mm256_and_si256(_mm256_srlv_epi32(v0, shifts), one_i);
      __m256i bits1 = _mm256_and_si256(_mm256_srlv_epi32(v1, shifts), one_i);
      __m256i bits2 = _mm256_and_si256(_mm256_srlv_epi32(v2, shifts), one_i);
      __m256i bits3 = _mm256_and_si256(_mm256_srlv_epi32(v3, shifts), one_i);

      _mm256_storeu_ps(out + (i << 3), _mm256_cvtepi32_ps(bits0));
      _mm256_storeu_ps(out + ((i + 1) << 3), _mm256_cvtepi32_ps(bits1));
      _mm256_storeu_ps(out + ((i + 2) << 3), _mm256_cvtepi32_ps(bits2));
      _mm256_storeu_ps(out + ((i + 3) << 3), _mm256_cvtepi32_ps(bits3));
    }

    for (; i < nbytes; ++i) {
      __m256i v = _mm256_set1_epi32(in[i]);
      __m256i bits = _mm256_srlv_epi32(v, shifts);
      bits = _mm256_and_si256(bits, one_i);
      __m256 vf = _mm256_cvtepi32_ps(bits);
      _mm256_storeu_ps(out + (i << 3), vf);
    }
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  static inline void unpack(const uint8_t *in, float *out, std::size_t d) {
    const std::size_t nbytes = d >> 3;

    const int32x4_t shifts_hi = {-7, -6, -5, -4};
    const int32x4_t shifts_lo = {-3, -2, -1, 0};
    const uint32x4_t one = vdupq_n_u32(1u);

    for (std::size_t i = 0; i < nbytes; ++i) {
      uint32x4_t v = vdupq_n_u32(in[i]);

      uint32x4_t b0 = vshlq_u32(v, shifts_hi);
      uint32x4_t b1 = vshlq_u32(v, shifts_lo);

      b0 = vandq_u32(b0, one);
      b1 = vandq_u32(b1, one);

      float32x4_t f0 = vcvtq_f32_u32(b0);
      float32x4_t f1 = vcvtq_f32_u32(b1);

      vst1q_f32(out + (i << 3) + 0, f0);
      vst1q_f32(out + (i << 3) + 4, f1);
    }
  }
#else
  static inline void unpack(const uint8_t *in, float *out, size_t d) {
    for (size_t i = 0; i < d / 8; ++i) {
      uint8_t byte = in[i];
      for (size_t j = 0; j < 8; ++j) {
        out[i * 8 + j] = ((byte >> (7 - j)) & 1) ? 1.0f : 0.0f;
      }
    }
  }
#endif

  static inline Lorann::ColVector to_float_vector(const BinaryType *data, int d) {
    const uint8_t *raw = reinterpret_cast<const uint8_t *>(data);
    Lorann::ColVector vec(d);
    unpack(raw, vec.data(), d);
    return vec;
  }

  static inline Lorann::MappedMatrix to_float_matrix(const BinaryType *data, const int n,
                                                     const int d) {
    const uint8_t *raw = reinterpret_cast<const uint8_t *>(data);
    auto buf = std::shared_ptr<float[]>{new float[n * d], std::default_delete<float[]>()};
    unpack(raw, buf.get(), n * d);
    return {buf.get(), n, d, buf};
  }

  static inline float dot_product(const BinaryType *x, const BinaryType *y, const size_t n_bytes) {
    const simsimd_b8_t *a = reinterpret_cast<const simsimd_b8_t *>(x);
    const simsimd_b8_t *b = reinterpret_cast<const simsimd_b8_t *>(y);

    simsimd_distance_t distance;
    simsimd_hamming_b8(a, b, n_bytes, &distance);
    return distance;
  }

  static inline float squared_euclidean(const BinaryType *x, const BinaryType *y,
                                        const size_t n_bytes) {
    const simsimd_b8_t *a = reinterpret_cast<const simsimd_b8_t *>(x);
    const simsimd_b8_t *b = reinterpret_cast<const simsimd_b8_t *>(y);

    simsimd_distance_t distance;
    simsimd_hamming_b8(a, b, n_bytes, &distance);
    return distance;
  }
};

}  // namespace detail

}  // namespace Lorann