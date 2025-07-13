#pragma once

#include <cstdint>
#include <memory>

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
struct Traits<uint8_t> {
  static constexpr TypeMarker type_marker = UINT8;
  static constexpr int dim_divisor = 1;

  static inline Lorann::ColVector to_float_vector(const uint8_t *x, int d) {
    return Eigen::Map<const Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>>(x, d).cast<float>();
  }

  static inline Lorann::MappedMatrix to_float_matrix(const uint8_t *data, const int n,
                                                     const int d) {
    auto buf = std::shared_ptr<float[]>{new float[n * d], std::default_delete<float[]>()};
    for (int i = 0; i < n * d; ++i) {
      buf[i] = static_cast<float>(data[i]);
    }
    return {buf.get(), n, d, buf};
  }

#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
  static inline float dot_product(const uint8_t *a, const uint8_t *b, const size_t dim) {
    __m512i acc_dp = _mm512_setzero_si512();
    __m512i acc_a_sum = _mm512_setzero_si512();

    const __m512i ones = _mm512_set1_epi8(1);
    const __m512i offset_128 = _mm512_set1_epi8(128);

    size_t i = 0;
    size_t limit = dim - (dim % 64);

    for (; i < limit; i += 64) {
      const __m512i a_vec = _mm512_loadu_si512(a + i);
      const __m512i b_vec = _mm512_loadu_si512(b + i);
      const __m512i b_s8_equiv = _mm512_sub_epi8(b_vec, offset_128);

      acc_dp = _mm512_dpbusd_epi32(acc_dp, a_vec, b_s8_equiv);
      acc_a_sum = _mm512_dpbusd_epi32(acc_a_sum, a_vec, ones);
    }

    const int32_t dp_sum = _mm512_reduce_add_epi32(acc_dp);
    const int32_t a_sum = _mm512_reduce_add_epi32(acc_a_sum);

    int32_t total_sum = dp_sum + 128 * a_sum;

    for (; i < dim; ++i) {
      total_sum += static_cast<int32_t>(a[i]) * b[i];
    }

    return static_cast<float>(total_sum);
  }
#elif defined(__AVX512F__)
  static inline int32_t hsum_epi32(__m512i v) {
    __m256i lo256 = _mm512_castsi512_si256(v);
    __m256i hi256 = _mm512_extracti32x8_epi32(v, 1);
    __m256i sum256 = _mm256_add_epi32(lo256, hi256);

    __m128i lo128 = _mm256_castsi256_si128(sum256);
    __m128i hi128 = _mm256_extracti128_si256(sum256, 1);
    __m128i sum128 = _mm_add_epi32(lo128, hi128);

    sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 4));
    return _mm_cvtsi128_si32(sum128);
  }

  static inline int32_t dot_product(const uint8_t *a, const uint8_t *b, const std::size_t dim) {
    const __m512i v_ones = _mm512_set1_epi16(1);
    __m512i acc32 = _mm512_setzero_si512();

    std::size_t i = 0;
    for (; i + 64 <= dim; i += 64) {
      __m512i va8 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + i));
      __m512i vb8 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + i));

      __m512i va16_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(va8));
      __m512i vb16_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(vb8));
      __m512i va16_hi = _mm512_cvtepu8_epi16(_mm512_extracti32x8_epi32(va8, 1));
      __m512i vb16_hi = _mm512_cvtepu8_epi16(_mm512_extracti32x8_epi32(vb8, 1));

      __m512i prod_lo = _mm512_mullo_epi16(va16_lo, vb16_lo);
      __m512i prod_hi = _mm512_mullo_epi16(va16_hi, vb16_hi);

      acc32 = _mm512_add_epi32(acc32, _mm512_madd_epi16(prod_lo, v_ones));
      acc32 = _mm512_add_epi32(acc32, _mm512_madd_epi16(prod_hi, v_ones));
    }

    int32_t result = hsum_epi32(acc32);

    for (; i < dim; ++i) {
      result += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }

    return static_cast<float> result;
  }
#elif defined(__AVX2__)
  static inline float dot_product(const uint8_t *a, const uint8_t *b, std::size_t dim) {
    const std::size_t kSimd = 32;
    const std::size_t kPair16 = kSimd / 2;

    __m256i acc32 = _mm256_setzero_si256();

    while (dim >= kSimd) {
      __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a));
      __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b));

      {
        __m128i loa = _mm256_castsi256_si128(va);
        __m128i lob = _mm256_castsi256_si128(vb);

        __m256i wa = _mm256_cvtepu8_epi16(loa);
        __m256i wb = _mm256_cvtepu8_epi16(lob);

        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(wa, wb));
      }
      {
        __m128i hia = _mm256_extracti128_si256(va, 1);
        __m128i hib = _mm256_extracti128_si256(vb, 1);

        __m256i wa = _mm256_cvtepu8_epi16(hia);
        __m256i wb = _mm256_cvtepu8_epi16(hib);

        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(wa, wb));
      }

      a += kSimd;
      b += kSimd;
      dim -= kSimd;
    }

    __m128i sum128 =
        _mm_add_epi32(_mm256_castsi256_si128(acc32), _mm256_extracti128_si256(acc32, 1));
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    int32_t result = _mm_cvtsi128_si32(sum128);

    while (dim--) {
      result += static_cast<int32_t>(*a++) * static_cast<int32_t>(*b++);
    }

    return static_cast<float>(result);
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  static inline float dot_product(const uint8_t *a, const uint8_t *b, const size_t dim) {
    std::size_t i = 0;
    uint32x4_t acc = vdupq_n_u32(0);

#if defined(__ARM_FEATURE_DOTPROD)
    for (; i + 64 <= dim; i += 64) {
      uint8x16_t a0 = vld1q_u8(a + i);
      uint8x16_t b0 = vld1q_u8(b + i);
      uint8x16_t a1 = vld1q_u8(a + i + 16);
      uint8x16_t b1 = vld1q_u8(b + i + 16);
      uint8x16_t a2 = vld1q_u8(a + i + 32);
      uint8x16_t b2 = vld1q_u8(b + i + 32);
      uint8x16_t a3 = vld1q_u8(a + i + 48);
      uint8x16_t b3 = vld1q_u8(b + i + 48);

      acc = vdotq_u32(acc, a0, b0);
      acc = vdotq_u32(acc, a1, b1);
      acc = vdotq_u32(acc, a2, b2);
      acc = vdotq_u32(acc, a3, b3);
    }
#endif

    for (; i + 16 <= dim; i += 16) {
      uint8x16_t va = vld1q_u8(a + i);
      uint8x16_t vb = vld1q_u8(b + i);

      uint16x8_t mul_lo = vmull_u8(vget_low_u8(va), vget_low_u8(vb));
      uint16x8_t mul_hi = vmull_u8(vget_high_u8(va), vget_high_u8(vb));

      acc = vpadalq_u16(acc, mul_lo);
      acc = vpadalq_u16(acc, mul_hi);
    }

    uint32_t scalar = vaddvq_u32(acc);
    for (; i < dim; ++i) {
      scalar += static_cast<uint32_t>(a[i]) * static_cast<uint32_t>(b[i]);
    }

    return static_cast<float>(scalar);
  }
#else
  static inline float dot_product(const uint8_t *a, const uint8_t *b, const size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
      sum += static_cast<float>(a[i]) * static_cast<float>(b[i]);
    }
    return sum;
  }
#endif

#if defined(__AVX512F__)
  static inline float squared_euclidean(const uint8_t *a, const uint8_t *b, const std::size_t dim) {
    std::size_t i = 0;
    __m512i acc = _mm512_setzero_si512();

    for (; i + 63 < dim; i += 64) {
      __m512i va = _mm512_loadu_si512(a + i);
      __m512i vb = _mm512_loadu_si512(b + i);

      __m256i va_lo = _mm512_castsi512_si256(va);
      __m256i vb_lo = _mm512_castsi512_si256(vb);

      __m512i a16_lo = _mm512_cvtepu8_epi16(va_lo);
      __m512i b16_lo = _mm512_cvtepu8_epi16(vb_lo);
      __m512i d16_lo = _mm512_sub_epi16(a16_lo, b16_lo);
      __m512i s32_lo = _mm512_madd_epi16(d16_lo, d16_lo);
      acc = _mm512_add_epi32(acc, s32_lo);

      __m256i va_hi = _mm512_extracti64x4_epi64(va, 1);
      __m256i vb_hi = _mm512_extracti64x4_epi64(vb, 1);

      __m512i a16_hi = _mm512_cvtepu8_epi16(va_hi);
      __m512i b16_hi = _mm512_cvtepu8_epi16(vb_hi);
      __m512i d16_hi = _mm512_sub_epi16(a16_hi, b16_hi);
      __m512i s32_hi = _mm512_madd_epi16(d16_hi, d16_hi);
      acc = _mm512_add_epi32(acc, s32_hi);
    }

    int32_t result = _mm512_reduce_add_epi32(acc);

    for (; i < dim; ++i) {
      int16_t d = static_cast<int16_t>(a[i]) - static_cast<int16_t>(b[i]);
      result += d * d;
    }

    return static_cast<float>(result);
  }
#elif defined(__AVX2__)
  static inline float squared_euclidean(const uint8_t *a, const uint8_t *b, const std::size_t dim) {
    std::size_t i = 0;
    __m256i acc = _mm256_setzero_si256();

    for (; i + 31 < dim; i += 32) {
      __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
      __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));

      __m128i va_lo8 = _mm256_castsi256_si128(va);
      __m128i vb_lo8 = _mm256_castsi256_si128(vb);
      __m256i va_lo16 = _mm256_cvtepu8_epi16(va_lo8);
      __m256i vb_lo16 = _mm256_cvtepu8_epi16(vb_lo8);
      __m256i diff_lo = _mm256_sub_epi16(va_lo16, vb_lo16);
      __m256i sq_lo = _mm256_madd_epi16(diff_lo, diff_lo);
      acc = _mm256_add_epi32(acc, sq_lo);

      __m128i va_hi8 = _mm256_extracti128_si256(va, 1);
      __m128i vb_hi8 = _mm256_extracti128_si256(vb, 1);
      __m256i va_hi16 = _mm256_cvtepu8_epi16(va_hi8);
      __m256i vb_hi16 = _mm256_cvtepu8_epi16(vb_hi8);
      __m256i diff_hi = _mm256_sub_epi16(va_hi16, vb_hi16);
      __m256i sq_hi = _mm256_madd_epi16(diff_hi, diff_hi);
      acc = _mm256_add_epi32(acc, sq_hi);
    }

    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 4));
    int32_t result = _mm_cvtsi128_si32(sum128);

    for (; i < dim; ++i) {
      int16_t d = static_cast<int16_t>(a[i]) - static_cast<int16_t>(b[i]);
      result += d * d;
    }

    return static_cast<float>(result);
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  static inline float squared_euclidean(const uint8_t *a, const uint8_t *b, const std::size_t dim) {
    std::size_t i = 0;
    uint32x4_t acc = vdupq_n_u32(0);

    for (; i + 15 < dim; i += 16) {
      uint8x16_t va = vld1q_u8(a + i);
      uint8x16_t vb = vld1q_u8(b + i);

      uint8x16_t diff = vabdq_u8(va, vb);

      uint16x8_t d_lo = vmovl_u8(vget_low_u8(diff));
      uint16x8_t d_hi = vmovl_u8(vget_high_u8(diff));

      uint32x4_t sq0 = vmull_u16(vget_low_u16(d_lo), vget_low_u16(d_lo));
      uint32x4_t sq1 = vmull_u16(vget_high_u16(d_lo), vget_high_u16(d_lo));
      acc = vaddq_u32(acc, sq0);
      acc = vaddq_u32(acc, sq1);

      uint32x4_t sq2 = vmull_u16(vget_low_u16(d_hi), vget_low_u16(d_hi));
      uint32x4_t sq3 = vmull_u16(vget_high_u16(d_hi), vget_high_u16(d_hi));
      acc = vaddq_u32(acc, sq2);
      acc = vaddq_u32(acc, sq3);
    }

    uint64x2_t sum64 = vpaddlq_u32(acc);
    uint64_t total64 = vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1);
    uint32_t result = static_cast<uint32_t>(total64);

    for (; i < dim; ++i) {
      int16_t d = static_cast<int16_t>(a[i]) - static_cast<int16_t>(b[i]);
      result += static_cast<uint32_t>(d * d);
    }

    return static_cast<float>(result);
  }
#else
  static inline float squared_euclidean(const uint8_t *a, const uint8_t *b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) {
      const float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
      s += diff * diff;
    }
    return s;
  }
#endif
};

}  // namespace detail

}  // namespace Lorann