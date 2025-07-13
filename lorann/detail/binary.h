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

#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
  static inline float dot_product(const BinaryType *x, const BinaryType *y, const size_t n_bytes) {
    const uint8_t *a = reinterpret_cast<const uint8_t *>(x);
    const uint8_t *b = reinterpret_cast<const uint8_t *>(y);

    const std::size_t step = 64;
    const std::size_t n_chunks = n_bytes / step;

    __m512i acc = _mm512_setzero_si512();

    for (std::size_t blk = 0; blk < n_chunks; ++blk) {
      const __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + blk * step));
      const __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + blk * step));
      const __m512i vand = _mm512_and_si512(va, vb);
      const __m512i pc64 = _mm512_popcnt_epi64(vand);
      acc = _mm512_add_epi64(acc, pc64);
    }

#ifdef __AVX512VL__
    uint64_t sum = static_cast<uint64_t>(_mm512_reduce_add_epi64(acc));
#else
    const __m256i lo = _mm512_castsi512_si256(acc);
    const __m256i hi = _mm512_extracti64x4_epi64(acc, 1);
    const __m256i sum256 = _mm256_add_epi64(lo, hi);

    uint64_t sum = static_cast<uint64_t>(_mm256_extract_epi64(sum256, 0)) +
                   static_cast<uint64_t>(_mm256_extract_epi64(sum256, 1)) +
                   static_cast<uint64_t>(_mm256_extract_epi64(sum256, 2)) +
                   static_cast<uint64_t>(_mm256_extract_epi64(sum256, 3));
#endif
    for (std::size_t i = n_chunks * step; i < n_bytes; ++i) {
      sum += static_cast<uint64_t>(POPCOUNT(a[i] & b[i]));
    }

    return static_cast<float>(sum);
  }
#elif defined(__AVX2__)
  static inline float dot_product(const BinaryType *x, const BinaryType *y, const size_t n_bytes) {
    const uint8_t *a = reinterpret_cast<const uint8_t *>(x);
    const uint8_t *b = reinterpret_cast<const uint8_t *>(y);

    const std::size_t step = 32;
    const std::size_t n_chunks = n_bytes / step;

    const __m256i lut = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2,
                                         1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);

    const __m256i lo_mask = _mm256_set1_epi8(0x0F);
    const __m256i vzero = _mm256_setzero_si256();

    __m256i acc = _mm256_setzero_si256();

    for (std::size_t blk = 0; blk < n_chunks; ++blk) {
      const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + blk * step));
      const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + blk * step));
      const __m256i vand = _mm256_and_si256(va, vb);

      const __m256i lo_nib = _mm256_and_si256(vand, lo_mask);
      const __m256i hi_nib = _mm256_and_si256(_mm256_srli_epi16(vand, 4), lo_mask);

      const __m256i pc_lo = _mm256_shuffle_epi8(lut, lo_nib);
      const __m256i pc_hi = _mm256_shuffle_epi8(lut, hi_nib);
      const __m256i pc_byte = _mm256_add_epi8(pc_lo, pc_hi);

      acc = _mm256_add_epi64(acc, _mm256_sad_epu8(pc_byte, vzero));
    }

    uint64_t sum = static_cast<uint64_t>(_mm256_extract_epi64(acc, 0)) +
                   static_cast<uint64_t>(_mm256_extract_epi64(acc, 1)) +
                   static_cast<uint64_t>(_mm256_extract_epi64(acc, 2)) +
                   static_cast<uint64_t>(_mm256_extract_epi64(acc, 3));

    for (std::size_t i = n_chunks * step; i < n_bytes; ++i) {
      sum += static_cast<uint64_t>(POPCOUNT(a[i] & b[i]));
    }

    return static_cast<float>(sum);
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  static inline float dot_product(const BinaryType *x, const BinaryType *y, const size_t n_bytes) {
    const uint8_t *a = reinterpret_cast<const uint8_t *>(x);
    const uint8_t *b = reinterpret_cast<const uint8_t *>(y);

    const std::size_t step = 16;
    const std::size_t n_chunks = n_bytes / step;

    uint32x4_t acc32 = vdupq_n_u32(0);

    for (std::size_t blk = 0; blk < n_chunks; ++blk) {
      const uint8x16_t va = vld1q_u8(a + blk * step);
      const uint8x16_t vb = vld1q_u8(b + blk * step);
      const uint8x16_t vand = vandq_u8(va, vb);

      const uint8x16_t pc8 = vcntq_u8(vand);
      const uint16x8_t pc16 = vpaddlq_u8(pc8);

      acc32 = vpadalq_u16(acc32, pc16);
    }

#if defined(__aarch64__)
    uint64_t sum = static_cast<uint64_t>(vaddvq_u32(acc32));
#else
    uint64x2_t acc64 = vpaddlq_u32(acc32);
    uint64_t sum = vgetq_lane_u64(acc64, 0) + vgetq_lane_u64(acc64, 1);
#endif

    for (std::size_t i = n_chunks * step; i < n_bytes; ++i) {
      sum += static_cast<uint64_t>(POPCOUNT(a[i] & b[i]));
    }

    return static_cast<float>(sum);
  }
#else
  static inline float dot_product(const BinaryType *x, const BinaryType *y, const size_t n_bytes) {
    const uint8_t *a = reinterpret_cast<const uint8_t *>(x);
    const uint8_t *b = reinterpret_cast<const uint8_t *>(y);

    int sum = 0;
    for (int i = 0; i < n_bytes; ++i) {
      const uint8_t and_result = a[i] & b[i];
      sum += POPCOUNT(and_result);
    }

    return static_cast<float>(sum);
  }
#endif

#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
  static inline float squared_euclidean(const BinaryType *x, const BinaryType *y,
                                        const size_t n_bytes) {
    const uint8_t *a = reinterpret_cast<const uint8_t *>(x);
    const uint8_t *b = reinterpret_cast<const uint8_t *>(y);

    const std::size_t step = 64;
    const std::size_t n_chunks = n_bytes / step;

    __m512i acc = _mm512_setzero_si512();

    for (std::size_t blk = 0; blk < n_chunks; ++blk) {
      const __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + blk * step));
      const __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + blk * step));
      const __m512i vxor = _mm512_xor_si512(va, vb);
      const __m512i pc64 = _mm512_popcnt_epi64(vxor);
      acc = _mm512_add_epi64(acc, pc64);
    }

#ifdef __AVX512VL__
    uint64_t sum = static_cast<uint64_t>(_mm512_reduce_add_epi64(acc));
#else
    const __m256i lo = _mm512_castsi512_si256(acc);
    const __m256i hi = _mm512_extracti64x4_epi64(acc, 1);
    const __m256i sum256 = _mm256_add_epi64(lo, hi);

    uint64_t sum = static_cast<uint64_t>(_mm256_extract_epi64(sum256, 0)) +
                   static_cast<uint64_t>(_mm256_extract_epi64(sum256, 1)) +
                   static_cast<uint64_t>(_mm256_extract_epi64(sum256, 2)) +
                   static_cast<uint64_t>(_mm256_extract_epi64(sum256, 3));
#endif
    for (std::size_t i = n_chunks * step; i < n_bytes; ++i) {
      sum += static_cast<uint64_t>(POPCOUNT(a[i] ^ b[i]));
    }

    return static_cast<float>(sum);
  }
#elif defined(__AVX2__)
  static inline float squared_euclidean(const BinaryType *x, const BinaryType *y,
                                        const size_t n_bytes) {
    const uint8_t *a = reinterpret_cast<const uint8_t *>(x);
    const uint8_t *b = reinterpret_cast<const uint8_t *>(y);

    const std::size_t step = 32;
    const std::size_t n_chunks = n_bytes / step;

    const __m256i lut = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2,
                                         1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);

    const __m256i lo_mask = _mm256_set1_epi8(0x0F);
    const __m256i vzero = _mm256_setzero_si256();
    __m256i acc = _mm256_setzero_si256();

    for (std::size_t blk = 0; blk < n_chunks; ++blk) {
      const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + blk * step));
      const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + blk * step));
      const __m256i vxor = _mm256_xor_si256(va, vb);

      const __m256i lo_nib = _mm256_and_si256(vxor, lo_mask);
      const __m256i hi_nib = _mm256_and_si256(_mm256_srli_epi16(vxor, 4), lo_mask);

      const __m256i pc_lo = _mm256_shuffle_epi8(lut, lo_nib);
      const __m256i pc_hi = _mm256_shuffle_epi8(lut, hi_nib);
      const __m256i pc_byte = _mm256_add_epi8(pc_lo, pc_hi);

      acc = _mm256_add_epi64(acc, _mm256_sad_epu8(pc_byte, vzero));
    }

    uint64_t sum = static_cast<uint64_t>(_mm256_extract_epi64(acc, 0)) +
                   static_cast<uint64_t>(_mm256_extract_epi64(acc, 1)) +
                   static_cast<uint64_t>(_mm256_extract_epi64(acc, 2)) +
                   static_cast<uint64_t>(_mm256_extract_epi64(acc, 3));

    for (std::size_t i = n_chunks * step; i < n_bytes; ++i) {
      sum += static_cast<uint64_t>(POPCOUNT(a[i] ^ b[i]));
    }

    return static_cast<float>(sum);
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  static inline float squared_euclidean(const BinaryType *x, const BinaryType *y,
                                        const size_t n_bytes) {
    const uint8_t *a = reinterpret_cast<const uint8_t *>(x);
    const uint8_t *b = reinterpret_cast<const uint8_t *>(y);

    const std::size_t step = 16;
    const std::size_t n_chunks = n_bytes / step;

    uint32x4_t acc32 = vdupq_n_u32(0);

    for (std::size_t blk = 0; blk < n_chunks; ++blk) {
      const uint8x16_t va = vld1q_u8(a + blk * step);
      const uint8x16_t vb = vld1q_u8(b + blk * step);
      const uint8x16_t vxor = veorq_u8(va, vb);

      const uint8x16_t pc8 = vcntq_u8(vxor);
      const uint16x8_t pc16 = vpaddlq_u8(pc8);

      acc32 = vpadalq_u16(acc32, pc16);
    }

#if defined(__aarch64__)
    uint64_t sum = static_cast<uint64_t>(vaddvq_u32(acc32));
#else
    uint64x2_t acc64 = vpaddlq_u32(acc32);
    uint64_t sum = vgetq_lane_u64(acc64, 0) + vgetq_lane_u64(acc64, 1);
#endif

    for (std::size_t i = n_chunks * step; i < n_bytes; ++i) {
      sum += static_cast<uint64_t>(POPCOUNT(a[i] ^ b[i]));
    }

    return static_cast<float>(sum);
  }
#else
  static inline float squared_euclidean(const BinaryType *x, const BinaryType *y,
                                        const size_t n_bytes) {
    const uint8_t *a = reinterpret_cast<const uint8_t *>(x);
    const uint8_t *b = reinterpret_cast<const uint8_t *>(y);

    int sum = 0;
    for (int i = 0; i < n_bytes; ++i) {
      const uint8_t and_result = a[i] ^ b[i];
      sum += POPCOUNT(and_result);
    }

    return static_cast<float>(sum);
  }
#endif
};

}  // namespace detail

}  // namespace Lorann