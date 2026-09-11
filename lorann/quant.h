#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

#include "utils.h"

namespace Lorann {

namespace joint_quantization {
template <int Bits, bool MatrixB>
inline void quantize_matrix(const ColMatrix &matrix, uint8_t *out, float *factors);
}  // namespace joint_quantization

#if defined(__AVX2__)

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
#define MM512_SET_M256I(a, b) _mm512_inserti64x4(_mm512_castsi256_si512(b), (a), 1)

LORANN_ALWAYS_INLINE inline __m128i unpack128(const uint8_t *rsi) {
  const __m128i bytes = _mm_loadl_epi64((const __m128i *)rsi);
  const __m128i lo = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), _mm_set1_epi8(0x0F));
  return _mm_or_si128(lo, _mm_slli_si128(hi, 8));
}

LORANN_ALWAYS_INLINE inline __m256i unpack256(const uint8_t *rsi) {
  const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
  const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
  const __m256i low_mask = _mm256_set1_epi8(0xF);
  return _mm256_and_si256(low_mask, bytes);
}

#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
LORANN_ALWAYS_INLINE inline __m512i unpack512(const uint8_t *rsi) {
  const __m256i tmp = _mm256_loadu_si256((const __m256i *)rsi);
  const __m256i shifted = _mm256_srli_epi16(tmp, 4);
  const __m512i bytes = MM512_SET_M256I(shifted, tmp);
  const __m512i low_mask = _mm512_set1_epi8(0xF);
  return _mm512_and_si512(low_mask, bytes);
}
#endif

LORANN_ALWAYS_INLINE inline __m128i dpbusd(const __m128i a, const __m128i b) {
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
  __m128i sum = _mm_setzero_si128();
  asm("vpdpbusd %2, %1, %0" : "+x"(sum) : "x"(a), "mx"(b));
#else
  const __m128i dot = _mm_maddubs_epi16(a, b);
  const __m128i ones = _mm_set1_epi16(1);
  const __m128i sum = _mm_madd_epi16(ones, dot);
#endif
  return sum;
}

LORANN_ALWAYS_INLINE inline __m128i dpbusd(__m128i c, const __m128i a, const __m128i b) {
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
  asm("vpdpbusd %2, %1, %0" : "+x"(c) : "x"(a), "mx"(b));
#else
  const __m128i dot = _mm_maddubs_epi16(a, b);
  const __m128i ones = _mm_set1_epi16(1);
  const __m128i sum = _mm_madd_epi16(ones, dot);
  c = _mm_add_epi32(sum, c);
#endif
  return c;
}

LORANN_ALWAYS_INLINE inline __m256i dpbusd(const __m256i a, const __m256i b) {
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
  __m256i sum = _mm256_setzero_si256();
  asm("vpdpbusd %2, %1, %0" : "+x"(sum) : "x"(a), "mx"(b));
#else
  const __m256i dot = _mm256_maddubs_epi16(a, b);
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i sum = _mm256_madd_epi16(ones, dot);
#endif
  return sum;
}

LORANN_ALWAYS_INLINE inline __m256i dpbusd(__m256i c, const __m256i a, const __m256i b) {
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
  asm("vpdpbusd %2, %1, %0" : "+x"(c) : "x"(a), "mx"(b));
#else
  const __m256i dot = _mm256_maddubs_epi16(a, b);
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i sum = _mm256_madd_epi16(ones, dot);
  c = _mm256_add_epi32(sum, c);
#endif
  return c;
}

#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
LORANN_ALWAYS_INLINE inline __m512i dpbusd(const __m512i a, const __m512i b) {
  __m512i sum = _mm512_setzero_si512();
  asm("vpdpbusd %2, %1, %0" : "+x"(sum) : "x"(a), "mx"(b));
  return sum;
}

LORANN_ALWAYS_INLINE inline __m512i dpbusd(__m512i c, const __m512i a, const __m512i b) {
  asm("vpdpbusd %2, %1, %0" : "+x"(c) : "x"(a), "mx"(b));
  return c;
}
#endif

#endif

#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__ARM_FEATURE_DOTPROD)
#pragma GCC push_options
#if defined(__ARM_FEATURE_MATMUL_INT8)
#pragma GCC target("arch=armv8.2-a+dotprod+i8mm")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod+i8mm"))), \
                             apply_to = function)
#else
#pragma GCC target("arch=armv8.2-a+dotprod")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod"))), apply_to = function)
#endif
#endif

template <int Bits>
struct SQQuantizer {
  static_assert(Bits == 4 || Bits == 8, "SQ quantizers require 4 or 8 bits");
  static constexpr int bits = Bits;
  static constexpr int compensation_factor = 1 << (Bits - 1);
  static constexpr int div_factor = 8 / Bits;
#if defined(__FMA__)
  void scale_result(float *LORANN_RESTRICT result, const float compensation,
                    const float *LORANN_RESTRICT scale, const float *LORANN_RESTRICT fix,
                    const float factor, const float correction, const int n) const {
    const int k_stride = 8;
    int i = 0;

    const __m256 v_comp = _mm256_set1_ps(compensation);
    const __m256 v_invf = _mm256_set1_ps(1.0f / factor);
    const __m256 v_corr = _mm256_set1_ps(correction);

    for (; i + k_stride <= n; i += k_stride) {
      __m256 v_res = _mm256_loadu_ps(result + i);
      __m256 v_sc = _mm256_loadu_ps(scale + i);
      __m256 v_fix = _mm256_loadu_ps(fix + i);

      v_res = _mm256_sub_ps(v_res, v_comp);
      v_res = _mm256_mul_ps(v_res, v_invf);
      __m256 v_rcps = _mm256_rcp_ps(v_sc);
      __m256 two = _mm256_set1_ps(2.0f);
      v_rcps = _mm256_mul_ps(v_rcps, _mm256_sub_ps(two, _mm256_mul_ps(v_sc, v_rcps)));
      v_res = _mm256_mul_ps(v_res, v_rcps);
      v_res = _mm256_fmadd_ps(v_corr, v_fix, v_res);
      _mm256_storeu_ps(result + i, v_res);
    }

    for (; i < n; ++i) {
      result[i] = (result[i] - compensation) / (factor * scale[i]) + correction * fix[i];
    }
  }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
  void scale_result(float *LORANN_RESTRICT result, const float compensation,
                    const float *LORANN_RESTRICT scale, const float *LORANN_RESTRICT fix,
                    const float factor, const float correction, const int n) const {
#if defined(__aarch64__)
    const float32x4_t vcomp = vdupq_n_f32(compensation);
    const float32x4_t vfact = vdupq_n_f32(factor);
    const float32x4_t vcorr = vdupq_n_f32(correction);

    int i = 0;
    for (; i + 15 < n; i += 16) {
      float32x4_t r0 = vld1q_f32(result + i + 0);
      float32x4_t r1 = vld1q_f32(result + i + 4);
      float32x4_t r2 = vld1q_f32(result + i + 8);
      float32x4_t r3 = vld1q_f32(result + i + 12);

      float32x4_t s0 = vld1q_f32(scale + i + 0);
      float32x4_t s1 = vld1q_f32(scale + i + 4);
      float32x4_t s2 = vld1q_f32(scale + i + 8);
      float32x4_t s3 = vld1q_f32(scale + i + 12);

      float32x4_t f0 = vld1q_f32(fix + i + 0);
      float32x4_t f1 = vld1q_f32(fix + i + 4);
      float32x4_t f2 = vld1q_f32(fix + i + 8);
      float32x4_t f3 = vld1q_f32(fix + i + 12);

      r0 = vsubq_f32(r0, vcomp);
      r1 = vsubq_f32(r1, vcomp);
      r2 = vsubq_f32(r2, vcomp);
      r3 = vsubq_f32(r3, vcomp);

      s0 = vmulq_f32(s0, vfact);
      s1 = vmulq_f32(s1, vfact);
      s2 = vmulq_f32(s2, vfact);
      s3 = vmulq_f32(s3, vfact);

      r0 = vdivq_f32(r0, s0);
      r1 = vdivq_f32(r1, s1);
      r2 = vdivq_f32(r2, s2);
      r3 = vdivq_f32(r3, s3);

#if defined(__ARM_FEATURE_FMA)
      r0 = vfmaq_f32(r0, f0, vcorr);
      r1 = vfmaq_f32(r1, f1, vcorr);
      r2 = vfmaq_f32(r2, f2, vcorr);
      r3 = vfmaq_f32(r3, f3, vcorr);
#else
      r0 = vmlaq_f32(r0, f0, vcorr);
      r1 = vmlaq_f32(r1, f1, vcorr);
      r2 = vmlaq_f32(r2, f2, vcorr);
      r3 = vmlaq_f32(r3, f3, vcorr);
#endif

      vst1q_f32(result + i + 0, r0);
      vst1q_f32(result + i + 4, r1);
      vst1q_f32(result + i + 8, r2);
      vst1q_f32(result + i + 12, r3);
    }

    for (; i < n; ++i) {
      result[i] = (result[i] - compensation) / (factor * scale[i]) + correction * fix[i];
    }
#else
    const float32x4_t vcomp = vdupq_n_f32(compensation);
    const float32x4_t vfact = vdupq_n_f32(factor);
    const float32x4_t vcorr = vdupq_n_f32(correction);

    int i = 0;
    for (; i + 7 < n; i += 8) {
      float32x4_t r0 = vld1q_f32(result + i + 0);
      float32x4_t r1 = vld1q_f32(result + i + 4);

      float32x4_t s0 = vld1q_f32(scale + i + 0);
      float32x4_t s1 = vld1q_f32(scale + i + 4);

      float32x4_t f0 = vld1q_f32(fix + i + 0);
      float32x4_t f1 = vld1q_f32(fix + i + 4);

      r0 = vsubq_f32(r0, vcomp);
      r1 = vsubq_f32(r1, vcomp);

      s0 = vmulq_f32(s0, vfact);
      s1 = vmulq_f32(s1, vfact);

      float32x4_t recip0 = vrecpeq_f32(s0);
      float32x4_t recip1 = vrecpeq_f32(s1);

      recip0 = vmulq_f32(recip0, vrecpsq_f32(s0, recip0));
      recip0 = vmulq_f32(recip0, vrecpsq_f32(s0, recip0));

      recip1 = vmulq_f32(recip1, vrecpsq_f32(s1, recip1));
      recip1 = vmulq_f32(recip1, vrecpsq_f32(s1, recip1));

      r0 = vmulq_f32(r0, recip0);
      r1 = vmulq_f32(r1, recip1);

      r0 = vmlaq_f32(r0, f0, vcorr);
      r1 = vmlaq_f32(r1, f1, vcorr);

      vst1q_f32(result + i + 0, r0);
      vst1q_f32(result + i + 4, r1);
    }

    for (; i < n; ++i) {
      result[i] = (result[i] - compensation) / (factor * scale[i]) + correction * fix[i];
    }
#endif
  }
#else
  void scale_result(float *LORANN_RESTRICT result, const float compensation,
                    const float *LORANN_RESTRICT scale, const float *LORANN_RESTRICT fix,
                    const float factor, const float correction, const int n) const {
    for (int i = 0; i < n; ++i) {
      result[i] = (result[i] - compensation) / (factor * scale[i]) + correction * fix[i];
    }
  }
#endif
 private:
#if defined(__AVX2__)
  template <int Bytes>
  static auto load_B_packet(const void *data) {
    if constexpr (Bytes == 16)
      return _mm_loadu_si128(static_cast<const __m128i *>(data));
    else if constexpr (Bytes == 32)
      return _mm256_loadu_si256(static_cast<const __m256i *>(data));
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    else
      return _mm512_loadu_si512(data);
#endif
  }

  template <int Coefficients>
  struct BQuery {
    static constexpr int batch_size = 1;
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    static constexpr int width = Coefficients;
#else
    static constexpr int width = Coefficients == 16 ? 16 : 32;
#endif
    static constexpr int chunks = Coefficients / width;
    using Packet = decltype(load_B_packet<width>(nullptr));
    Packet packets[chunks];

    explicit BQuery(const int8_t *query) {
      for (int part = 0; part < chunks; ++part)
        packets[part] = load_B_packet<width>(query + part * width);
    }

    static auto load_codes(const uint8_t *column, int part) {
      if constexpr (Bits == 8) {
        return load_B_packet<width>(column + part * width);
      } else if constexpr (Coefficients == width) {
        if constexpr (width == 16)
          return unpack128(column);
        else if constexpr (width == 32)
          return unpack256(column);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        else
          return unpack512(column);
#endif
      } else {
        // Low and high nibbles hold the first and second 32-coefficient halves.
        const auto packed = load_B_packet<32>(column);
        const auto mask = _mm256_set1_epi8(0x0f);
        return _mm256_and_si256(part == 0 ? packed : _mm256_srli_epi16(packed, 4), mask);
      }
    }

    template <int Columns>
    void score(const uint8_t *codes, float *result) const {
      for (int col = 0; col < Columns; ++col) {
        const uint8_t *column = codes + col * (Coefficients / div_factor);
        auto sum = dpbusd(load_codes(column, 0), packets[0]);
        for (int part = 1; part < chunks; ++part)
          sum = dpbusd(sum, load_codes(column, part), packets[part]);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        if constexpr (width == 64)
          result[col] = _mm512_reduce_add_epi32(sum);
        else
#endif
          result[col] = horizontal_add(sum);
      }
    }
  };
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  template <int Coefficients>
  struct BQuery {
    static constexpr int chunks = Coefficients / 16;
#if defined(__ARM_FEATURE_DOTPROD) && defined(__ARM_FEATURE_MATMUL_INT8)
    static constexpr int batch_size = 4;
    int8x16_t packets[chunks];

    explicit BQuery(const int8_t *query) {
      for (int part = 0; part < chunks; ++part) packets[part] = vld1q_s8(query + part * 16);
    }

    int32x4_t accumulate(int32x4_t sum, uint8x16_t codes, int part) const {
      return vusdotq_s32(sum, codes, packets[part]);
    }
#else
    static constexpr int batch_size = 2;
    int16x8_t low[chunks], high[chunks];

    explicit BQuery(const int8_t *query) {
      for (int part = 0; part < chunks; ++part) {
        const int8x16_t packet = vld1q_s8(query + part * 16);
        low[part] = vmovl_s8(vget_low_s8(packet));
        high[part] = vmovl_s8(vget_high_s8(packet));
      }
    }

    int32x4_t accumulate(int32x4_t sum, uint8x16_t codes, int part) const {
      const int16x8_t codes_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(codes)));
      const int16x8_t codes_high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(codes)));
      sum = vmlal_s16(sum, vget_low_s16(codes_low), vget_low_s16(low[part]));
      sum = vmlal_s16(sum, vget_high_s16(codes_low), vget_high_s16(low[part]));
      sum = vmlal_s16(sum, vget_low_s16(codes_high), vget_low_s16(high[part]));
      return vmlal_s16(sum, vget_high_s16(codes_high), vget_high_s16(high[part]));
    }
#endif

    static uint8x16_t load_codes(const uint8_t *column, int part) {
      if constexpr (Bits == 8) {
        return vld1q_u8(column + part * 16);
      } else if constexpr (Coefficients == 16) {
        const uint8x8_t packed = vld1_u8(column);
        return vcombine_u8(vand_u8(packed, vdup_n_u8(0x0f)), vshr_n_u8(packed, 4));
      } else {
        constexpr int half_chunks = Coefficients / 32;
        const uint8x16_t packed = vld1q_u8(column + (part % half_chunks) * 16);
        return part < half_chunks ? vandq_u8(packed, vdupq_n_u8(0x0f)) : vshrq_n_u8(packed, 4);
      }
    }

    template <int Columns>
    void score(const uint8_t *codes, float *result) const {
      int32x4_t sums[Columns];
      for (int col = 0; col < Columns; ++col) sums[col] = vdupq_n_s32(0);
      for (int part = 0; part < chunks; ++part)
        for (int col = 0; col < Columns; ++col)
          sums[col] = accumulate(sums[col],
                                 load_codes(codes + col * (Coefficients / div_factor), part), part);
      for (int col = 0; col < Columns; ++col)
        result[col] = static_cast<float>(vaddvq_s32(sums[col]));
    }
  };
#else
  template <int Coefficients>
  struct BQuery {
    static constexpr int batch_size = 1;
    const int8_t *query;

    explicit BQuery(const int8_t *query) : query(query) {}

    template <int Columns>
    void score(const uint8_t *codes, float *result) const {
      for (int col = 0; col < Columns; ++col) {
        int32_t sum = 0;
        const uint8_t *column = codes + col * (Coefficients / div_factor);
        for (int row = 0; row < Coefficients; ++row) {
          int code;
          if constexpr (Bits == 4)
            code = (column[row % (Coefficients / 2)] >> (row < Coefficients / 2 ? 0 : 4)) & 0xf;
          else
            code = column[row];
          sum += code * query[row];
        }
        result[col] = static_cast<float>(sum);
      }
    }
  };
#endif

 public:
  // Compute integer dot products, stored as floats, without quantization corrections.
  template <int Coefficients>
  inline void raw_matvec_product_B(const uint8_t *LORANN_RESTRICT A,
                                   const int8_t *LORANN_RESTRICT x, float *LORANN_RESTRICT result,
                                   const size_t cols) const {
    static_assert(Coefficients == 16 || Coefficients == 32 || Coefficients == 64,
                  "SQ B kernels require 16, 32, or 64 coefficients");
    const BQuery<Coefficients> query(x);
    constexpr int batch = BQuery<Coefficients>::batch_size;
    size_t col = 0;
    for (; col + batch <= cols; col += batch)
      query.template score<batch>(A + col * (Coefficients / div_factor), result + col);
    for (; col < cols; ++col)
      query.template score<1>(A + col * (Coefficients / div_factor), result + col);
  }

  // Apply scale, zero-point, and principal-axis corrections to the raw dot products.
  inline void quantized_matvec_product_B(const ColMatrixUInt8 &qA, const VectorInt8 &v,
                                         const Vector &correction, const float scale,
                                         const float factor, const float compensation,
                                         float *result) const {
    const float *scales = correction.data();
    const float *fix = correction.data() + qA.cols();

    const int rank = qA.rows() * div_factor;
    if (rank == 32)
      raw_matvec_product_B<32>(qA.data(), v.data(), result, qA.cols());
    else if (rank == 16)
      raw_matvec_product_B<16>(qA.data(), v.data(), result, qA.cols());
    else if (rank == 64)
      raw_matvec_product_B<64>(qA.data(), v.data(), result, qA.cols());
    else
      throw std::invalid_argument("SQ B kernels require 16, 32, or 64 coefficients");

    scale_result(result, compensation, scales, fix, scale, factor, qA.cols());
  }
};

struct SQ4Quantizer : SQQuantizer<4> {
#if defined(__AVX2__)
  inline void matvec_product_A(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                               float *LORANN_RESTRICT result, const size_t rows,
                               const size_t cols) const {
    for (size_t j = 0; j < cols; ++j) {
      __m256i sum = _mm256_setzero_si256();

      for (size_t i = 0; i < rows; i += 32) {
        const __m256i col_chunk = unpack256(A + (i + j * rows) / 2);
        const __m256i vec_chunk = _mm256_loadu_si256((const __m256i *)(x + i));
        sum = dpbusd(sum, col_chunk, vec_chunk);
      }

      result[j] = horizontal_add(sum);
    }
  }

#elif defined(__ARM_FEATURE_DOTPROD) && defined(__ARM_FEATURE_MATMUL_INT8)
  inline void matvec_product_A(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                               float *LORANN_RESTRICT result, const size_t rows,
                               const size_t cols) const {
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);

    for (size_t j = 0; j < cols; ++j) {
      int32x4_t acc = vdupq_n_s32(0);

      size_t i = 0;
      const uint8_t *Ap = A + ((j * rows) >> 1);

      for (; i + 32 <= rows; i += 32) {
        uint8x16_t packed = vld1q_u8(Ap);
        Ap += 16;

        uint8x16_t lo_u8 = vandq_u8(packed, mask_lo);
        uint8x16_t hi_u8 = vshrq_n_u8(packed, 4);

        int8x16_t x_lo = vld1q_s8(x + i);
        int8x16_t x_hi = vld1q_s8(x + i + 16);

        int8x16_t lo_s8 = vreinterpretq_s8_u8(lo_u8);
        int8x16_t hi_s8 = vreinterpretq_s8_u8(hi_u8);

        acc = vdotq_s32(acc, lo_s8, x_lo);
        acc = vdotq_s32(acc, hi_s8, x_hi);
      }

      int32_t sum = vaddvq_s32(acc);

      for (; i < rows; i += 2) {
        uint8_t byte = A[((i + j * rows) >> 1)];
        int32_t lo4 = (byte & 0x0F);
        int32_t hi4 = ((byte >> 4) & 0x0F);

        sum += lo4 * (int32_t)x[i];
        if (i + 1 < rows) sum += hi4 * (int32_t)x[i + 1];
      }

      result[j] = (float)sum;
    }
  }

#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
  inline void matvec_product_A(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                               float *LORANN_RESTRICT result, size_t rows, size_t cols) const {
    const uint8x16_t mask_lo4 = vdupq_n_u8(0x0F);

    for (size_t j = 0; j < cols; ++j) {
      int32x4_t acc0 = vdupq_n_s32(0);
      int32x4_t acc1 = vdupq_n_s32(0);
      int32x4_t acc2 = vdupq_n_s32(0);
      int32x4_t acc3 = vdupq_n_s32(0);

      size_t i = 0;
      const uint8_t *Ap = A + ((j * rows) >> 1);

      for (; i + 32 <= rows; i += 32) {
        uint8x16_t a_bytes = vld1q_u8(Ap);
        Ap += 16;

        uint8x16_t lo4_u8 = vandq_u8(a_bytes, mask_lo4);
        uint8x16_t hi4_u8 = vshrq_n_u8(a_bytes, 4);

        int8x16_t x_lo_s8 = vld1q_s8(x + i);
        int8x16_t x_hi_s8 = vld1q_s8(x + i + 16);

        int16x8_t lo4_0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(lo4_u8)));
        int16x8_t lo4_1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(lo4_u8)));
        int16x8_t hi4_0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(hi4_u8)));
        int16x8_t hi4_1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(hi4_u8)));

        int16x8_t x_lo_0 = vmovl_s8(vget_low_s8(x_lo_s8));
        int16x8_t x_lo_1 = vmovl_s8(vget_high_s8(x_lo_s8));
        int16x8_t x_hi_0 = vmovl_s8(vget_low_s8(x_hi_s8));
        int16x8_t x_hi_1 = vmovl_s8(vget_high_s8(x_hi_s8));

        acc0 = vmlal_s16(acc0, vget_low_s16(lo4_0), vget_low_s16(x_lo_0));
        acc1 = vmlal_s16(acc1, vget_high_s16(lo4_0), vget_high_s16(x_lo_0));
        acc2 = vmlal_s16(acc2, vget_low_s16(lo4_1), vget_low_s16(x_lo_1));
        acc3 = vmlal_s16(acc3, vget_high_s16(lo4_1), vget_high_s16(x_lo_1));

        acc0 = vmlal_s16(acc0, vget_low_s16(hi4_0), vget_low_s16(x_hi_0));
        acc1 = vmlal_s16(acc1, vget_high_s16(hi4_0), vget_high_s16(x_hi_0));
        acc2 = vmlal_s16(acc2, vget_low_s16(hi4_1), vget_low_s16(x_hi_1));
        acc3 = vmlal_s16(acc3, vget_high_s16(hi4_1), vget_high_s16(x_hi_1));
      }

      int32x4_t acc01 = vaddq_s32(acc0, acc1);
      int32x4_t acc23 = vaddq_s32(acc2, acc3);
      int32x4_t acc = vaddq_s32(acc01, acc23);

      int32_t sum = vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1) + vgetq_lane_s32(acc, 2) +
                    vgetq_lane_s32(acc, 3);

      for (; i < rows; i += 2) {
        uint8_t byte = A[((i + j * rows) >> 1)];
        int32_t lo4 = byte & 0x0F;
        int32_t hi4 = (byte >> 4) & 0x0F;

        sum += lo4 * (int32_t)x[i];
        if (i + 16 < rows) sum += hi4 * (int32_t)x[i + 16];
      }

      result[j] = (float)sum;
    }
  }

#else
  inline void matvec_product_A(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                               float *LORANN_RESTRICT result, const size_t rows,
                               const size_t cols) const {
    for (size_t j = 0; j < cols; ++j) {
      int32_t sum = 0;

      for (size_t i = 0; i < rows; i += 32) {
        for (int k = 0; k < 16; ++k) {
          sum += ((int32_t)(A[k + (i + j * rows) / 2] >> 4)) * ((int32_t)x[i + k + 16]);
          sum += ((int32_t)(A[k + (i + j * rows) / 2] & 0xF)) * ((int32_t)x[i + k]);
        }
      }

      result[j] = sum;
    }
  }

#endif

  inline void quantized_matvec_product_A(const ColMatrixUInt8 &qA, const VectorInt8 &v,
                                         const Vector &correction, const float scale,
                                         const float factor, const float compensation,
                                         float *result) const {
    const float *scales = correction.data();
    const float *fix = correction.data() + qA.cols();
    matvec_product_A(qA.data(), v.data(), result, qA.rows() * 2, qA.cols());
    scale_result(result, compensation, scales, fix, scale, factor, qA.cols());
  }

  inline float quantize_vector(const float *LORANN_RESTRICT v, const int len,
                               int8_t *LORANN_RESTRICT result) const {
    const float factor = compute_quantization_factor(v, len, 4);
    for (int i = 0; i < len; ++i) {
      result[i] = (int8_t)nearest_int(factor * v[i]);
    }
    return factor;
  }

  inline void quantize_matrix_B_unsigned(const ColMatrix &A, uint8_t *LORANN_RESTRICT result,
                                         float *LORANN_RESTRICT factors) const {
    joint_quantization::quantize_matrix<4, true>(A, result, factors);
  }

  inline void quantize_matrix_A_unsigned(const ColMatrix &A, uint8_t *LORANN_RESTRICT result,
                                         float *LORANN_RESTRICT factors) const {
    joint_quantization::quantize_matrix<4, false>(A, result, factors);
  }

  // Centroid codes use absmax scaling.
  inline void quantize_centroids_unsigned(const ColMatrix &A, uint8_t *LORANN_RESTRICT result,
                                          float *LORANN_RESTRICT factors) const {
    constexpr int qk = 32;
    const int n = A.rows();
    const int nb = n / qk;

    for (int i = 0; i < A.cols(); ++i) {
      const float *v = A.data() + i * n;
      const float factor = compute_quantization_factor(v, n, 4);
      for (int j = 0; j < nb; ++j) {
        for (int k = 0; k < qk / 2; ++k) {
          const uint8_t a = LORANN_MIN(15, factor * v[j * qk + 0 + k] + 8.5f);
          const uint8_t b = LORANN_MIN(15, factor * v[j * qk + qk / 2 + k] + 8.5f);

          result[i * n / 2 + j * qk / 2 + k] = a;
          result[i * n / 2 + j * qk / 2 + k] |= b << 4;
        }
      }
      factors[i] = factor;
    }
  }
};

struct SQ8Quantizer : SQQuantizer<8> {
#if defined(__AVX2__)
  inline void matvec_product_A(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                               float *LORANN_RESTRICT result, const size_t rows,
                               const size_t cols) const {
    for (size_t j = 0; j < cols; ++j) {
      __m256i sum = _mm256_setzero_si256();

      for (size_t i = 0; i < rows; i += 32) {
        const __m256i col_chunk = _mm256_loadu_si256((const __m256i *)(A + i + j * rows));
        const __m256i vec_chunk = _mm256_loadu_si256((const __m256i *)(x + i));
        sum = dpbusd(sum, col_chunk, vec_chunk);
      }

      result[j] = horizontal_add(sum);
    }
  }

#elif defined(__ARM_FEATURE_DOTPROD) && defined(__ARM_FEATURE_MATMUL_INT8)
  inline void matvec_product_A(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                               float *LORANN_RESTRICT result, const size_t rows,
                               const size_t cols) const {
    for (size_t j = 0; j < cols; ++j) {
      const uint8_t *a_col = A + j * rows;
      int32_t sum32 = 0;

      int32x4_t acc = vdupq_n_s32(0);

      size_t i = 0;
      for (; i + 32 <= rows; i += 32) {
        uint8x16_t a0 = vld1q_u8(a_col + i);
        uint8x16_t a1 = vld1q_u8(a_col + i + 16);

        int8x16_t x0 = vld1q_s8(x + i);
        int8x16_t x1 = vld1q_s8(x + i + 16);

        acc = vusdotq_s32(acc, a0, x0);
        acc = vusdotq_s32(acc, a1, x1);
      }

      sum32 += vaddvq_s32(acc);
      for (; i < rows; ++i) {
        sum32 += ((int32_t)a_col[i]) * ((int32_t)x[i]);
      }

      result[j] = (float)sum32;
    }
  }

#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
  inline void matvec_product_A(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                               float *LORANN_RESTRICT result, const size_t rows,
                               const size_t cols) const {
    for (size_t j = 0; j < cols; ++j) {
      const uint8_t *a_col = A + j * rows;
      int32_t sum32 = 0;

      int32x4_t acc0 = vdupq_n_s32(0);
      int32x4_t acc1 = vdupq_n_s32(0);

      size_t i = 0;
      for (; i + 16 <= rows; i += 16) {
        uint8x16_t a_u8 = vld1q_u8(a_col + i);
        int8x16_t x_s8 = vld1q_s8(x + i);

        uint16x8_t a0_u16 = vmovl_u8(vget_low_u8(a_u8));
        uint16x8_t a1_u16 = vmovl_u8(vget_high_u8(a_u8));

        int16x8_t x0_s16 = vmovl_s8(vget_low_s8(x_s8));
        int16x8_t x1_s16 = vmovl_s8(vget_high_s8(x_s8));

        int16x8_t a0_s16 = vreinterpretq_s16_u16(a0_u16);
        int16x8_t a1_s16 = vreinterpretq_s16_u16(a1_u16);

        acc0 = vmlal_s16(acc0, vget_low_s16(a0_s16), vget_low_s16(x0_s16));
        acc0 = vmlal_s16(acc0, vget_high_s16(a0_s16), vget_high_s16(x0_s16));

        acc1 = vmlal_s16(acc1, vget_low_s16(a1_s16), vget_low_s16(x1_s16));
        acc1 = vmlal_s16(acc1, vget_high_s16(a1_s16), vget_high_s16(x1_s16));
      }

      int32x4_t acc = vaddq_s32(acc0, acc1);
      sum32 += vaddvq_s32(acc);

      if (i + 8 <= rows) {
        uint8x8_t a_u8 = vld1_u8(a_col + i);
        int8x8_t x_s8 = vld1_s8(x + i);

        uint16x8_t a_u16 = vmovl_u8(a_u8);
        int16x8_t x_s16 = vmovl_s8(x_s8);
        int16x8_t a_s16 = vreinterpretq_s16_u16(a_u16);

        int32x4_t tmp = vdupq_n_s32(0);
        tmp = vmlal_s16(tmp, vget_low_s16(a_s16), vget_low_s16(x_s16));
        tmp = vmlal_s16(tmp, vget_high_s16(a_s16), vget_high_s16(x_s16));
        sum32 += vaddvq_s32(tmp);
        i += 8;
      }

      for (; i < rows; ++i) {
        sum32 += ((int32_t)a_col[i]) * ((int32_t)x[i]);
      }

      result[j] = (float)sum32;
    }
  }

#else
  inline void matvec_product_A(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                               float *LORANN_RESTRICT result, const size_t rows,
                               const size_t cols) const {
    for (size_t j = 0; j < cols; ++j) {
      int32_t sum = 0;

      for (size_t i = 0; i < rows; ++i) {
        sum += ((int32_t)A[i + j * rows]) * ((int32_t)x[i]);
      }

      result[j] = sum;
    }
  }

#endif

  inline void quantized_matvec_product_A(const ColMatrixUInt8 &qA, const VectorInt8 &v,
                                         const Vector &correction, const float scale,
                                         const float factor, const float compensation,
                                         float *result) const {
    const float *scales = correction.data();
    const float *fix = correction.data() + qA.cols();
    matvec_product_A(qA.data(), v.data(), result, qA.rows(), qA.cols());
    scale_result(result, compensation, scales, fix, scale, factor, qA.cols());
  }

  inline float quantize_vector(const float *LORANN_RESTRICT v, const int len,
                               int8_t *LORANN_RESTRICT result) const {
    const float factor = compute_quantization_factor(v, len, 8);
    for (int i = 0; i < len; ++i) {
      result[i] = (int8_t)nearest_int(factor * v[i]);
    }
    return factor;
  }

  inline float quantize_vector_unsigned(const float *LORANN_RESTRICT v, const int len,
                                        uint8_t *LORANN_RESTRICT result) const {
    const float factor = compute_quantization_factor(v, len, 8);
    for (int i = 0; i < len; ++i) {
      result[i] = (uint8_t)(nearest_int(factor * v[i]) + 128);
    }
    return factor;
  }

  inline void quantize_matrix_B_unsigned(const ColMatrix &A, uint8_t *LORANN_RESTRICT result,
                                         float *LORANN_RESTRICT factors) const {
    for (int i = 0; i < A.cols(); ++i) {
      factors[i] = quantize_vector_unsigned(A.data() + i * A.rows() + 1, A.rows() - 1,
                                            result + i * (A.rows() - 1));
    }
  }

  inline void quantize_matrix_A_unsigned(const ColMatrix &A, uint8_t *LORANN_RESTRICT result,
                                         float *LORANN_RESTRICT factors) const {
    for (int i = 0; i < A.cols(); ++i) {
      factors[i] =
          quantize_vector_unsigned(A.data() + i * A.rows(), A.rows(), result + i * A.rows());
    }
  }

  inline void quantize_centroids_unsigned(const ColMatrix &A, uint8_t *LORANN_RESTRICT result,
                                          float *LORANN_RESTRICT factors) const {
    quantize_matrix_A_unsigned(A, result, factors);
  }
};

#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__ARM_FEATURE_DOTPROD)
#pragma clang attribute pop
#pragma GCC pop_options
#endif

// Joint quantization fits integer codes and scales for the low-rank model A*B,
// then refines them using the error in the reconstructed model product.
//
// First, fit the quantized part of each A/B column v by minimizing
// ||v - s*k||^2, where k contains integer codes and s is the decoding scale.
// SQ4 uses all codes in [-8, 7] and searches both signs of s, allowing the extra
// level to serve either tail. The fitted scale can trade clipping of large
// coefficients for smaller rounding errors in the remaining coefficients.
// For fixed nonzero codes, the optimal scale is s = dot(v,k)/dot(k,k). Sweeping the
// intervals where rounding leaves the codes unchanged finds candidate scales
// analytically; candidates are checked with the stored floating-point factors.
//
// Next, hold the reconstructed A fixed and jointly refine each B column's codes,
// scale, and floating-point correction to reduce
// ||A*B - A_hat*B_hat||_F^2, where hats denote the reconstructed factors. This lets
// B compensate for quantization errors in A and accounts for interactions between
// latent coordinates. Up to four coordinate-descent sweeps accept improvements
// to the product error.
//
// Query kernels use the reciprocal scale 1/s. Protected coefficients are stored
// as floats. Fitting and refinement run during index construction.

namespace joint_quantization {

inline double finite_maximum(const float *values, int size) {
  double maximum = 0;
  for (int i = 0; i < size; ++i) {
    if (!std::isfinite(values[i]))
      throw std::invalid_argument("Model quantization requires finite values");
    maximum = std::max(maximum, std::abs(double(values[i])));
  }
  return maximum;
}

// Boundary streams advance monotonically, so only the root needs to sift down.
template <typename Boundary, size_t Count>
inline void replace_boundary_root(std::array<Boundary, Count> &heap, size_t size,
                                  const Boundary &next) {
  size_t parent = 0, child = 1;
  while (child < size) {
    if (child + 1 < size && heap[child + 1].scale < heap[child].scale) ++child;
    if (next.scale <= heap[child].scale) break;
    heap[parent] = heap[child];
    parent = child;
    child = 2 * parent + 1;
  }
  heap[parent] = next;
}

// Symmetric column fitter, also used to preserve exact SQ4 grids when tied.
// Sorted magnitudes give one boundary stream per integer level; scratch storage
// is reused across columns. FullRangeFitter below handles the asymmetric SQ4 range.
template <int Bits>
class ColumnFitter {
  static_assert(Bits == 4 || Bits == 8, "Model quantization supports SQ4 and SQ8");
  static constexpr int kLimit = (1 << (Bits - 1)) - 1;

  struct Boundary {
    double scale;
    double inverse;
    size_t index;
    int level;
  };

 public:
  // Keep both the decoding scale and its reciprocal normal: the SIMD decoder
  // uses an approximate reciprocal followed by a Newton step.
  static constexpr float kMinFactor = std::numeric_limits<float>::min();
  static constexpr float kMaxFactor = 1.f / kMinFactor;

  static int encode(float value, float factor) {
    const double scaled = std::clamp(double(value) * factor, -double(kLimit), double(kLimit));
    // Explicit symmetric nearest rounding, independent of the host rounding mode.
    const int magnitude = int(std::floor(std::abs(scaled) + 0.5));
    return scaled < 0 ? -magnitude : magnitude;
  }

  float fit(const float *values, int size) {
    const double maximum = finite_maximum(values, size);
    if (maximum == 0) return 1.f;
    const double energy = prepare_magnitudes(values, size, maximum);
    double best_scale = 1. / kLimit;
    double best_error = error(best_scale);
    refit_absmax(best_scale, best_error);
    if (best_error > 0) sweep_scales(energy, best_scale, best_error);
    return choose_stored_factor(values, size, maximum, best_scale);
  }

 private:
  double prepare_magnitudes(const float *values, int size, double maximum) {
    magnitudes_.clear();
    magnitudes_.reserve(size);
    double energy = 0;
    for (int i = 0; i < size; ++i) {
      if (values[i] == 0) continue;
      const double x = std::abs(double(values[i])) / maximum;
      magnitudes_.push_back(x);
      energy += x * x;
    }
    std::sort(magnitudes_.begin(), magnitudes_.end());

    return energy;
  }

  void refit_absmax(double &best_scale, double &best_error) const {
    // Analytically refitting the absmax codes supplies a tighter initial bound.
    double value_code_dot = 0, code_norm = 0;
    for (double x : magnitudes_) {
      const double code = std::floor(x * kLimit + 0.5);
      value_code_dot += x * code;
      code_norm += code * code;
    }
    const double refitted = value_code_dot / code_norm;
    const double refitted_error = error(refitted);
    if (refitted_error < best_error) {
      best_scale = refitted;
      best_error = refitted_error;
    }
  }

  double scale_lower_bound(double best_error) const {
    // Saturated coordinates alone give an error lower bound. Solve
    // sum_{x>t} (x-t)^2 = best_error for t = kLimit*s. Smaller scales cannot
    // improve the current result. Sorting makes each quadratic tail explicit.
    double tail_sum = 0, tail_energy = 0;
    for (size_t i = magnitudes_.size(); i > 0; --i) {
      const double x = magnitudes_[i - 1];
      tail_sum += x;
      tail_energy += x * x;
      const double count = magnitudes_.size() - i + 1;
      if (tail_energy <= best_error) continue;
      const double discriminant =
          std::max(0., tail_sum * tail_sum - count * (tail_energy - best_error));
      // Rationalized smaller root avoids subtracting nearly equal numbers.
      const double threshold = (tail_energy - best_error) / (tail_sum + std::sqrt(discriminant));
      if (i == 1 || threshold >= magnitudes_[i - 2]) {
        return std::nextafter(threshold / kLimit, 0.);
      }
    }

    return 0.;
  }

  size_t initialize_boundaries(double lower) {
    size_t heap_size = 0;
    for (int level = 0; level < kLimit; ++level) {
      const double inverse = 1. / (level + 0.5);
      const auto first =
          std::upper_bound(magnitudes_.begin(), magnitudes_.end(), lower,
                           [inverse](double bound, double x) { return bound < x * inverse; });
      const size_t index = first - magnitudes_.begin();
      if (index < magnitudes_.size())
        heap_[heap_size++] = {*first * inverse, inverse, index, level};
    }
    std::make_heap(heap_.begin(), heap_.begin() + heap_size,
                   [](const Boundary &a, const Boundary &b) { return a.scale > b.scale; });

    return heap_size;
  }

  void sweep_scales(double energy, double &best_scale, double &best_error) {
    double lower = scale_lower_bound(best_error);
    size_t heap_size = initialize_boundaries(lower);

    double value_code_dot = 0;
    double code_norm = 0;
    double zero_energy = 0;
    for (double x : magnitudes_) {
      int code =
          lower > 0 ? int(std::clamp(std::ceil(x / lower - 0.5), 0., double(kLimit))) : kLimit;
      // Match the boundary streams exactly, including ties at the lower bound.
      while (code > 0 && x * (1. / (code - 0.5)) <= lower) --code;
      while (code < kLimit && x * (1. / (code + 0.5)) > lower) ++code;
      value_code_dot += x * code;
      code_norm += code * code;
      if (code == 0) zero_energy += x * x;
    }
    double best_score = energy - best_error;
    for (;;) {
      const double upper = heap_size ? heap_[0].scale : std::numeric_limits<double>::infinity();
      // Most intervals do not contain their stationary point; reject them
      // using multiplication before paying for a division and error evaluation.
      if (code_norm > 0 && value_code_dot >= lower * code_norm &&
          value_code_dot <= upper * code_norm) {
        const double scale = value_code_dot / code_norm;
        if (value_code_dot * scale >= best_score) {
          const double candidate_error = error(scale);
          if (candidate_error < best_error) {
            best_scale = scale;
            best_error = candidate_error;
            best_score = energy - best_error;
          }
        }
      }
      // Coordinates rounded to zero stay zero as scale grows. Their energy
      // is another monotone lower bound, and can terminate the sweep early.
      if (heap_size == 0 || zero_energy >= best_error) break;

      Boundary next = heap_[0];
      const double x = magnitudes_[next.index];
      value_code_dot -= x;
      code_norm -= 2 * next.level + 1;
      lower = next.scale;
      if (next.level == 0) zero_energy += x * x;
      if (++next.index == magnitudes_.size()) {
        next = heap_[--heap_size];
        if (heap_size == 0) continue;
      } else {
        next.scale = magnitudes_[next.index] * next.inverse;
      }
      // Each stream advances monotonically: one root replacement/sift-down
      // suffices, instead of separate priority-queue pop and push operations.
      replace_boundary_root(heap_, heap_size, next);
    }
  }

  static float choose_stored_factor(const float *values, int size, double maximum,
                                    double best_scale) {
    // The ideal decoding scale need not have a representable float reciprocal.
    // Evaluate both adjacent representable factors, using the codes that will
    // actually be packed. Retain absmax as a guard against numerical roundoff.
    float factor = bounded_factor(double(kLimit) / maximum);
    double loss = stored_error(values, size, factor);
    const double ideal_factor = 1. / (best_scale * maximum);
    const float rounded = bounded_factor(ideal_factor);
    auto consider = [&](float candidate) {
      if (candidate < kMinFactor || candidate > kMaxFactor) return;
      const double candidate_loss = stored_error(values, size, candidate);
      if (candidate_loss < loss) {
        factor = candidate;
        loss = candidate_loss;
      }
    };
    consider(rounded);
    if (rounded > ideal_factor) consider(std::nextafter(rounded, 0.f));
    if (rounded < ideal_factor)
      consider(std::nextafter(rounded, std::numeric_limits<float>::infinity()));
    return factor;
  }

  static float bounded_factor(double factor) {
    return float(std::clamp(factor, double(kMinFactor), double(kMaxFactor)));
  }

  static double stored_error(const float *values, int size, float factor) {
    double result = 0;
    for (int i = 0; i < size; ++i) {
      const double residual = double(values[i]) - encode(values[i], factor) / double(factor);
      result += residual * residual;
    }
    return result;
  }

  double error(double scale) const {
    double result = 0;
    for (double x : magnitudes_) {
      const double code = std::min(double(kLimit), std::floor(x / scale + 0.5));
      const double residual = x - scale * code;
      result += residual * residual;
    }
    return result;
  }

  std::vector<double> magnitudes_;
  std::array<Boundary, kLimit> heap_;
};

// Use all sixteen SQ4 codes, choosing which tail receives the extra level.
// Each orientation is solved by sweeping every relevant rounding interval.
class FullRangeFitter {
  static constexpr std::array<double, 8> kBoundaryInverse = {2.,      2. / 3.,  2. / 5.,  2. / 7.,
                                                             2. / 9., 2. / 11., 2. / 13., 2. / 15.};

 public:
  static int encode(float value, float factor) {
    return int(std::clamp(std::floor(double(value) * factor + 0.5), -8., 7.));
  }

  float fit(const float *values, int size) {
    const double maximum = finite_maximum(values, size);
    if (maximum == 0) return 1.f;
    float best_factor = bounded_factor(7. / maximum);
    double best_loss = stored_loss(values, size, best_factor);
    if (best_loss == 0) return best_factor;

    refit_initial_factor(values, size, best_factor, best_loss);
    if (best_loss == 0) return preserve_exact_grid(values, size, best_factor);

    const double energy = prepare_values(values, size, maximum);
    for (int sign : {1, -1})
      sweep_orientation(values, size, maximum, energy, sign, best_factor, best_loss);
    return best_loss == 0 ? preserve_exact_grid(values, size, best_factor) : best_factor;
  }

 private:
  struct Value {
    double magnitude, saturation;
    int limit;
  };
  struct Stream {
    double scale;
    size_t index;
    int level;
  };
  static float bounded_factor(double factor) {
    return float(std::clamp(factor, double(ColumnFitter<4>::kMinFactor),
                            double(ColumnFitter<4>::kMaxFactor)));
  }

  static void refit_initial_factor(const float *values, int size, float &best_factor,
                                   double &best_loss) {
    // A cheap feasible bound suffices: the two full-range sweeps below already
    // solve the complete problem, including every symmetric candidate.
    float initial = best_factor;
    for (int iteration = 0; iteration < 3; ++iteration) {
      double value_code_dot = 0, code_norm = 0;
      for (int i = 0; i < size; ++i) {
        const int code = encode(values[i], initial);
        value_code_dot += double(values[i]) * code;
        code_norm += code * code;
      }
      if (value_code_dot == 0) break;
      const float next = bounded_factor(code_norm / value_code_dot);
      if (next == initial) break;
      initial = next;
      const double error = stored_loss(values, size, initial);
      if (error < best_loss) {
        best_loss = error;
        best_factor = initial;
      }
    }
  }

  static float preserve_exact_grid(const float *values, int size, float factor) {
    // Exact columns can admit several grids. Keep the symmetric
    // grid when it is also exact, since B refinement depends on its spacing.
    ColumnFitter<4> symmetric;
    const float original = symmetric.fit(values, size);
    return stored_loss(values, size, original) == 0 ? original : factor;
  }

  static void consider_factor(double ideal, const float *values, int size, float &best_factor,
                              double &best_loss) {
    const double bound = ColumnFitter<4>::kMaxFactor;
    const float rounded = float(std::clamp(ideal, -bound, bound));
    for (float factor : {rounded, std::nextafter(rounded, 0.f),
                         std::nextafter(rounded, std::copysign(INFINITY, rounded))}) {
      if (!std::isfinite(factor) || std::abs(factor) < ColumnFitter<4>::kMinFactor ||
          std::abs(factor) > ColumnFitter<4>::kMaxFactor)
        continue;
      const double candidate = stored_loss(values, size, factor);
      if (candidate < best_loss) {
        best_loss = candidate;
        best_factor = factor;
      }
    }
  }

  double prepare_values(const float *values, int size, double maximum) {
    magnitudes_.clear();
    double energy = 0;
    for (int i = 0; i < size; ++i) {
      if (values[i] == 0) continue;
      const double magnitude = std::abs(double(values[i])) / maximum;
      magnitudes_.push_back({magnitude, 0, values[i] < 0 ? 8 : 7});
      energy += magnitude * magnitude;
    }
    std::sort(magnitudes_.begin(), magnitudes_.end(),
              [](const Value &a, const Value &b) { return a.magnitude < b.magnitude; });
    return energy;
  }

  void order_by_saturation() {
    saturation_order_.clear();
    // Merge the two sign groups in descending saturation order. Their
    // magnitudes are already sorted, so neither orientation needs a re-sort.
    int index7 = int(magnitudes_.size()), index8 = index7;
    auto advance = [&](int &index, int limit) {
      while (--index >= 0)
        if (magnitudes_[index].limit == limit) return magnitudes_[index].magnitude / limit;
      return -1.;
    };
    double next7 = advance(index7, 7), next8 = advance(index8, 8);
    while (index7 >= 0 || index8 >= 0) {
      if (next7 >= next8) {
        saturation_order_.push_back({magnitudes_[index7].magnitude, next7, 7});
        next7 = advance(index7, 7);
      } else {
        saturation_order_.push_back({magnitudes_[index8].magnitude, next8, 8});
        next8 = advance(index8, 8);
      }
    }
  }

  double scale_lower_bound(double best_error) const {
    // Saturated coordinates bound the admissible decoding scales from below.
    double value_energy = 0, value_code_dot = 0, code_norm = 0;
    for (size_t i = 0; i < saturation_order_.size(); ++i) {
      const auto v = saturation_order_[i];
      value_energy += v.magnitude * v.magnitude;
      value_code_dot += v.magnitude * v.limit;
      code_norm += v.limit * v.limit;
      if (value_energy <= best_error) continue;
      const double root =
          (value_energy - best_error) /
          (value_code_dot + std::sqrt(std::max(0., value_code_dot * value_code_dot -
                                                       code_norm * (value_energy - best_error))));
      if (i + 1 == saturation_order_.size() || root >= saturation_order_[i + 1].saturation) {
        return std::nextafter(root, 0.);
      }
    }
    return 0.;
  }

  size_t initialize_boundaries(double lower, std::array<Stream, 8> &heap) const {
    // Seven boundary streams use every magnitude; the eighth uses only
    // coordinates assigned the extra negative code. Merge them lazily rather
    // than allocating and sorting all coefficient/level pairs.
    size_t heap_size = 0;
    for (int level = 0; level < 8; ++level) {
      const double inverse = kBoundaryInverse[level];
      size_t index = std::upper_bound(magnitudes_.begin(), magnitudes_.end(), lower,
                                      [inverse](double bound, const Value &v) {
                                        return bound < v.magnitude * inverse;
                                      }) -
                     magnitudes_.begin();
      if (level == 7)
        while (index < magnitudes_.size() && magnitudes_[index].limit != 8) ++index;
      if (index < magnitudes_.size())
        heap[heap_size++] = {magnitudes_[index].magnitude * inverse, index, level};
    }
    std::make_heap(heap.begin(), heap.begin() + heap_size,
                   [](const Stream &a, const Stream &b) { return a.scale > b.scale; });
    return heap_size;
  }

  void sweep_orientation(const float *values, int size, double maximum, double energy, int sign,
                         float &best_factor, double &best_loss) {
    if (sign < 0)
      for (auto &v : magnitudes_) v.limit = 15 - v.limit;
    order_by_saturation();
    double best_error = best_loss / maximum / maximum;
    double best_scale = 1. / std::abs(double(best_factor)) / maximum;
    double lower = scale_lower_bound(best_error);
    double value_code_dot = 0, code_norm = 0, zero_energy = 0;
    for (const auto v : saturation_order_) {
      int code = lower > 0
                     ? int(std::clamp(std::ceil(v.magnitude / lower - 0.5), 0., double(v.limit)))
                     : v.limit;
      while (code > 0 && v.magnitude * kBoundaryInverse[code - 1] <= lower) --code;
      while (code < v.limit && v.magnitude * kBoundaryInverse[code] > lower) ++code;
      value_code_dot += v.magnitude * code;
      code_norm += code * code;
      if (code == 0) zero_energy += v.magnitude * v.magnitude;
    }
    std::array<Stream, 8> heap;
    size_t heap_size = initialize_boundaries(lower, heap);
    bool improved = false;
    for (;;) {
      const double upper = heap_size ? heap[0].scale : INFINITY;
      if (code_norm > 0 && value_code_dot >= lower * code_norm &&
          value_code_dot <= upper * code_norm &&
          value_code_dot * value_code_dot / code_norm >= energy - best_error) {
        const double scale = value_code_dot / code_norm;
        double error = 0;
        for (const auto v : saturation_order_) {
          const double code = std::min(double(v.limit), std::floor(v.magnitude / scale + 0.5));
          const double residual = v.magnitude - scale * code;
          error += residual * residual;
        }
        if (error < best_error) {
          best_error = error;
          best_scale = scale;
          improved = true;
        }
      }
      if (heap_size == 0 || zero_energy >= best_error) break;
      Stream next = heap[0];
      const double magnitude = magnitudes_[next.index].magnitude;
      value_code_dot -= magnitude;
      code_norm -= 2 * next.level + 1;
      lower = next.scale;
      if (next.level == 0) zero_energy += magnitude * magnitude;
      ++next.index;
      if (next.level == 7)
        while (next.index < magnitudes_.size() && magnitudes_[next.index].limit != 8) ++next.index;
      if (next.index == magnitudes_.size()) {
        next = heap[--heap_size];
        if (heap_size == 0) continue;
      } else {
        next.scale = magnitudes_[next.index].magnitude * kBoundaryInverse[next.level];
      }
      replace_boundary_root(heap, heap_size, next);
    }
    if (improved)
      consider_factor(sign / (best_scale * maximum), values, size, best_factor, best_loss);
    // The other orientation can improve the stored codes at the same scale.
    consider_factor(sign * std::abs(double(best_factor)), values, size, best_factor, best_loss);
  }

  static double stored_loss(const float *values, int size, float factor) {
    double sum = 0;
    for (int i = 0; i < size; ++i) {
      const double error = double(values[i]) - encode(values[i], factor) / double(factor);
      sum += error * error;
    }
    return sum;
  }
  std::vector<Value> saturation_order_;
  std::vector<Value> magnitudes_;
};

template <int Bits, bool MatrixB>
inline void quantize_matrix(const ColMatrix &matrix, uint8_t *out, float *factors) {
  constexpr int offset = 1 << (Bits - 1);
  constexpr int divisor = 8 / Bits;
  const int rows = matrix.rows() - int(MatrixB);
  std::conditional_t<Bits == 4, FullRangeFitter, ColumnFitter<Bits>> fitter;
  for (int col = 0; col < matrix.cols(); ++col) {
    const float *values = matrix.col(col).data() + int(MatrixB);
    const float factor = fitter.fit(values, rows);
    factors[col] = factor;
    uint8_t *packed = out + col * rows / divisor;
    if constexpr (Bits == 8) {
      for (int row = 0; row < rows; ++row)
        packed[row] = uint8_t(fitter.encode(values[row], factor) + offset);
    } else {
      // The A kernel packs within 32-coordinate blocks; B packs across its
      // complete quantized latent dimension (16, 32, or 64 coordinates).
      const int block = MatrixB ? rows : 32;
      for (int start = 0; start < rows; start += block) {
        for (int row = 0; row < block / 2; ++row) {
          const int low = fitter.encode(values[start + row], factor) + offset;
          const int high = fitter.encode(values[start + block / 2 + row], factor) + offset;
          packed[start / 2 + row] = uint8_t(low | (high << 4));
        }
      }
    }
  }
}

inline Eigen::MatrixXd decode_first_stage(const ColMatrixUInt8 &packed, const Vector &correction) {
  Eigen::MatrixXd decoded(packed.rows() * 2, packed.cols());
  for (int col = 0; col < decoded.cols(); ++col) {
    for (int row = 0; row < decoded.rows(); ++row) {
      const int byte = (row / 32) * 16 + row % 16;
      const int shift = row % 32 < 16 ? 0 : 4;
      const int code = ((packed(byte, col) >> shift) & 15) - 8;
      decoded(row, col) = code / double(correction[col]);
    }
    decoded(0, col) = correction[packed.cols() + col];
  }
  return decoded;
}

// One coordinate-descent sweep, updating the gradient after each changed code.
inline bool refine_codes(const Eigen::MatrixXd &gram, double diagonal_threshold,
                         Eigen::VectorXd &codes, Eigen::VectorXd &gradient) {
  bool changed = false;
  for (int i = 0; i < codes.size(); ++i) {
    if (gram(i, i) <= diagonal_threshold) continue;
    const double code = std::clamp(std::floor(codes[i] - gradient[i] / gram(i, i) + .5), -8., 7.);
    const double delta = code - codes[i];
    if (delta == 0) continue;
    changed = true;
    codes[i] = code;
    gradient += delta * gram.col(i);
  }
  return changed;
}

// Jointly fit B's codes, decoding scale and protected coefficient to the
// original model product, holding the quantized first-stage model fixed.
inline void refit_correction(const ColMatrix &a, const ColMatrix &b, const ColMatrixUInt8 &packed_a,
                             ColMatrixUInt8 &packed_b, const Vector &correction_a,
                             Vector &correction_b) {
  const Eigen::MatrixXd decoded_a = decode_first_stage(packed_a, correction_a);
  const Eigen::MatrixXd gram = decoded_a.transpose() * decoded_a;
  const double principal_norm = gram(0, 0);
  if (!std::isfinite(principal_norm) || principal_norm <= 1e-12 * gram.trace()) return;

  Eigen::MatrixXd original_a = a.cast<double>();
  original_a.row(0) = correction_a.tail(a.cols()).cast<double>();
  const Eigen::MatrixXd target = (decoded_a.transpose() * original_a) * b.cast<double>();

  // Eliminate the protected coefficient to optimize the integer-coded coordinates.
  const int rank = b.rows() - 1;
  const Eigen::VectorXd principal_cross = gram.col(0).tail(rank);
  const Eigen::MatrixXd reduced_gram =
      gram.bottomRightCorner(rank, rank) -
      principal_cross * principal_cross.transpose() / principal_norm;
  const double diagonal_threshold = 1e-12 * reduced_gram.trace();

  Eigen::VectorXd codes(rank), best_codes(rank), rhs(rank), gradient(rank), gram_codes(rank);
  Eigen::VectorXd decoded(rank + 1), gram_decoded(rank + 1);
  for (int col = 0; col < b.cols(); ++col) {
    for (int row = 0; row < rank; ++row) {
      const int shift = row < rank / 2 ? 0 : 4;
      codes[row] = ((packed_b(row % (rank / 2), col) >> shift) & 15) - 8;
    }
    rhs = target.col(col).tail(rank) - principal_cross * (target(0, col) / principal_norm);
    const double initial_scale = 1. / correction_b[col];
    double scale = initial_scale;
    best_codes = codes;
    float best_factor = correction_b[col];
    float best_correction = correction_b[b.cols() + col];

    // Evaluate the stored float parameters, omitting the constant target norm.
    auto reconstruction_loss = [&](const Eigen::VectorXd &candidate_codes, float factor,
                                   float correction) {
      decoded[0] = correction;
      decoded.tail(rank) = candidate_codes / double(factor);
      gram_decoded.noalias() = gram * decoded;
      return decoded.dot(gram_decoded) - 2 * decoded.dot(target.col(col));
    };
    double best_loss = reconstruction_loss(codes, best_factor, best_correction);
    auto retain_improvement = [&]() {
      const float factor = float(1. / scale);
      if (!std::isfinite(factor) || std::abs(factor) < ColumnFitter<4>::kMinFactor ||
          std::abs(factor) > ColumnFitter<4>::kMaxFactor)
        return;
      const float correction =
          float((target(0, col) - principal_cross.dot(codes) / double(factor)) / principal_norm);
      if (!std::isfinite(correction)) return;

      const double loss = reconstruction_loss(codes, factor, correction);
      if (loss < best_loss) {
        best_loss = loss;
        best_codes = codes;
        best_factor = factor;
        best_correction = correction;
      }
    };

    retain_improvement();
    gram_codes.noalias() = reduced_gram * codes;
    for (int iteration = 0; iteration < 4; ++iteration) {
      gradient = gram_codes - rhs / scale;
      const bool changed = refine_codes(reduced_gram, diagonal_threshold, codes, gradient);
      if (changed) {
        retain_improvement();
        gram_codes.noalias() = reduced_gram * codes;
      }

      const double denominator = codes.dot(gram_codes);
      if (denominator <= 0) break;
      const double multiplier = std::clamp(codes.dot(rhs) / denominator / initial_scale, .5, 2.);
      const double next_scale = initial_scale * multiplier;
      if (next_scale == scale) {
        if (!changed) break;
        continue;
      }
      scale = next_scale;
      retain_improvement();
    }

    correction_b[col] = best_factor;
    correction_b[b.cols() + col] = best_correction;
    for (int row = 0; row < rank / 2; ++row) {
      const uint8_t low = uint8_t(int(best_codes[row]) + 8);
      const uint8_t high = uint8_t(int(best_codes[row + rank / 2]) + 8);
      packed_b(row, col) = low | (high << 4);
    }
  }
}

}  // namespace joint_quantization

}  // namespace Lorann
