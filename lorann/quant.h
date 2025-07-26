#pragma once

#include "utils.h"

namespace Lorann {

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

#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod+i8mm")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod+i8mm"))), \
                             apply_to = function)
#endif

struct SQQuantizer {
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
};

struct SQ4Quantizer : SQQuantizer {
  static constexpr int compensation_factor = 8;
  static constexpr int div_factor = 2;

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

  void matvec_product_B_16(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                           float *LORANN_RESTRICT result, const size_t rows,
                           const size_t cols) const {
    const __m128i vec_chunk = _mm_loadu_si128((const __m128i *)(x));
    for (size_t j = 0; j < cols; ++j) {
      const __m128i col_chunk = unpack128(A + j * 8);
      const __m128i sum = dpbusd(col_chunk, vec_chunk);
      result[j] = horizontal_add(sum);
    }
  }

  void matvec_product_B_32(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                           float *LORANN_RESTRICT result, const size_t rows,
                           const size_t cols) const {
    const __m256i vec_chunk = _mm256_loadu_si256((const __m256i *)(x));
    for (size_t j = 0; j < cols; ++j) {
      const __m256i col_chunk = unpack256(A + j * 16);
      const __m256i sum = dpbusd(col_chunk, vec_chunk);
      result[j] = horizontal_add(sum);
    }
  }

#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
  void matvec_product_B_64(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                           float *LORANN_RESTRICT result, const size_t rows,
                           const size_t cols) const {
    const __m512i vec_chunk = _mm512_loadu_si512((const __m512i *)(x));
    for (size_t j = 0; j < cols; ++j) {
      const __m512i col_chunk = unpack512(A + j * 32);
      const __m512i sum = dpbusd(col_chunk, vec_chunk);
      result[j] = _mm512_reduce_add_epi32(sum);
    }
  }
#else
  void matvec_product_B_64(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                           float *LORANN_RESTRICT result, const size_t rows,
                           const size_t cols) const {
    for (size_t j = 0; j < cols; ++j) {
      int32_t sum = 0;

      for (int k = 0; k < 32; ++k) {
        sum += ((int32_t)(A[k + j * 32] >> 4)) * ((int32_t)x[k + 32]);
        sum += ((int32_t)(A[k + j * 32] & 0xF)) * ((int32_t)x[k]);
      }

      result[j] = sum;
    }
  }
#endif
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

  inline void matvec_product_B_16(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                                  float *LORANN_RESTRICT result, const size_t rows,
                                  const size_t cols) const {
    const uint8x8_t mask_lo8 = vdup_n_u8(0x0F);

    const int8x8_t x_lo_8 = vld1_s8(x);
    const int8x8_t x_hi_8 = vld1_s8(x + 8);

    int8x16_t x_lo_16 = vcombine_s8(x_lo_8, vdup_n_s8(0));
    int8x16_t x_hi_16 = vcombine_s8(x_hi_8, vdup_n_s8(0));

    for (size_t j = 0; j < cols; ++j) {
      const uint8_t *Aj = A + j * 8;

      uint8x8_t bytes8 = vld1_u8(Aj);
      uint8x8_t lo8 = vand_u8(bytes8, mask_lo8);
      uint8x8_t hi8 = vshr_n_u8(bytes8, 4);

      uint8x16_t lo16 = vcombine_u8(lo8, vdup_n_u8(0));
      uint8x16_t hi16 = vcombine_u8(hi8, vdup_n_u8(0));

      int32x4_t acc = vdupq_n_s32(0);
      acc = vusdotq_s32(acc, lo16, x_lo_16);
      acc = vusdotq_s32(acc, hi16, x_hi_16);

      int32_t sum = vaddvq_s32(acc);
      result[j] = (float)sum;
    }
  }

  inline void matvec_product_B_32(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                                  float *LORANN_RESTRICT result, const size_t rows,
                                  const size_t cols) const {
    const int8x16_t x_lo = vld1q_s8(x);
    const int8x16_t x_hi = vld1q_s8(x + 16);

    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);

    for (size_t j = 0; j < cols; ++j) {
      const uint8_t *Aj = A + j * 16;

      uint8x16_t packed = vld1q_u8(Aj);

      uint8x16_t lo_u8 = vandq_u8(packed, mask_lo);
      uint8x16_t hi_u8 = vshrq_n_u8(packed, 4);

      int32x4_t acc = vdupq_n_s32(0);
      acc = vusdotq_s32(acc, lo_u8, x_lo);
      acc = vusdotq_s32(acc, hi_u8, x_hi);

      int32_t sum = vaddvq_s32(acc);
      result[j] = (float)sum;
    }
  }

  inline void matvec_product_B_64(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                                  float *LORANN_RESTRICT result, const size_t rows,
                                  const size_t cols) const {
    const int8x16_t x_lo0 = vld1q_s8(x + 0);
    const int8x16_t x_lo1 = vld1q_s8(x + 16);
    const int8x16_t x_hi0 = vld1q_s8(x + 32);
    const int8x16_t x_hi1 = vld1q_s8(x + 48);

    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);

    for (size_t j = 0; j < cols; ++j) {
      const uint8_t *Aj = A + j * 32;

      uint8x16_t p0 = vld1q_u8(Aj);
      uint8x16_t lo0u = vandq_u8(p0, mask_lo);
      uint8x16_t hi0u = vshrq_n_u8(p0, 4);

      uint8x16_t p1 = vld1q_u8(Aj + 16);
      uint8x16_t lo1u = vandq_u8(p1, mask_lo);
      uint8x16_t hi1u = vshrq_n_u8(p1, 4);

      int32x4_t acc = vdupq_n_s32(0);
      acc = vusdotq_s32(acc, lo0u, x_lo0);
      acc = vusdotq_s32(acc, hi0u, x_hi0);
      acc = vusdotq_s32(acc, lo1u, x_lo1);
      acc = vusdotq_s32(acc, hi1u, x_hi1);

      int32_t sum = vaddvq_s32(acc);
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

  inline void matvec_product_B_16(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                                  float *LORANN_RESTRICT result, size_t rows, size_t cols) const {
    const uint8x8_t mask_lo8 = vdup_n_u8(0x0F);

    const int8x8_t x_lo_8 = vld1_s8(x);
    const int8x8_t x_hi_8 = vld1_s8(x + 8);

    for (size_t j = 0; j < cols; ++j) {
      const uint8_t *Aj = A + j * 8;

      uint8x8_t bytes8 = vld1_u8(Aj);
      uint8x8_t lo8 = vand_u8(bytes8, mask_lo8);
      uint8x8_t hi8 = vshr_n_u8(bytes8, 4);

      int8x8_t lo_s8 = vreinterpret_s8_u8(lo8);
      int8x8_t hi_s8 = vreinterpret_s8_u8(hi8);

      int16x8_t prod0 = vmull_s8(x_lo_8, lo_s8);
      int16x8_t prod1 = vmull_s8(x_hi_8, hi_s8);
      int32x4_t acc = vdupq_n_s32(0);

      acc = vpadalq_s16(acc, prod0);
      acc = vpadalq_s16(acc, prod1);

      int32_t sum = vaddvq_s32(acc);
      result[j] = (float)sum;
    }
  }

  inline void matvec_product_B_32(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                                  float *LORANN_RESTRICT result, size_t rows, size_t cols) const {
    const int8x16_t x_lo = vld1q_s8(x);
    const int8x16_t x_hi = vld1q_s8(x + 16);

    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);

    for (size_t j = 0; j < cols; ++j) {
      const uint8_t *Aj = A + j * 16;

      uint8x16_t packed = vld1q_u8(Aj);

      uint8x16_t lo_u8 = vandq_u8(packed, mask_lo);
      uint8x16_t hi_u8 = vshrq_n_u8(packed, 4);

      int8x16_t lo_s8 = vreinterpretq_s8_u8(lo_u8);
      int8x16_t hi_s8 = vreinterpretq_s8_u8(hi_u8);

      int16x8_t p0 = vmull_s8(vget_low_s8(lo_s8), vget_low_s8(x_lo));
      int16x8_t p1 = vmull_s8(vget_high_s8(lo_s8), vget_high_s8(x_lo));
      int16x8_t p2 = vmull_s8(vget_low_s8(hi_s8), vget_low_s8(x_hi));
      int16x8_t p3 = vmull_s8(vget_high_s8(hi_s8), vget_high_s8(x_hi));

      int32x4_t acc = vdupq_n_s32(0);
      acc = vpadalq_s16(acc, p0);
      acc = vpadalq_s16(acc, p1);
      acc = vpadalq_s16(acc, p2);
      acc = vpadalq_s16(acc, p3);

      int32_t sum = vaddvq_s32(acc);
      result[j] = (float)sum;
    }
  }

  inline void matvec_product_B_64(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                                  float *LORANN_RESTRICT result, size_t rows, size_t cols) const {
    const int8x16_t x_lo0 = vld1q_s8(x + 0);
    const int8x16_t x_lo1 = vld1q_s8(x + 16);
    const int8x16_t x_hi0 = vld1q_s8(x + 32);
    const int8x16_t x_hi1 = vld1q_s8(x + 48);

    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);

    for (size_t j = 0; j < cols; ++j) {
      const uint8_t *Aj = A + j * 32;

      uint8x16_t p0 = vld1q_u8(Aj);
      uint8x16_t lo0u = vandq_u8(p0, mask_lo);
      uint8x16_t hi0u = vshrq_n_u8(p0, 4);

      uint8x16_t p1 = vld1q_u8(Aj + 16);
      uint8x16_t lo1u = vandq_u8(p1, mask_lo);
      uint8x16_t hi1u = vshrq_n_u8(p1, 4);

      int8x16_t lo0s = vreinterpretq_s8_u8(lo0u);
      int8x16_t hi0s = vreinterpretq_s8_u8(hi0u);
      int8x16_t lo1s = vreinterpretq_s8_u8(lo1u);
      int8x16_t hi1s = vreinterpretq_s8_u8(hi1u);

      int16x8_t p0a = vmull_s8(vget_low_s8(lo0s), vget_low_s8(x_lo0));
      int16x8_t p0b = vmull_s8(vget_high_s8(lo0s), vget_high_s8(x_lo0));
      int16x8_t p0c = vmull_s8(vget_low_s8(hi0s), vget_low_s8(x_hi0));
      int16x8_t p0d = vmull_s8(vget_high_s8(hi0s), vget_high_s8(x_hi0));

      int16x8_t p1a = vmull_s8(vget_low_s8(lo1s), vget_low_s8(x_lo1));
      int16x8_t p1b = vmull_s8(vget_high_s8(lo1s), vget_high_s8(x_lo1));
      int16x8_t p1c = vmull_s8(vget_low_s8(hi1s), vget_low_s8(x_hi1));
      int16x8_t p1d = vmull_s8(vget_high_s8(hi1s), vget_high_s8(x_hi1));

      int32x4_t acc = vdupq_n_s32(0);
      acc = vpadalq_s16(acc, p0a);
      acc = vpadalq_s16(acc, p0b);
      acc = vpadalq_s16(acc, p0c);
      acc = vpadalq_s16(acc, p0d);
      acc = vpadalq_s16(acc, p1a);
      acc = vpadalq_s16(acc, p1b);
      acc = vpadalq_s16(acc, p1c);
      acc = vpadalq_s16(acc, p1d);

      int32_t sum = vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1) + vgetq_lane_s32(acc, 2) +
                    vgetq_lane_s32(acc, 3);
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

  inline void matvec_product_B_16(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                                  float *LORANN_RESTRICT result, const size_t rows,
                                  const size_t cols) const {
    for (size_t j = 0; j < cols; ++j) {
      int32_t sum = 0;

      for (int k = 0; k < 8; ++k) {
        sum += ((int32_t)(A[k + j * 8] >> 4)) * ((int32_t)x[k + 8]);
        sum += ((int32_t)(A[k + j * 8] & 0xF)) * ((int32_t)x[k]);
      }

      result[j] = sum;
    }
  }

  inline void matvec_product_B_32(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                                  float *LORANN_RESTRICT result, const size_t rows,
                                  const size_t cols) const {
    for (size_t j = 0; j < cols; ++j) {
      int32_t sum = 0;

      for (int k = 0; k < 16; ++k) {
        sum += ((int32_t)(A[k + j * 16] >> 4)) * ((int32_t)x[k + 16]);
        sum += ((int32_t)(A[k + j * 16] & 0xF)) * ((int32_t)x[k]);
      }

      result[j] = sum;
    }
  }

  inline void matvec_product_B_64(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                                  float *LORANN_RESTRICT result, const size_t rows,
                                  const size_t cols) const {
    for (size_t j = 0; j < cols; ++j) {
      int32_t sum = 0;

      for (int k = 0; k < 32; ++k) {
        sum += ((int32_t)(A[k + j * 32] >> 4)) * ((int32_t)x[k + 32]);
        sum += ((int32_t)(A[k + j * 32] & 0xF)) * ((int32_t)x[k]);
      }

      result[j] = sum;
    }
  }
#endif

  inline void quantized_matvec_product_B(const ColMatrixUInt8 &qA, const VectorInt8 &v,
                                         const Vector &correction, const float scale,
                                         const float factor, const float compensation,
                                         float *result) const {
    const float *scales = correction.data();
    const float *fix = correction.data() + qA.cols();

    const int rank = qA.rows() * 2;
    if (rank == 32)
      matvec_product_B_32(qA.data(), v.data(), result, rank, qA.cols());
    else if (rank == 16)
      matvec_product_B_16(qA.data(), v.data(), result, rank, qA.cols());
    else
      matvec_product_B_64(qA.data(), v.data(), result, rank, qA.cols());

    scale_result(result, compensation, scales, fix, scale, factor, qA.cols());
  }

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
    const int n = A.rows() - 1;
    const int qk = n;

    for (int i = 0; i < A.cols(); ++i) {
      const float *v = A.data() + i * (n + 1) + 1;
      const float factor = compute_quantization_factor(v, n, 4);
      for (int k = 0; k < qk / 2; ++k) {
        const uint8_t a = LORANN_MIN(15, factor * v[k] + 8.5f);
        const uint8_t b = LORANN_MIN(15, factor * v[qk / 2 + k] + 8.5f);

        result[i * n / 2 + k] = a;
        result[i * n / 2 + k] |= b << 4;
      }
      factors[i] = factor;
    }
  }

  inline void quantize_matrix_A_unsigned(const ColMatrix &A, uint8_t *LORANN_RESTRICT result,
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

struct SQ8Quantizer : SQQuantizer {
  static constexpr int compensation_factor = 128;
  static constexpr int div_factor = 1;

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

  void matvec_product_B_16(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                           float *LORANN_RESTRICT result, const size_t rows,
                           const size_t cols) const {
    const __m128i vec_chunk = _mm_loadu_si128((const __m128i *)(x));
    for (size_t j = 0; j < cols; ++j) {
      const __m128i col_chunk = _mm_loadu_si128((const __m128i *)(A + j * 16));
      const __m128i sum = dpbusd(col_chunk, vec_chunk);
      result[j] = horizontal_add(sum);
    }
  }

  void matvec_product_B_32(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                           float *LORANN_RESTRICT result, const size_t rows,
                           const size_t cols) const {
    const __m256i vec_chunk = _mm256_loadu_si256((const __m256i *)(x));
    for (size_t j = 0; j < cols; ++j) {
      const __m256i col_chunk = _mm256_loadu_si256((const __m256i *)(A + j * 32));
      const __m256i sum = dpbusd(col_chunk, vec_chunk);
      result[j] = horizontal_add(sum);
    }
  }

#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
  void matvec_product_B_64(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                           float *LORANN_RESTRICT result, const size_t rows,
                           const size_t cols) const {
    const __m512i vec_chunk = _mm512_loadu_si512((const __m512i *)(x));
    for (size_t j = 0; j < cols; ++j) {
      const __m512i col_chunk = _mm512_loadu_si512((const __m512i *)(A + j * 64));
      const __m512i sum = dpbusd(col_chunk, vec_chunk);
      result[j] = _mm512_reduce_add_epi32(sum);
    }
  }
#else
  void matvec_product_B_64(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                           float *LORANN_RESTRICT result, const size_t rows,
                           const size_t cols) const {
    matvec_product_A(A, x, result, rows, cols);
  }
#endif
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

  inline void matvec_product_B_16(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                                  float *LORANN_RESTRICT result, const size_t rows,
                                  const size_t cols) const {
    const int8x16_t vx = vld1q_s8(x);

    size_t j = 0;
    for (; j + 4 <= cols; j += 4) {
      const uint8_t *a0 = A + (j + 0) * 16;
      const uint8_t *a1 = A + (j + 1) * 16;
      const uint8_t *a2 = A + (j + 2) * 16;
      const uint8_t *a3 = A + (j + 3) * 16;

      uint8x16_t va0 = vld1q_u8(a0);
      uint8x16_t va1 = vld1q_u8(a1);
      uint8x16_t va2 = vld1q_u8(a2);
      uint8x16_t va3 = vld1q_u8(a3);

      int32x4_t acc0 = vdupq_n_s32(0);
      int32x4_t acc1 = vdupq_n_s32(0);
      int32x4_t acc2 = vdupq_n_s32(0);
      int32x4_t acc3 = vdupq_n_s32(0);

      acc0 = vusdotq_s32(acc0, va0, vx);
      acc1 = vusdotq_s32(acc1, va1, vx);
      acc2 = vusdotq_s32(acc2, va2, vx);
      acc3 = vusdotq_s32(acc3, va3, vx);

      result[j + 0] = (float)vaddvq_s32(acc0);
      result[j + 1] = (float)vaddvq_s32(acc1);
      result[j + 2] = (float)vaddvq_s32(acc2);
      result[j + 3] = (float)vaddvq_s32(acc3);
    }

    for (; j < cols; ++j) {
      const uint8_t *a = A + j * 16;
      uint8x16_t va = vld1q_u8(a);
      int32x4_t acc = vdupq_n_s32(0);
      acc = vusdotq_s32(acc, va, vx);
      result[j] = (float)vaddvq_s32(acc);
    }
  }

  inline void matvec_product_B_32(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                                  float *LORANN_RESTRICT result, const size_t rows,
                                  const size_t cols) const {
    const int8x16_t vx0 = vld1q_s8(x);
    const int8x16_t vx1 = vld1q_s8(x + 16);

    size_t j = 0;
    for (; j + 4 <= cols; j += 4) {
      const uint8_t *a0 = A + (j + 0) * 32;
      const uint8_t *a1 = A + (j + 1) * 32;
      const uint8_t *a2 = A + (j + 2) * 32;
      const uint8_t *a3 = A + (j + 3) * 32;

      uint8x16_t a0_0 = vld1q_u8(a0);
      uint8x16_t a0_1 = vld1q_u8(a0 + 16);
      uint8x16_t a1_0 = vld1q_u8(a1);
      uint8x16_t a1_1 = vld1q_u8(a1 + 16);
      uint8x16_t a2_0 = vld1q_u8(a2);
      uint8x16_t a2_1 = vld1q_u8(a2 + 16);
      uint8x16_t a3_0 = vld1q_u8(a3);
      uint8x16_t a3_1 = vld1q_u8(a3 + 16);

      int32x4_t acc0 = vdupq_n_s32(0);
      int32x4_t acc1 = vdupq_n_s32(0);
      int32x4_t acc2 = vdupq_n_s32(0);
      int32x4_t acc3 = vdupq_n_s32(0);

      acc0 = vusdotq_s32(acc0, a0_0, vx0);
      acc0 = vusdotq_s32(acc0, a0_1, vx1);

      acc1 = vusdotq_s32(acc1, a1_0, vx0);
      acc1 = vusdotq_s32(acc1, a1_1, vx1);

      acc2 = vusdotq_s32(acc2, a2_0, vx0);
      acc2 = vusdotq_s32(acc2, a2_1, vx1);

      acc3 = vusdotq_s32(acc3, a3_0, vx0);
      acc3 = vusdotq_s32(acc3, a3_1, vx1);

      result[j + 0] = (float)vaddvq_s32(acc0);
      result[j + 1] = (float)vaddvq_s32(acc1);
      result[j + 2] = (float)vaddvq_s32(acc2);
      result[j + 3] = (float)vaddvq_s32(acc3);
    }

    for (; j < cols; ++j) {
      const uint8_t *a = A + j * 32;
      uint8x16_t a0 = vld1q_u8(a);
      uint8x16_t a1 = vld1q_u8(a + 16);
      int32x4_t acc = vdupq_n_s32(0);
      acc = vusdotq_s32(acc, a0, vx0);
      acc = vusdotq_s32(acc, a1, vx1);
      result[j] = (float)vaddvq_s32(acc);
    }
  }

  inline void matvec_product_B_64(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                                  float *LORANN_RESTRICT result, const size_t rows,
                                  const size_t cols) const {
    const int8x16_t vx0 = vld1q_s8(x + 0);
    const int8x16_t vx1 = vld1q_s8(x + 16);
    const int8x16_t vx2 = vld1q_s8(x + 32);
    const int8x16_t vx3 = vld1q_s8(x + 48);

    size_t j = 0;
    for (; j + 4 <= cols; j += 4) {
      const uint8_t *a0 = A + (j + 0) * 64;
      const uint8_t *a1 = A + (j + 1) * 64;
      const uint8_t *a2 = A + (j + 2) * 64;
      const uint8_t *a3 = A + (j + 3) * 64;

      uint8x16_t a0_0 = vld1q_u8(a0 + 0);
      uint8x16_t a0_1 = vld1q_u8(a0 + 16);
      uint8x16_t a0_2 = vld1q_u8(a0 + 32);
      uint8x16_t a0_3 = vld1q_u8(a0 + 48);

      uint8x16_t a1_0 = vld1q_u8(a1 + 0);
      uint8x16_t a1_1 = vld1q_u8(a1 + 16);
      uint8x16_t a1_2 = vld1q_u8(a1 + 32);
      uint8x16_t a1_3 = vld1q_u8(a1 + 48);

      uint8x16_t a2_0 = vld1q_u8(a2 + 0);
      uint8x16_t a2_1 = vld1q_u8(a2 + 16);
      uint8x16_t a2_2 = vld1q_u8(a2 + 32);
      uint8x16_t a2_3 = vld1q_u8(a2 + 48);

      uint8x16_t a3_0 = vld1q_u8(a3 + 0);
      uint8x16_t a3_1 = vld1q_u8(a3 + 16);
      uint8x16_t a3_2 = vld1q_u8(a3 + 32);
      uint8x16_t a3_3 = vld1q_u8(a3 + 48);

      int32x4_t acc0 = vdupq_n_s32(0);
      int32x4_t acc1 = vdupq_n_s32(0);
      int32x4_t acc2 = vdupq_n_s32(0);
      int32x4_t acc3 = vdupq_n_s32(0);

      acc0 = vusdotq_s32(acc0, a0_0, vx0);
      acc0 = vusdotq_s32(acc0, a0_1, vx1);
      acc0 = vusdotq_s32(acc0, a0_2, vx2);
      acc0 = vusdotq_s32(acc0, a0_3, vx3);

      acc1 = vusdotq_s32(acc1, a1_0, vx0);
      acc1 = vusdotq_s32(acc1, a1_1, vx1);
      acc1 = vusdotq_s32(acc1, a1_2, vx2);
      acc1 = vusdotq_s32(acc1, a1_3, vx3);

      acc2 = vusdotq_s32(acc2, a2_0, vx0);
      acc2 = vusdotq_s32(acc2, a2_1, vx1);
      acc2 = vusdotq_s32(acc2, a2_2, vx2);
      acc2 = vusdotq_s32(acc2, a2_3, vx3);

      acc3 = vusdotq_s32(acc3, a3_0, vx0);
      acc3 = vusdotq_s32(acc3, a3_1, vx1);
      acc3 = vusdotq_s32(acc3, a3_2, vx2);
      acc3 = vusdotq_s32(acc3, a3_3, vx3);

      result[j + 0] = (float)vaddvq_s32(acc0);
      result[j + 1] = (float)vaddvq_s32(acc1);
      result[j + 2] = (float)vaddvq_s32(acc2);
      result[j + 3] = (float)vaddvq_s32(acc3);
    }

    for (; j < cols; ++j) {
      const uint8_t *a = A + j * 64;
      uint8x16_t a_0 = vld1q_u8(a + 0);
      uint8x16_t a_1 = vld1q_u8(a + 16);
      uint8x16_t a_2 = vld1q_u8(a + 32);
      uint8x16_t a_3 = vld1q_u8(a + 48);

      int32x4_t acc = vdupq_n_s32(0);
      acc = vusdotq_s32(acc, a_0, vx0);
      acc = vusdotq_s32(acc, a_1, vx1);
      acc = vusdotq_s32(acc, a_2, vx2);
      acc = vusdotq_s32(acc, a_3, vx3);

      result[j] = (float)vaddvq_s32(acc);
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

  inline void matvec_product_B_16(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                                  float *LORANN_RESTRICT result, const size_t rows,
                                  const size_t cols) const {
    const int8x16_t vx8 = vld1q_s8(x);
    const int16x8_t vx0 = vmovl_s8(vget_low_s8(vx8));
    const int16x8_t vx1 = vmovl_s8(vget_high_s8(vx8));

    size_t j = 0;
    for (; j + 2 <= cols; j += 2) {
      const uint8_t *a0 = A + (j + 0) * 16;
      const uint8_t *a1 = A + (j + 1) * 16;

      uint8x16_t va0u8 = vld1q_u8(a0);
      uint8x16_t va1u8 = vld1q_u8(a1);

      int16x8_t va0_0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(va0u8)));
      int16x8_t va0_1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(va0u8)));
      int16x8_t va1_0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(va1u8)));
      int16x8_t va1_1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(va1u8)));

      int32x4_t acc0 = vdupq_n_s32(0);
      int32x4_t acc1 = vdupq_n_s32(0);

      acc0 = vmlal_s16(acc0, vget_low_s16(va0_0), vget_low_s16(vx0));
      acc0 = vmlal_s16(acc0, vget_high_s16(va0_0), vget_high_s16(vx0));
      acc0 = vmlal_s16(acc0, vget_low_s16(va0_1), vget_low_s16(vx1));
      acc0 = vmlal_s16(acc0, vget_high_s16(va0_1), vget_high_s16(vx1));

      acc1 = vmlal_s16(acc1, vget_low_s16(va1_0), vget_low_s16(vx0));
      acc1 = vmlal_s16(acc1, vget_high_s16(va1_0), vget_high_s16(vx0));
      acc1 = vmlal_s16(acc1, vget_low_s16(va1_1), vget_low_s16(vx1));
      acc1 = vmlal_s16(acc1, vget_high_s16(va1_1), vget_high_s16(vx1));

      result[j + 0] = (float)vaddvq_s32(acc0);
      result[j + 1] = (float)vaddvq_s32(acc1);
    }

    for (; j < cols; ++j) {
      uint8x16_t va = vld1q_u8(A + j * 16);
      int16x8_t va0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(va)));
      int16x8_t va1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(va)));

      int32x4_t acc = vdupq_n_s32(0);
      acc = vmlal_s16(acc, vget_low_s16(va0), vget_low_s16(vx0));
      acc = vmlal_s16(acc, vget_high_s16(va0), vget_high_s16(vx0));
      acc = vmlal_s16(acc, vget_low_s16(va1), vget_low_s16(vx1));
      acc = vmlal_s16(acc, vget_high_s16(va1), vget_high_s16(vx1));

      result[j] = (float)vaddvq_s32(acc);
    }
  }

  inline void matvec_product_B_32(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                                  float *LORANN_RESTRICT result, const size_t rows,
                                  const size_t cols) const {
    const int8x16_t vx8_0 = vld1q_s8(x);
    const int8x16_t vx8_1 = vld1q_s8(x + 16);
    const int16x8_t vx0_l = vmovl_s8(vget_low_s8(vx8_0));
    const int16x8_t vx0_h = vmovl_s8(vget_high_s8(vx8_0));
    const int16x8_t vx1_l = vmovl_s8(vget_low_s8(vx8_1));
    const int16x8_t vx1_h = vmovl_s8(vget_high_s8(vx8_1));

    size_t j = 0;
    for (; j + 2 <= cols; j += 2) {
      const uint8_t *a0 = A + (j + 0) * 32;
      const uint8_t *a1 = A + (j + 1) * 32;

      uint8x16_t a0_0u8 = vld1q_u8(a0);
      uint8x16_t a0_1u8 = vld1q_u8(a0 + 16);
      uint8x16_t a1_0u8 = vld1q_u8(a1);
      uint8x16_t a1_1u8 = vld1q_u8(a1 + 16);

      int16x8_t a0_0l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a0_0u8)));
      int16x8_t a0_0h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a0_0u8)));
      int16x8_t a0_1l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a0_1u8)));
      int16x8_t a0_1h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a0_1u8)));

      int16x8_t a1_0l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a1_0u8)));
      int16x8_t a1_0h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a1_0u8)));
      int16x8_t a1_1l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a1_1u8)));
      int16x8_t a1_1h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a1_1u8)));

      int32x4_t acc0 = vdupq_n_s32(0);
      int32x4_t acc1 = vdupq_n_s32(0);

      acc0 = vmlal_s16(acc0, vget_low_s16(a0_0l), vget_low_s16(vx0_l));
      acc0 = vmlal_s16(acc0, vget_high_s16(a0_0l), vget_high_s16(vx0_l));
      acc0 = vmlal_s16(acc0, vget_low_s16(a0_0h), vget_low_s16(vx0_h));
      acc0 = vmlal_s16(acc0, vget_high_s16(a0_0h), vget_high_s16(vx0_h));

      acc0 = vmlal_s16(acc0, vget_low_s16(a0_1l), vget_low_s16(vx1_l));
      acc0 = vmlal_s16(acc0, vget_high_s16(a0_1l), vget_high_s16(vx1_l));
      acc0 = vmlal_s16(acc0, vget_low_s16(a0_1h), vget_low_s16(vx1_h));
      acc0 = vmlal_s16(acc0, vget_high_s16(a0_1h), vget_high_s16(vx1_h));

      acc1 = vmlal_s16(acc1, vget_low_s16(a1_0l), vget_low_s16(vx0_l));
      acc1 = vmlal_s16(acc1, vget_high_s16(a1_0l), vget_high_s16(vx0_l));
      acc1 = vmlal_s16(acc1, vget_low_s16(a1_0h), vget_low_s16(vx0_h));
      acc1 = vmlal_s16(acc1, vget_high_s16(a1_0h), vget_high_s16(vx0_h));

      acc1 = vmlal_s16(acc1, vget_low_s16(a1_1l), vget_low_s16(vx1_l));
      acc1 = vmlal_s16(acc1, vget_high_s16(a1_1l), vget_high_s16(vx1_l));
      acc1 = vmlal_s16(acc1, vget_low_s16(a1_1h), vget_low_s16(vx1_h));
      acc1 = vmlal_s16(acc1, vget_high_s16(a1_1h), vget_high_s16(vx1_h));

      result[j + 0] = (float)vaddvq_s32(acc0);
      result[j + 1] = (float)vaddvq_s32(acc1);
    }

    for (; j < cols; ++j) {
      const uint8_t *a = A + j * 32;
      uint8x16_t a0u8 = vld1q_u8(a);
      uint8x16_t a1u8 = vld1q_u8(a + 16);

      int16x8_t a0l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a0u8)));
      int16x8_t a0h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a0u8)));
      int16x8_t a1l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a1u8)));
      int16x8_t a1h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a1u8)));

      int32x4_t acc = vdupq_n_s32(0);
      acc = vmlal_s16(acc, vget_low_s16(a0l), vget_low_s16(vx0_l));
      acc = vmlal_s16(acc, vget_high_s16(a0l), vget_high_s16(vx0_l));
      acc = vmlal_s16(acc, vget_low_s16(a0h), vget_low_s16(vx0_h));
      acc = vmlal_s16(acc, vget_high_s16(a0h), vget_high_s16(vx0_h));

      acc = vmlal_s16(acc, vget_low_s16(a1l), vget_low_s16(vx1_l));
      acc = vmlal_s16(acc, vget_high_s16(a1l), vget_high_s16(vx1_l));
      acc = vmlal_s16(acc, vget_low_s16(a1h), vget_low_s16(vx1_h));
      acc = vmlal_s16(acc, vget_high_s16(a1h), vget_high_s16(vx1_h));

      result[j] = (float)vaddvq_s32(acc);
    }
  }

  inline void matvec_product_B_64(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                                  float *LORANN_RESTRICT result, const size_t rows,
                                  const size_t cols) const {
    const int8x16_t vx8_0 = vld1q_s8(x + 0);
    const int8x16_t vx8_1 = vld1q_s8(x + 16);
    const int8x16_t vx8_2 = vld1q_s8(x + 32);
    const int8x16_t vx8_3 = vld1q_s8(x + 48);

    const int16x8_t vx0_l = vmovl_s8(vget_low_s8(vx8_0));
    const int16x8_t vx0_h = vmovl_s8(vget_high_s8(vx8_0));
    const int16x8_t vx1_l = vmovl_s8(vget_low_s8(vx8_1));
    const int16x8_t vx1_h = vmovl_s8(vget_high_s8(vx8_1));
    const int16x8_t vx2_l = vmovl_s8(vget_low_s8(vx8_2));
    const int16x8_t vx2_h = vmovl_s8(vget_high_s8(vx8_2));
    const int16x8_t vx3_l = vmovl_s8(vget_low_s8(vx8_3));
    const int16x8_t vx3_h = vmovl_s8(vget_high_s8(vx8_3));

    size_t j = 0;
    for (; j + 2 <= cols; j += 2) {
      const uint8_t *a0 = A + (j + 0) * 64;
      const uint8_t *a1 = A + (j + 1) * 64;

      uint8x16_t a0_0u8 = vld1q_u8(a0 + 0);
      uint8x16_t a0_1u8 = vld1q_u8(a0 + 16);
      uint8x16_t a0_2u8 = vld1q_u8(a0 + 32);
      uint8x16_t a0_3u8 = vld1q_u8(a0 + 48);

      uint8x16_t a1_0u8 = vld1q_u8(a1 + 0);
      uint8x16_t a1_1u8 = vld1q_u8(a1 + 16);
      uint8x16_t a1_2u8 = vld1q_u8(a1 + 32);
      uint8x16_t a1_3u8 = vld1q_u8(a1 + 48);

      auto W = [](uint8x16_t v) {
        int16x8_t lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v)));
        int16x8_t hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v)));
        return std::pair<int16x8_t, int16x8_t>(lo, hi);
      };

      auto [a0_0l, a0_0h] = W(a0_0u8);
      auto [a0_1l, a0_1h] = W(a0_1u8);
      auto [a0_2l, a0_2h] = W(a0_2u8);
      auto [a0_3l, a0_3h] = W(a0_3u8);

      auto [a1_0l, a1_0h] = W(a1_0u8);
      auto [a1_1l, a1_1h] = W(a1_1u8);
      auto [a1_2l, a1_2h] = W(a1_2u8);
      auto [a1_3l, a1_3h] = W(a1_3u8);

      int32x4_t acc0 = vdupq_n_s32(0);
      int32x4_t acc1 = vdupq_n_s32(0);

      acc0 = vmlal_s16(acc0, vget_low_s16(a0_0l), vget_low_s16(vx0_l));
      acc0 = vmlal_s16(acc0, vget_high_s16(a0_0l), vget_high_s16(vx0_l));
      acc0 = vmlal_s16(acc0, vget_low_s16(a0_0h), vget_low_s16(vx0_h));
      acc0 = vmlal_s16(acc0, vget_high_s16(a0_0h), vget_high_s16(vx0_h));

      acc0 = vmlal_s16(acc0, vget_low_s16(a0_1l), vget_low_s16(vx1_l));
      acc0 = vmlal_s16(acc0, vget_high_s16(a0_1l), vget_high_s16(vx1_l));
      acc0 = vmlal_s16(acc0, vget_low_s16(a0_1h), vget_low_s16(vx1_h));
      acc0 = vmlal_s16(acc0, vget_high_s16(a0_1h), vget_high_s16(vx1_h));

      acc0 = vmlal_s16(acc0, vget_low_s16(a0_2l), vget_low_s16(vx2_l));
      acc0 = vmlal_s16(acc0, vget_high_s16(a0_2l), vget_high_s16(vx2_l));
      acc0 = vmlal_s16(acc0, vget_low_s16(a0_2h), vget_low_s16(vx2_h));
      acc0 = vmlal_s16(acc0, vget_high_s16(a0_2h), vget_high_s16(vx2_h));

      acc0 = vmlal_s16(acc0, vget_low_s16(a0_3l), vget_low_s16(vx3_l));
      acc0 = vmlal_s16(acc0, vget_high_s16(a0_3l), vget_high_s16(vx3_l));
      acc0 = vmlal_s16(acc0, vget_low_s16(a0_3h), vget_low_s16(vx3_h));
      acc0 = vmlal_s16(acc0, vget_high_s16(a0_3h), vget_high_s16(vx3_h));

      acc1 = vmlal_s16(acc1, vget_low_s16(a1_0l), vget_low_s16(vx0_l));
      acc1 = vmlal_s16(acc1, vget_high_s16(a1_0l), vget_high_s16(vx0_l));
      acc1 = vmlal_s16(acc1, vget_low_s16(a1_0h), vget_low_s16(vx0_h));
      acc1 = vmlal_s16(acc1, vget_high_s16(a1_0h), vget_high_s16(vx0_h));

      acc1 = vmlal_s16(acc1, vget_low_s16(a1_1l), vget_low_s16(vx1_l));
      acc1 = vmlal_s16(acc1, vget_high_s16(a1_1l), vget_high_s16(vx1_l));
      acc1 = vmlal_s16(acc1, vget_low_s16(a1_1h), vget_low_s16(vx1_h));
      acc1 = vmlal_s16(acc1, vget_high_s16(a1_1h), vget_high_s16(vx1_h));

      acc1 = vmlal_s16(acc1, vget_low_s16(a1_2l), vget_low_s16(vx2_l));
      acc1 = vmlal_s16(acc1, vget_high_s16(a1_2l), vget_high_s16(vx2_l));
      acc1 = vmlal_s16(acc1, vget_low_s16(a1_2h), vget_low_s16(vx2_h));
      acc1 = vmlal_s16(acc1, vget_high_s16(a1_2h), vget_high_s16(vx2_h));

      acc1 = vmlal_s16(acc1, vget_low_s16(a1_3l), vget_low_s16(vx3_l));
      acc1 = vmlal_s16(acc1, vget_high_s16(a1_3l), vget_high_s16(vx3_l));
      acc1 = vmlal_s16(acc1, vget_low_s16(a1_3h), vget_low_s16(vx3_h));
      acc1 = vmlal_s16(acc1, vget_high_s16(a1_3h), vget_high_s16(vx3_h));

      result[j + 0] = (float)vaddvq_s32(acc0);
      result[j + 1] = (float)vaddvq_s32(acc1);
    }

    for (; j < cols; ++j) {
      const uint8_t *a = A + j * 64;

      uint8x16_t a0u8 = vld1q_u8(a + 0);
      uint8x16_t a1u8 = vld1q_u8(a + 16);
      uint8x16_t a2u8 = vld1q_u8(a + 32);
      uint8x16_t a3u8 = vld1q_u8(a + 48);

      int16x8_t a0l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a0u8)));
      int16x8_t a0h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a0u8)));
      int16x8_t a1l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a1u8)));
      int16x8_t a1h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a1u8)));
      int16x8_t a2l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a2u8)));
      int16x8_t a2h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a2u8)));
      int16x8_t a3l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a3u8)));
      int16x8_t a3h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a3u8)));

      int32x4_t acc = vdupq_n_s32(0);
      acc = vmlal_s16(acc, vget_low_s16(a0l), vget_low_s16(vx0_l));
      acc = vmlal_s16(acc, vget_high_s16(a0l), vget_high_s16(vx0_l));
      acc = vmlal_s16(acc, vget_low_s16(a0h), vget_low_s16(vx0_h));
      acc = vmlal_s16(acc, vget_high_s16(a0h), vget_high_s16(vx0_h));

      acc = vmlal_s16(acc, vget_low_s16(a1l), vget_low_s16(vx1_l));
      acc = vmlal_s16(acc, vget_high_s16(a1l), vget_high_s16(vx1_l));
      acc = vmlal_s16(acc, vget_low_s16(a1h), vget_low_s16(vx1_h));
      acc = vmlal_s16(acc, vget_high_s16(a1h), vget_high_s16(vx1_h));

      acc = vmlal_s16(acc, vget_low_s16(a2l), vget_low_s16(vx2_l));
      acc = vmlal_s16(acc, vget_high_s16(a2l), vget_high_s16(vx2_l));
      acc = vmlal_s16(acc, vget_low_s16(a2h), vget_low_s16(vx2_h));
      acc = vmlal_s16(acc, vget_high_s16(a2h), vget_high_s16(vx2_h));

      acc = vmlal_s16(acc, vget_low_s16(a3l), vget_low_s16(vx3_l));
      acc = vmlal_s16(acc, vget_high_s16(a3l), vget_high_s16(vx3_l));
      acc = vmlal_s16(acc, vget_low_s16(a3h), vget_low_s16(vx3_h));
      acc = vmlal_s16(acc, vget_high_s16(a3h), vget_high_s16(vx3_h));

      result[j] = (float)vaddvq_s32(acc);
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

  inline void matvec_product_B_16(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                                  float *LORANN_RESTRICT result, const size_t rows,
                                  const size_t cols) const {
    matvec_product_A(A, x, result, rows, cols);
  }

  inline void matvec_product_B_32(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                                  float *LORANN_RESTRICT result, const size_t rows,
                                  const size_t cols) const {
    matvec_product_A(A, x, result, rows, cols);
  }

  inline void matvec_product_B_64(const uint8_t *LORANN_RESTRICT A, const int8_t *LORANN_RESTRICT x,
                                  float *LORANN_RESTRICT result, const size_t rows,
                                  const size_t cols) const {
    matvec_product_A(A, x, result, rows, cols);
  }
#endif

  inline void quantized_matvec_product_B(const ColMatrixUInt8 &qA, const VectorInt8 &v,
                                         const Vector &correction, const float scale,
                                         const float factor, const float compensation,
                                         float *result) const {
    const float *scales = correction.data();
    const float *fix = correction.data() + qA.cols();

    const int rank = qA.rows();
    if (rank == 32)
      matvec_product_B_32(qA.data(), v.data(), result, rank, qA.cols());
    else if (rank == 16)
      matvec_product_B_16(qA.data(), v.data(), result, rank, qA.cols());
    else
      matvec_product_B_64(qA.data(), v.data(), result, rank, qA.cols());

    scale_result(result, compensation, scales, fix, scale, factor, qA.cols());
  }

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
};

#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
#pragma clang attribute pop
#pragma GCC pop_options
#endif

}  // namespace Lorann