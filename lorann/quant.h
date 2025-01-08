#pragma once

#include "utils.h"

namespace Lorann {

#if defined(__AVX2__)

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
#define MM512_SET_M256I(a, b) _mm512_inserti64x4(_mm512_castsi256_si512(b), (a), 1)

inline __m128i unpack128(const uint8_t *rsi) {
  const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
  const __m128i bytes =
      _mm_set_epi64x(_mm_extract_epi64(_mm_srli_epi16(tmp, 4), 1), _mm_extract_epi64(tmp, 0));
  const __m128i low_mask = _mm_set1_epi8(0xF);
  return _mm_and_si128(low_mask, bytes);
}

inline __m256i unpack256(const uint8_t *rsi) {
  const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
  const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
  const __m256i low_mask = _mm256_set1_epi8(0xF);
  return _mm256_and_si256(low_mask, bytes);
}

#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
inline __m512i unpack512(const uint8_t *rsi) {
  const __m256i tmp = _mm256_loadu_si256((const __m256i *)rsi);
  const __m256i shifted = _mm256_srli_epi16(tmp, 4);
  const __m512i bytes = MM512_SET_M256I(shifted, tmp);
  const __m512i low_mask = _mm512_set1_epi8(0xF);
  return _mm512_and_si512(low_mask, bytes);
}
#endif

inline __m128i dpbusd(const __m128i a, const __m128i b) {
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

inline __m128i dpbusd(__m128i c, const __m128i a, const __m128i b) {
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

inline __m256i dpbusd(const __m256i a, const __m256i b) {
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

inline __m256i dpbusd(__m256i c, const __m256i a, const __m256i b) {
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
inline __m512i dpbusd(const __m512i a, const __m512i b) {
  __m512i sum = _mm512_setzero_si512();
  asm("vpdpbusd %2, %1, %0" : "+x"(sum) : "x"(a), "mx"(b));
  return sum;
}

inline __m512i dpbusd(__m512i c, const __m512i a, const __m512i b) {
  asm("vpdpbusd %2, %1, %0" : "+x"(c) : "x"(a), "mx"(b));
  return c;
}
#endif

#endif

struct SQQuantizer {
#if defined(__AVX2__)
  void scale_result(float *__restrict__ result, const float compensation,
                    const float *__restrict__ scale, const float *__restrict__ fix,
                    const float factor, const float correction, const int n) const {
    const __m256 v_factor = _mm256_set1_ps(factor);
    const __m256 v_correction = _mm256_set1_ps(correction);
    const __m256 v_compensation = _mm256_set1_ps(compensation);

    int i;
    for (i = 0; i + 8 <= n; i += 8) {
      const __m256 v_result = _mm256_loadu_ps(&result[i]);
      const __m256 v_scale = _mm256_loadu_ps(&scale[i]);
      const __m256 v_fix = _mm256_loadu_ps(&fix[i]);

      const __m256 v_sub = _mm256_sub_ps(v_result, v_compensation);
      const __m256 v_mul = _mm256_mul_ps(v_factor, v_scale);
      const __m256 v_mul_fix = _mm256_mul_ps(v_correction, v_fix);
      const __m256 v_div = _mm256_div_ps(v_sub, v_mul);
      const __m256 v_add = _mm256_add_ps(v_div, v_mul_fix);

      _mm256_storeu_ps(&result[i], v_add);
    }

    for (; i < n; ++i) {
      result[i] = (result[i] - compensation) / (factor * scale[i]) + correction * fix[i];
    }
  }
#else
  void scale_result(float *result, const float compensation, const float *scale, const float *fix,
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

  inline void matvec_product_A(const uint8_t *A, const int8_t *x, float *result, const size_t rows,
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

  void matvec_product_B_16(const uint8_t *A, const int8_t *x, float *result, const size_t rows,
                           const size_t cols) const {
    const __m128i vec_chunk = _mm_loadu_si128((const __m128i *)(x));
    for (size_t j = 0; j < cols; ++j) {
      const __m128i col_chunk = unpack128(A + j * 8);
      const __m128i sum = dpbusd(col_chunk, vec_chunk);
      result[j] = horizontal_add(sum);
    }
  }

  void matvec_product_B_32(const uint8_t *A, const int8_t *x, float *result, const size_t rows,
                           const size_t cols) const {
    const __m256i vec_chunk = _mm256_loadu_si256((const __m256i *)(x));
    for (size_t j = 0; j < cols; ++j) {
      const __m256i col_chunk = unpack256(A + j * 16);
      const __m256i sum = dpbusd(col_chunk, vec_chunk);
      result[j] = horizontal_add(sum);
    }
  }

#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
  void matvec_product_B_64(const uint8_t *A, const int8_t *x, float *result, const size_t rows,
                           const size_t cols) const {
    const __m512i vec_chunk = _mm512_loadu_si512((const __m512i *)(x));
    for (size_t j = 0; j < cols; ++j) {
      const __m512i col_chunk = unpack512(A + j * 32);
      const __m512i sum = dpbusd(col_chunk, vec_chunk);
      result[j] = _mm512_reduce_add_epi32(sum);
    }
  }
#else
  void matvec_product_B_64(const uint8_t *A, const int8_t *x, float *result, const size_t rows,
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
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__ARM_FEATURE_DOTPROD)
  inline void matvec_product_A(const uint8_t *A, const int8_t *x, float *result, const size_t rows,
                               const size_t cols) const {
    for (size_t j = 0; j < cols; ++j) {
      int32x4_t sum_vec = vdupq_n_s32(0);
      for (size_t i = 0; i < rows; i += 32) {
        const int8x16_t x_high = vld1q_s8(&x[i + 16]);
        const int8x16_t x_low = vld1q_s8(&x[i]);

        const uint8x16_t a_vec = vld1q_u8(&A[(i + j * rows) / 2]);
        const uint8x16_t a_high_u8 = vshrq_n_u8(a_vec, 4);
        const uint8x16_t a_low_u8 = vandq_u8(a_vec, vdupq_n_u8(0x0F));
        const int8x16_t a_high_s8 = vreinterpretq_s8_u8(a_high_u8);
        const int8x16_t a_low_s8 = vreinterpretq_s8_u8(a_low_u8);

        sum_vec = vdotq_s32(sum_vec, a_high_s8, x_high);
        sum_vec = vdotq_s32(sum_vec, a_low_s8, x_low);
      }
      result[j] = vaddvq_s32(sum_vec);
    }
  }

  inline void matvec_product_B_16(const uint8_t *A, const int8_t *x, float *result,
                                  const size_t rows, const size_t cols) const {
    for (size_t j = 0; j < cols; ++j) {
      int32x4_t sum_vec = vdupq_n_s32(0);

      const int8x8_t x_low = vld1_s8(x);
      const int8x8_t x_high = vld1_s8(x + 8);

      const uint8x8_t a_vec = vld1_u8(A + j * 8);
      const uint8x8_t a_high = vshr_n_u8(a_vec, 4);
      const uint8x8_t a_low = vand_u8(a_vec, vdup_n_u8(0x0F));

      const int16x8_t x_low_16 = vmovl_s8(x_low);
      const int16x8_t x_high_16 = vmovl_s8(x_high);
      const int16x8_t a_high_16 = vreinterpretq_s16_u16(vmovl_u8(a_high));
      const int16x8_t a_low_16 = vreinterpretq_s16_u16(vmovl_u8(a_low));

      sum_vec =
          vdotq_s32(sum_vec, vreinterpretq_s32_s16(a_low_16), vreinterpretq_s32_s16(x_low_16));
      sum_vec =
          vdotq_s32(sum_vec, vreinterpretq_s32_s16(a_high_16), vreinterpretq_s32_s16(x_high_16));

      result[j] = vaddvq_s32(sum_vec);
    }
  }

  inline void matvec_product_B_32(const uint8_t *A, const int8_t *x, float *result,
                                  const size_t rows, const size_t cols) const {
    const int8x16_t x_val_high = vld1q_s8(&x[16]);
    const int8x16_t x_val_low = vld1q_s8(&x[0]);

    for (size_t j = 0; j < cols; ++j) {
      int32x4_t sum_vec = vdupq_n_s32(0);

      const uint8x16_t a_vec = vld1q_u8(&A[j * 16]);
      const uint8x16_t a_high_u8 = vshrq_n_u8(a_vec, 4);
      const uint8x16_t a_low_u8 = vandq_u8(a_vec, vdupq_n_u8(0x0F));
      const int8x16_t a_high_s8 = vreinterpretq_s8_u8(a_high_u8);
      const int8x16_t a_low_s8 = vreinterpretq_s8_u8(a_low_u8);

      sum_vec = vdotq_s32(sum_vec, a_high_s8, x_val_high);
      sum_vec = vdotq_s32(sum_vec, a_low_s8, x_val_low);

      result[j] = vaddvq_s32(sum_vec);
    }
  }

  inline void matvec_product_B_64(const uint8_t *A, const int8_t *x, float *result,
                                  const size_t rows, const size_t cols) const {
    for (size_t j = 0; j < cols; ++j) {
      int32_t sum = 0;

      for (int k = 0; k < 32; ++k) {
        sum += ((int32_t)(A[k + j * 32] >> 4)) * ((int32_t)x[k + 32]);
        sum += ((int32_t)(A[k + j * 32] & 0xF)) * ((int32_t)x[k]);
      }

      result[j] = sum;
    }
  }
#else
  inline void matvec_product_A(const uint8_t *A, const int8_t *x, float *result, const size_t rows,
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

  inline void matvec_product_B_16(const uint8_t *A, const int8_t *x, float *result,
                                  const size_t rows, const size_t cols) const {
    for (size_t j = 0; j < cols; ++j) {
      int32_t sum = 0;

      for (int k = 0; k < 8; ++k) {
        sum += ((int32_t)(A[k + j * 8] >> 4)) * ((int32_t)x[k + 8]);
        sum += ((int32_t)(A[k + j * 8] & 0xF)) * ((int32_t)x[k]);
      }

      result[j] = sum;
    }
  }

  inline void matvec_product_B_32(const uint8_t *A, const int8_t *x, float *result,
                                  const size_t rows, const size_t cols) const {
    for (size_t j = 0; j < cols; ++j) {
      int32_t sum = 0;

      for (int k = 0; k < 16; ++k) {
        sum += ((int32_t)(A[k + j * 16] >> 4)) * ((int32_t)x[k + 16]);
        sum += ((int32_t)(A[k + j * 16] & 0xF)) * ((int32_t)x[k]);
      }

      result[j] = sum;
    }
  }

  inline void matvec_product_B_64(const uint8_t *A, const int8_t *x, float *result,
                                  const size_t rows, const size_t cols) const {
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

  inline float quantize_vector(const float *v, const int len, int8_t *result) const {
    const float factor = compute_quantization_factor(v, len, 4);
    for (int i = 0; i < len; ++i) {
      result[i] = (int8_t)nearest_int(factor * v[i]);
    }
    return factor;
  }

  inline void quantize_matrix_B_unsigned(const ColMatrix &A, uint8_t *result,
                                         float *factors) const {
    const int n = A.rows() - 1;
    const int qk = n;

    for (int i = 0; i < A.cols(); ++i) {
      const float *v = A.data() + i * (n + 1) + 1;
      const float factor = compute_quantization_factor(v, n, 4);
      for (int k = 0; k < qk / 2; ++k) {
        const uint8_t a = MIN(15, factor * v[k] + 8.5f);
        const uint8_t b = MIN(15, factor * v[qk / 2 + k] + 8.5f);

        result[i * n / 2 + k] = a;
        result[i * n / 2 + k] |= b << 4;
      }
      factors[i] = factor;
    }
  }

  inline void quantize_matrix_A_unsigned(const ColMatrix &A, uint8_t *result,
                                         float *factors) const {
    constexpr int qk = 32;
    const int n = A.rows();
    const int nb = n / qk;

    for (int i = 0; i < A.cols(); ++i) {
      const float *v = A.data() + i * n;
      const float factor = compute_quantization_factor(v, n, 4);
      for (int j = 0; j < nb; ++j) {
        for (int k = 0; k < qk / 2; ++k) {
          const uint8_t a = MIN(15, factor * v[j * qk + 0 + k] + 8.5f);
          const uint8_t b = MIN(15, factor * v[j * qk + qk / 2 + k] + 8.5f);

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
  inline void matvec_product_A(const uint8_t *A, const int8_t *x, float *result, const size_t rows,
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

  void matvec_product_B_16(const uint8_t *A, const int8_t *x, float *result, const size_t rows,
                           const size_t cols) const {
    const __m128i vec_chunk = _mm_loadu_si128((const __m128i *)(x));
    for (size_t j = 0; j < cols; ++j) {
      const __m128i col_chunk = _mm_loadu_si128((const __m128i *)(A + j * 16));
      const __m128i sum = dpbusd(col_chunk, vec_chunk);
      result[j] = horizontal_add(sum);
    }
  }

  void matvec_product_B_32(const uint8_t *A, const int8_t *x, float *result, const size_t rows,
                           const size_t cols) const {
    const __m256i vec_chunk = _mm256_loadu_si256((const __m256i *)(x));
    for (size_t j = 0; j < cols; ++j) {
      const __m256i col_chunk = _mm256_loadu_si256((const __m256i *)(A + j * 32));
      const __m256i sum = dpbusd(col_chunk, vec_chunk);
      result[j] = horizontal_add(sum);
    }
  }

#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
  void matvec_product_B_64(const uint8_t *A, const int8_t *x, float *result, const size_t rows,
                           const size_t cols) const {
    const __m512i vec_chunk = _mm512_loadu_si512((const __m512i *)(x));
    for (size_t j = 0; j < cols; ++j) {
      const __m512i col_chunk = _mm512_loadu_si512((const __m512i *)(A + j * 64));
      const __m512i sum = dpbusd(col_chunk, vec_chunk);
      result[j] = _mm512_reduce_add_epi32(sum);
    }
  }
#else
  void matvec_product_B_64(const uint8_t *A, const int8_t *x, float *result, const size_t rows,
                           const size_t cols) const {
    matvec_product_A(A, x, result, rows, cols);
  }
#endif
#else
  inline void matvec_product_A(const uint8_t *A, const int8_t *x, float *result, const size_t rows,
                               const size_t cols) const {
    for (size_t j = 0; j < cols; ++j) {
      int32_t sum = 0;

      for (size_t i = 0; i < rows; ++i) {
        sum += ((int32_t)A[i + j * rows]) * ((int32_t)x[i]);
      }

      result[j] = sum;
    }
  }

  inline void matvec_product_B_16(const uint8_t *A, const int8_t *x, float *result,
                                  const size_t rows, const size_t cols) const {
    matvec_product_A(A, x, result, rows, cols);
  }

  inline void matvec_product_B_32(const uint8_t *A, const int8_t *x, float *result,
                                  const size_t rows, const size_t cols) const {
    matvec_product_A(A, x, result, rows, cols);
  }

  inline void matvec_product_B_64(const uint8_t *A, const int8_t *x, float *result,
                                  const size_t rows, const size_t cols) const {
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

  inline float quantize_vector(const float *v, const int len, int8_t *result) const {
    const float factor = compute_quantization_factor(v, len, 8);
    for (int i = 0; i < len; ++i) {
      result[i] = (int8_t)nearest_int(factor * v[i]);
    }
    return factor;
  }

  inline float quantize_vector_unsigned(const float *v, const int len, uint8_t *result) const {
    const float factor = compute_quantization_factor(v, len, 8);
    for (int i = 0; i < len; ++i) {
      result[i] = (uint8_t)(nearest_int(factor * v[i]) + 128);
    }
    return factor;
  }

  inline void quantize_matrix_B_unsigned(const ColMatrix &A, uint8_t *result,
                                         float *factors) const {
    for (int i = 0; i < A.cols(); ++i) {
      factors[i] = quantize_vector_unsigned(A.data() + i * A.rows() + 1, A.rows() - 1,
                                            result + i * (A.rows() - 1));
    }
  }

  inline void quantize_matrix_A_unsigned(const ColMatrix &A, uint8_t *result,
                                         float *factors) const {
    for (int i = 0; i < A.cols(); ++i) {
      factors[i] =
          quantize_vector_unsigned(A.data() + i * A.rows(), A.rows(), result + i * A.rows());
    }
  }
};

}  // namespace Lorann