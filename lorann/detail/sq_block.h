#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>

#if defined(__AVX512VNNI__)
#include <immintrin.h>
#endif

namespace Lorann {
namespace detail {
template <int Bits, int Coefficients>
struct SQBlockLayout {
  static_assert(Coefficients == 16 || Coefficients == 32 || Coefficients == 64,
                "SQ blocks support 16, 32, or 64 coefficients");
  static_assert(Bits == 4 || Bits == 8, "SQ blocks require 4 or 8 bits");
  static constexpr int column_bytes = Coefficients * Bits / 8;
  static constexpr int parts = column_bytes / 4;
  static constexpr int bytes = 16 * column_bytes;
};

template <typename Function>
inline decltype(auto) dispatch_sq_coefficients(int coefficients, Function &&function) {
  switch (coefficients) {
    case 16:
      return std::forward<Function>(function)(std::integral_constant<int, 16>{});
    case 32:
      return std::forward<Function>(function)(std::integral_constant<int, 32>{});
    case 64:
      return std::forward<Function>(function)(std::integral_constant<int, 64>{});
    default:
      throw std::invalid_argument("SQ blocks require 16, 32, or 64 coefficients");
  }
}

// Group four-byte slices from 16 columns so each SIMD lane scores one point.
// Preserve coefficient encoding; leave partial blocks in column order.
template <int Bits, int Coefficients>
inline void pack_sq_blocks(uint8_t *data, int cols) {
  using Layout = SQBlockLayout<Bits, Coefficients>;
  uint8_t tmp[Layout::bytes];
  for (int c = 0; c + 16 <= cols; c += 16) {
    std::memcpy(tmp, data + static_cast<std::size_t>(c) * Layout::column_bytes, sizeof(tmp));
    for (int part = 0; part < Layout::parts; ++part)
      for (int row = 0; row < 16; ++row)
        std::memcpy(data + static_cast<std::size_t>(c) * Layout::column_bytes + part * 64 + row * 4,
                    tmp + row * Layout::column_bytes + part * 4, 4);
  }
}
template <int Bits, int Coefficients>
inline void unpack_sq_block(const uint8_t *data, uint8_t *out) {
  using Layout = SQBlockLayout<Bits, Coefficients>;
  for (int part = 0; part < SQBlockLayout<Bits, Coefficients>::parts; ++part)
    for (int row = 0; row < 16; ++row)
      std::memcpy(out + row * Layout::column_bytes + part * 4, data + part * 64 + row * 4, 4);
}
#if defined(__AVX512VNNI__)
template <int Bits, int Coefficients>
inline int score_sq_blocks(const uint8_t *data, const int8_t *query, float *result, int cols,
                           const float *scales, const float *fix, float compensation, float factor,
                           float correction, const float *norms) {
  using Layout = SQBlockLayout<Bits, Coefficients>;
  constexpr int parts = Coefficients / 4;
  __m512i q[parts];
  for (int i = 0; i < parts; ++i) {
    int v;
    std::memcpy(&v, query + 4 * i, 4);
    q[i] = _mm512_set1_epi32(v);
  }
  const __m256 comp = _mm256_set1_ps(compensation);
  const __m256 invf = _mm256_set1_ps(1.f / factor);
  const __m256 corr = _mm256_set1_ps(correction);
  auto finish = [&](const __m256 raw, int c) {
    const __m256 scale = _mm256_loadu_ps(scales + c);
    const __m256 residual = _mm256_mul_ps(_mm256_sub_ps(raw, comp), invf);
    __m256 reciprocal = _mm256_rcp_ps(scale);
    reciprocal = _mm256_mul_ps(
        reciprocal, _mm256_sub_ps(_mm256_set1_ps(2.f), _mm256_mul_ps(scale, reciprocal)));
    __m256 output =
        _mm256_fmadd_ps(corr, _mm256_loadu_ps(fix + c), _mm256_mul_ps(residual, reciprocal));
    if (norms) output = _mm256_add_ps(output, _mm256_loadu_ps(norms + c));
    _mm256_storeu_ps(result + c, output);
  };
  int c = 0;
  for (; c + 16 <= cols; c += 16) {
    const uint8_t *block = data + static_cast<std::size_t>(c) * Layout::column_bytes;
    __m512i a = _mm512_setzero_si512(), b = _mm512_setzero_si512();
    for (int part = 0; part < parts / 2; ++part) {
      if constexpr (Bits == 4) {
        // Low nibbles encode the first half of the coefficients; high nibbles encode the second.
        const __m512i packed = _mm512_loadu_si512(block + part * 64);
        const __m512i mask = _mm512_set1_epi8(0x0f);
        const __m512i low = _mm512_and_si512(packed, mask);
        const __m512i high = _mm512_and_si512(_mm512_srli_epi16(packed, 4), mask);
        a = _mm512_dpbusd_epi32(a, low, q[part]);
        b = _mm512_dpbusd_epi32(b, high, q[part + parts / 2]);
      } else {
        a = _mm512_dpbusd_epi32(a, _mm512_loadu_si512(block + part * 64), q[part]);
        b = _mm512_dpbusd_epi32(b, _mm512_loadu_si512(block + (part + parts / 2) * 64),
                                q[part + parts / 2]);
      }
    }
    const __m512 output = _mm512_cvtepi32_ps(_mm512_add_epi32(a, b));
    finish(_mm512_castps512_ps256(output), c);
    finish(_mm512_extractf32x8_ps(output, 1), c + 8);
  }
  return c;
}
#endif
}  // namespace detail
}  // namespace Lorann
