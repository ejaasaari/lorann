#pragma once

#include <cstddef>
#include <cstdint>

#include "traits.h"
#include "utils.h"

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

  static inline void unpack(const uint8_t *in, float *out, const size_t d) {
    for (size_t i = 0; i < d / 8; ++i) {
      uint8_t byte = in[i];
      for (size_t j = 0; j < 8; ++j) {
        out[i * 8 + j] = ((byte >> (7 - j)) & 1) ? 1.0f : 0.0f;
      }
    }
  }

  static inline Lorann::ColVector to_float_vector(const BinaryType *data, const int d) {
    Lorann::ColVector vec(d);
    unpack(reinterpret_cast<const uint8_t *>(data), vec.data(), d);
    return vec;
  }

  static inline Lorann::MappedMatrix to_float_matrix(const BinaryType *data, const int n,
                                                     const int d) {
    auto buf = make_aligned_shared_array<float>(n * d);
    unpack(reinterpret_cast<const uint8_t *>(data), buf.get(), n * d);
    return {buf.get(), n, d, buf};
  }

  static inline lorann_dist_t dot_product(const BinaryType *x, const BinaryType *y,
                                          const size_t n_bytes) {
    const simsimd_b8_t *a = reinterpret_cast<const simsimd_b8_t *>(x);
    const simsimd_b8_t *b = reinterpret_cast<const simsimd_b8_t *>(y);

    simsimd_distance_t distance;
    simsimd_hamming_b8(a, b, n_bytes, &distance);
    return distance;
  }

  static inline lorann_dist_t squared_euclidean(const BinaryType *x, const BinaryType *y,
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