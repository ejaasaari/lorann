#pragma once

#include <cstddef>
#include <cstdint>

#include "traits.h"
#include "utils.h"

namespace Lorann {

namespace detail {

template <>
struct Traits<uint8_t> {
  static constexpr TypeMarker type_marker = UINT8;
  static constexpr int dim_divisor = 1;

  static inline void convert_u8_to_f32(const uint8_t *src, float *dst, const size_t count) {
    for (size_t i = 0; i < count; ++i) {
      dst[i] = static_cast<float>(src[i]);
    }
  }

  static inline Lorann::ColVector to_float_vector(const uint8_t *data, const int d) {
    Lorann::ColVector vec(d);
    convert_u8_to_f32(data, vec.data(), d);
    return vec;
  }

  static inline Lorann::MappedMatrix to_float_matrix(const uint8_t *data, const int n,
                                                     const int d) {
    auto buf = make_aligned_shared_array<float>(n * d);
    convert_u8_to_f32(data, buf.get(), n * d);
    return {buf.get(), n, d, buf};
  }

  static inline lorann_dist_t dot_product(const uint8_t *a, const uint8_t *b, const size_t dim) {
    simsimd_distance_t distance;
    simsimd_dot_u8(a, b, dim, &distance);
    return distance;
  }

  static inline lorann_dist_t squared_euclidean(const uint8_t *a, const uint8_t *b,
                                                const std::size_t dim) {
    simsimd_distance_t distance;
    simsimd_l2sq_u8(a, b, dim, &distance);
    return distance;
  }
};

}  // namespace detail

}  // namespace Lorann