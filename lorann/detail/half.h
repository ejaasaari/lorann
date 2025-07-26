#pragma once

#include <simsimd/simsimd.h>

#include <cstddef>

#include "traits.h"
#include "utils.h"

namespace Lorann {

namespace detail {

template <>
struct Traits<simsimd_f16_t> {
  static constexpr TypeMarker type_marker = FLOAT16;
  static constexpr size_t dim_divisor = 1;

  static inline void convert_f16_to_f32(const simsimd_f16_t *src, float *dst, const size_t count) {
    for (size_t i = 0; i < count; ++i) {
      dst[i] = simsimd_f16_to_f32(src + i);
    }
  }

  static inline Lorann::ColVector to_float_vector(const simsimd_f16_t *data, const int d) {
    Lorann::ColVector vec(d);
    convert_f16_to_f32(data, vec.data(), d);
    return vec;
  }

  static inline Lorann::MappedMatrix to_float_matrix(const simsimd_f16_t *data, const int n,
                                                     const int d) {
    auto buf = make_aligned_shared_array<float>(n * d);
    convert_f16_to_f32(data, buf.get(), n * d);
    return {buf.get(), n, d, buf};
  }

  static inline lorann_dist_t dot_product(const simsimd_f16_t *a, const simsimd_f16_t *b,
                                          const size_t dim) {
    simsimd_distance_t distance;
    simsimd_dot_f16(a, b, dim, &distance);
    return distance;
  }

  static inline lorann_dist_t squared_euclidean(const simsimd_f16_t *a, const simsimd_f16_t *b,
                                                const size_t dim) {
    simsimd_distance_t distance;
    simsimd_l2sq_f16(a, b, dim, &distance);
    return distance;
  }
};

}  // namespace detail

}  // namespace Lorann
