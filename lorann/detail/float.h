#pragma once

#include <cstddef>

#include "traits.h"
#include "utils.h"

namespace Lorann {

namespace detail {

template <>
struct Traits<float> {
  static constexpr TypeMarker type_marker = FLOAT32;
  static constexpr int dim_divisor = 1;

  static inline Lorann::ColVector to_float_vector(const float *data, const int d) {
    return Eigen::Map<const Lorann::ColVector>(data, d);
  }

  static inline Lorann::MappedMatrix to_float_matrix(const float *data, const int n, const int d) {
    return {data, static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(d)};
  }

  static inline lorann_dist_t dot_product(const float *a, const float *b, const size_t dim) {
    simsimd_distance_t distance;
    simsimd_dot_f32(a, b, dim, &distance);
    return distance;
  }

  static inline lorann_dist_t squared_euclidean(const float *a, const float *b, const size_t dim) {
    simsimd_distance_t distance;
    simsimd_l2sq_f32(a, b, dim, &distance);
    return distance;
  }
};

}  // namespace detail

}  // namespace Lorann