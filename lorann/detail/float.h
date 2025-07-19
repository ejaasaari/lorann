#pragma once

#include <cstddef>

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
struct Traits<float> {
  static constexpr TypeMarker type_marker = FLOAT32;
  static constexpr int dim_divisor = 1;

  static inline Lorann::ColVector to_float_vector(const float *x, int d) {
    return Eigen::Map<const Lorann::ColVector>(x, d);
  }

  static inline Lorann::MappedMatrix to_float_matrix(const float *data, const int n, const int d) {
    return {data, static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(d)};
  }

  static inline float dot_product(const float *a, const float *b, const size_t dim) {
    simsimd_distance_t distance;
    simsimd_dot_f32(a, b, dim, &distance);
    return distance;
  }

  static inline float squared_euclidean(const float *a, const float *b, const size_t dim) {
    simsimd_distance_t distance;
    simsimd_l2sq_f32(a, b, dim, &distance);
    return distance;
  }
};

}  // namespace detail

}  // namespace Lorann