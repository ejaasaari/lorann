#pragma once

#include <cstdint>
#include <memory>

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
struct Traits<uint8_t> {
  static constexpr TypeMarker type_marker = UINT8;
  static constexpr int dim_divisor = 1;

  static inline Lorann::ColVector to_float_vector(const uint8_t *x, int d) {
    return Eigen::Map<const Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>>(x, d).cast<float>();
  }

  static inline Lorann::MappedMatrix to_float_matrix(const uint8_t *data, const int n,
                                                     const int d) {
    auto buf = std::shared_ptr<float[]>{new float[n * d], std::default_delete<float[]>()};
    for (int i = 0; i < n * d; ++i) {
      buf[i] = static_cast<float>(data[i]);
    }
    return {buf.get(), n, d, buf};
  }

  static inline float dot_product(const uint8_t *a, const uint8_t *b, const size_t dim) {
    simsimd_distance_t distance;
    simsimd_dot_u8(a, b, dim, &distance);
    return distance;
  }

  static inline float squared_euclidean(const uint8_t *a, const uint8_t *b, const std::size_t dim) {
    simsimd_distance_t distance;
    simsimd_l2sq_u8(a, b, dim, &distance);
    return distance;
  }
};

}  // namespace detail

}  // namespace Lorann