#pragma once

namespace Lorann {

namespace detail {

enum TypeMarker { FLOAT32 = 0, FLOAT16 = 1, BFLOAT16 = 2, UINT8 = 3, BINARY = 4 };

template <typename T>
struct Traits;

}  // namespace detail

}  // namespace Lorann