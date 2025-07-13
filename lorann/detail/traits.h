#pragma once

namespace Lorann {

namespace detail {

enum TypeMarker { FLOAT32 = 0, FLOAT16 = 1, UINT8 = 2, BINARY = 3 };

template <typename T>
struct Traits;

}  // namespace detail

}  // namespace Lorann