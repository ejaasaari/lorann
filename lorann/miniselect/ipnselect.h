#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>

// SPDX-License-Identifier: MIT
// Portions adapted from Rust's ipnsort implementation:
// https://github.com/rust-lang/rust/tree/master/library/core/src/slice/sort
// Copyright (c) The Rust Project Contributors.

namespace miniselect {
namespace ipnselect_detail {

template <typename T, typename Compare>
inline std::size_t partition_lomuto_branchless_cyclic(T *values, const std::size_t len,
                                                      const T &pivot, Compare &less) {
  static_assert(std::is_trivially_copyable<T>::value,
                "ipnselect requires trivially copyable values");

  if (len == 0) return 0;

  T gap_value = values[0];
  T *gap = values;
  T *right = values + 1;
  std::size_t num_less = 0;

  const auto loop_body = [&](T *source) {
    const bool source_is_less = less(*source, pivot);
    T *left = values + num_less;

    // Moving every element through the current gap makes the partition loop
    // independent of the comparison result. Only num_less is conditional.
    *gap = *left;
    *left = *source;
    gap = source;
    num_less += static_cast<std::size_t>(source_is_less);
  };

  // LLVM does not reliably unroll this loop on every target. ipnsort uses a
  // fixed factor of two for records up to 16 bytes, including our int indices.
  std::size_t remaining = len - 1;
  if constexpr (sizeof(T) <= 16) {
    while (remaining >= 2) {
      loop_body(right);
      loop_body(right + 1);
      right += 2;
      remaining -= 2;
    }
  }
  while (remaining > 0) {
    loop_body(right);
    ++right;
    --remaining;
  }

  loop_body(&gap_value);
  return num_less;
}

template <typename T, typename Compare>
inline std::size_t partition(T *values, const std::size_t len, const std::size_t pivot_pos,
                             Compare &less) {
  std::swap(values[0], values[pivot_pos]);
  const T pivot = values[0];
  const std::size_t num_less = partition_lomuto_branchless_cyclic(values + 1, len - 1, pivot, less);
  std::swap(values[0], values[num_less]);
  return num_less;
}

template <typename T, typename Compare>
inline const T *median3(const T *a, const T *b, const T *c, Compare &less) {
  const bool x = less(*a, *b);
  const bool y = less(*a, *c);
  if (x == y) {
    const bool z = less(*b, *c);
    return (z ^ x) ? c : b;
  }
  return a;
}

template <typename T, typename Compare>
inline const T *recursive_median3(const T *a, const T *b, const T *c, const std::size_t n,
                                  Compare &less) {
  constexpr std::size_t recursive_threshold = 64;
  if (n * 8 >= recursive_threshold) {
    const std::size_t eighth = n / 8;
    a = recursive_median3(a, a + eighth * 4, a + eighth * 7, eighth, less);
    b = recursive_median3(b, b + eighth * 4, b + eighth * 7, eighth, less);
    c = recursive_median3(c, c + eighth * 4, c + eighth * 7, eighth, less);
  }
  return median3(a, b, c, less);
}

template <typename T, typename Compare>
inline std::size_t choose_pivot(const T *values, const std::size_t len, Compare &less) {
  constexpr std::size_t recursive_threshold = 64;
  const std::size_t eighth = len / 8;
  const T *a = values;
  const T *b = values + eighth * 4;
  const T *c = values + eighth * 7;
  const T *pivot =
      len < recursive_threshold ? median3(a, b, c, less) : recursive_median3(a, b, c, eighth, less);
  return static_cast<std::size_t>(pivot - values);
}

template <typename T, typename Compare>
inline void insertion_sort(T *values, const std::size_t len, Compare &less) {
  for (std::size_t i = 1; i < len; ++i) {
    const T value = values[i];
    std::size_t j = i;
    while (j > 0 && less(value, values[j - 1])) {
      values[j] = values[j - 1];
      --j;
    }
    values[j] = value;
  }
}

template <typename T, typename Compare>
inline std::size_t median_index(const T *values, Compare &less, std::size_t a, std::size_t b,
                                std::size_t c) {
  if (less(values[c], values[a])) std::swap(a, c);
  if (less(values[c], values[b])) return c;
  if (less(values[b], values[a])) return a;
  return b;
}

template <typename T, typename Compare>
inline void ninther(T *values, Compare &less, const std::size_t a, std::size_t b,
                    const std::size_t c, std::size_t d, const std::size_t e, std::size_t f,
                    const std::size_t g, std::size_t h, const std::size_t i) {
  b = median_index(values, less, a, b, c);
  h = median_index(values, less, g, h, i);
  if (less(values[h], values[b])) std::swap(b, h);
  if (less(values[f], values[d])) std::swap(d, f);

  if (!less(values[e], values[d])) {
    if (less(values[f], values[e])) {
      d = f;
    } else {
      if (less(values[e], values[b]))
        std::swap(values[e], values[b]);
      else if (less(values[h], values[e]))
        std::swap(values[e], values[h]);
      return;
    }
  }

  if (less(values[d], values[b]))
    d = b;
  else if (less(values[h], values[d]))
    d = h;
  std::swap(values[d], values[e]);
}

template <typename T, typename Compare>
inline void deterministic_select(T *values, std::size_t len, std::size_t index, Compare &less);

template <typename T, typename Compare>
inline std::size_t median_of_ninthers(T *values, const std::size_t len, Compare &less) {
  const std::size_t fraction = len <= 1024 ? len / 12 : (len <= 128 * 1024 ? len / 64 : len / 1024);
  const std::size_t pivot = fraction / 2;
  const std::size_t lo = len / 2 - pivot;
  const std::size_t hi = fraction + lo;
  const std::size_t gap = (len - 9 * fraction) / 4;
  std::size_t a = lo - 4 * fraction - gap;
  std::size_t b = hi + gap;

  for (std::size_t i = lo; i < hi; ++i) {
    ninther(values, less, a, i - fraction, b, a + 1, i, b + 1, a + 2, i + fraction, b + 2);
    a += 3;
    b += 3;
  }

  deterministic_select(values + lo, fraction, pivot, less);
  return partition(values, len, lo + pivot, less);
}

template <typename T, typename Compare>
inline void deterministic_select(T *values, std::size_t len, std::size_t index, Compare &less) {
  constexpr std::size_t insertion_threshold = 16;

  for (;;) {
    if (len <= insertion_threshold) {
      insertion_sort(values, len, less);
      return;
    }
    if (index == len - 1) {
      std::size_t maximum = 0;
      for (std::size_t i = 1; i < len; ++i) {
        if (less(values[maximum], values[i])) maximum = i;
      }
      std::swap(values[maximum], values[index]);
      return;
    }
    if (index == 0) {
      std::size_t minimum = 0;
      for (std::size_t i = 1; i < len; ++i) {
        if (less(values[i], values[minimum])) minimum = i;
      }
      std::swap(values[minimum], values[0]);
      return;
    }

    const std::size_t pivot = median_of_ninthers(values, len, less);
    if (pivot == index) return;
    if (pivot > index) {
      len = pivot;
    } else {
      values += pivot + 1;
      len -= pivot + 1;
      index -= pivot + 1;
    }
  }
}

template <typename T, typename Compare>
inline void introselect(T *values, std::size_t len, std::size_t index, Compare &less) {
  constexpr std::size_t insertion_threshold = 16;
  int limit = 16;
  const T *ancestor_pivot = nullptr;

  for (;;) {
    if (len <= insertion_threshold) {
      insertion_sort(values, len, less);
      return;
    }
    if (limit == 0) {
      deterministic_select(values, len, index, less);
      return;
    }
    --limit;

    const std::size_t pivot_pos = choose_pivot(values, len, less);
    if (ancestor_pivot != nullptr && !less(*ancestor_pivot, values[pivot_pos])) {
      const auto less_equal = [&less](const T &a, const T &b) { return !less(b, a); };
      const std::size_t num_less_equal = partition(values, len, pivot_pos, less_equal);
      const std::size_t middle = num_less_equal + 1;
      if (middle > index) return;
      values += middle;
      len -= middle;
      index -= middle;
      ancestor_pivot = nullptr;
      continue;
    }

    const std::size_t middle = partition(values, len, pivot_pos, less);
    const T *pivot = values + middle;
    if (middle < index) {
      values += middle + 1;
      len -= middle + 1;
      index -= middle + 1;
      ancestor_pivot = pivot;
    } else if (middle > index) {
      len = middle;
    } else {
      return;
    }
  }
}

}  // namespace ipnselect_detail

// Partitions [first, last) so that the element at nth is the one which would
// occur there in sorted order. The comparator must not throw: the cyclic
// partition intentionally omits the recovery guard needed for fallible moves.
template <typename T, typename Compare>
inline void ipnselect_branchless(T *first, T *nth, T *last, Compare less) {
  static_assert(std::is_trivially_copyable<T>::value,
                "ipnselect requires trivially copyable values");
  if (first == last || nth == last) return;
  ipnselect_detail::introselect(first, static_cast<std::size_t>(last - first),
                                static_cast<std::size_t>(nth - first), less);
}

}  // namespace miniselect
