#pragma once

#include "utils.h"
#include <array>
#include <cmath>
#include <limits>

namespace Lorann {
namespace joint_quantization {

// Jointly minimize ||v - s*k||^2 for symmetric integer codes. For fixed codes,
// s = dot(v,k)/dot(k,k). Codes change only at s = |v_i|/(j + 1/2), so sweeping
// these intervals replaces a clipping grid and a fixed number of Lloyd steps.
// Sorted magnitudes give one ordered boundary stream per integer level.
// Scratch storage is reused for every column of a model matrix.
template <int Bits>
class ColumnFitter {
  static_assert(Bits == 4 || Bits == 8, "Model quantization supports SQ4 and SQ8");
  static constexpr int kLimit = (1 << (Bits - 1)) - 1;

  struct Boundary {
    double scale;
    double inverse;
    size_t index;
    int level;
  };

 public:
  // The stored factor is the reciprocal decoding scale, as required by the
  // existing integer kernels. Keep both factor and reciprocal normal: the
  // SIMD decoder uses an approximate reciprocal followed by a Newton step.
  static constexpr float kMinFactor = std::numeric_limits<float>::min();
  static constexpr float kMaxFactor = 1.f / kMinFactor;

  static int encode(float value, float factor) {
    const double scaled = std::clamp(double(value) * factor, -double(kLimit), double(kLimit));
    // Explicit symmetric nearest rounding, independent of the host rounding mode.
    const int magnitude = int(std::floor(std::abs(scaled) + 0.5));
    return scaled < 0 ? -magnitude : magnitude;
  }

  float fit(const float *values, int size) {
    double maximum = 0;
    for (int i = 0; i < size; ++i) {
      if (!std::isfinite(values[i]))
        throw std::invalid_argument("Model quantization requires finite values");
      maximum = std::max(maximum, std::abs(double(values[i])));
    }
    if (maximum == 0) return 1.f;

    magnitudes_.clear();
    magnitudes_.reserve(size);
    double energy = 0;
    for (int i = 0; i < size; ++i) {
      if (values[i] == 0) continue;
      const double x = std::abs(double(values[i])) / maximum;
      magnitudes_.push_back(x);
      energy += x * x;
    }
    std::sort(magnitudes_.begin(), magnitudes_.end());

    double best_scale = 1. / kLimit;
    double best_error = error(best_scale);
    // Analytically refitting the absmax codes supplies a tighter initial bound.
    double dot = 0, norm = 0;
    for (double x : magnitudes_) {
      const double code = std::floor(x * kLimit + 0.5);
      dot += x * code;
      norm += code * code;
    }
    const double refitted = dot / norm;
    const double refitted_error = error(refitted);
    if (refitted_error < best_error) {
      best_scale = refitted;
      best_error = refitted_error;
    }

    if (best_error > 0) {
      // Saturated coordinates alone give an error lower bound. Solve
      // sum_{x>t} (x-t)^2 = best_error for t = kLimit*s. Smaller scales cannot
      // improve the current result. Sorting makes each quadratic tail explicit.
      double tail_sum = 0, tail_energy = 0, lower = 0;
      for (size_t i = magnitudes_.size(); i > 0; --i) {
        const double x = magnitudes_[i - 1];
        tail_sum += x;
        tail_energy += x * x;
        const double count = magnitudes_.size() - i + 1;
        if (tail_energy <= best_error) continue;
        const double discriminant =
            std::max(0., tail_sum * tail_sum - count * (tail_energy - best_error));
        // Rationalized smaller root avoids subtracting nearly equal numbers.
        const double threshold =
            (tail_energy - best_error) / (tail_sum + std::sqrt(discriminant));
        if (i == 1 || threshold >= magnitudes_[i - 2]) {
          lower = std::nextafter(threshold / kLimit, 0.);
          break;
        }
      }

      size_t heap_size = 0;
      for (int level = 0; level < kLimit; ++level) {
        const double inverse = 1. / (level + 0.5);
        const auto first = std::upper_bound(
            magnitudes_.begin(), magnitudes_.end(), lower,
            [inverse](double bound, double x) { return bound < x * inverse; });
        const size_t index = first - magnitudes_.begin();
        if (index < magnitudes_.size())
          heap_[heap_size++] = {*first * inverse, inverse, index, level};
      }
      std::make_heap(heap_.begin(), heap_.begin() + heap_size,
                     [](const Boundary &a, const Boundary &b) { return a.scale > b.scale; });

      dot = 0;
      norm = 0;
      double zero_energy = 0;
      for (double x : magnitudes_) {
        int code = lower > 0
                       ? int(std::clamp(std::ceil(x / lower - 0.5), 0., double(kLimit)))
                       : kLimit;
        // Match the boundary streams exactly, including ties at the lower bound.
        while (code > 0 && x * (1. / (code - 0.5)) <= lower) --code;
        while (code < kLimit && x * (1. / (code + 0.5)) > lower) ++code;
        dot += x * code;
        norm += code * code;
        if (code == 0) zero_energy += x * x;
      }
      double best_score = energy - best_error;
      for (;;) {
        const double upper = heap_size ? heap_[0].scale : std::numeric_limits<double>::infinity();
        // Most intervals do not contain their stationary point; reject them
        // using multiplication before paying for a division and error evaluation.
        if (norm > 0 && dot >= lower * norm && dot <= upper * norm) {
          const double scale = dot / norm;
          if (dot * scale >= best_score) {
            const double candidate_error = error(scale);
            if (candidate_error < best_error) {
              best_scale = scale;
              best_error = candidate_error;
              best_score = energy - best_error;
            }
          }
        }
        // Coordinates rounded to zero stay zero as scale grows. Their energy
        // is another monotone lower bound, and can terminate the sweep early.
        if (heap_size == 0 || zero_energy >= best_error) break;

        Boundary next = heap_[0];
        const double x = magnitudes_[next.index];
        dot -= x;
        norm -= 2 * next.level + 1;
        lower = next.scale;
        if (next.level == 0) zero_energy += x * x;
        if (++next.index == magnitudes_.size()) {
          next = heap_[--heap_size];
          if (heap_size == 0) continue;
        } else {
          next.scale = magnitudes_[next.index] * next.inverse;
        }
        // Each stream advances monotonically: one root replacement/sift-down
        // suffices, instead of separate priority-queue pop and push operations.
        size_t parent = 0, child = 1;
        while (child < heap_size) {
          if (child + 1 < heap_size && heap_[child + 1].scale < heap_[child].scale) ++child;
          if (next.scale <= heap_[child].scale) break;
          heap_[parent] = heap_[child];
          parent = child;
          child = 2 * parent + 1;
        }
        heap_[parent] = next;
      }
    }

    // The ideal decoding scale need not have a representable float reciprocal.
    // Evaluate both adjacent representable factors, using the codes that will
    // actually be packed. Retain absmax as a guard against numerical roundoff.
    float factor = bounded_factor(double(kLimit) / maximum);
    double loss = stored_error(values, size, factor);
    const double ideal_factor = 1. / (best_scale * maximum);
    const float rounded = bounded_factor(ideal_factor);
    auto consider = [&](float candidate) {
      if (candidate < kMinFactor || candidate > kMaxFactor) return;
      const double candidate_loss = stored_error(values, size, candidate);
      if (candidate_loss < loss) {
        factor = candidate;
        loss = candidate_loss;
      }
    };
    consider(rounded);
    if (rounded > ideal_factor) consider(std::nextafter(rounded, 0.f));
    if (rounded < ideal_factor)
      consider(std::nextafter(rounded, std::numeric_limits<float>::infinity()));
    return factor;
  }

 private:
  static float bounded_factor(double factor) {
    return float(std::clamp(factor, double(kMinFactor), double(kMaxFactor)));
  }

  static double stored_error(const float *values, int size, float factor) {
    double result = 0;
    for (int i = 0; i < size; ++i) {
      const double residual = double(values[i]) - encode(values[i], factor) / double(factor);
      result += residual * residual;
    }
    return result;
  }

  double error(double scale) const {
    double result = 0;
    for (double x : magnitudes_) {
      const double code = std::min(double(kLimit), std::floor(x / scale + 0.5));
      const double residual = x - scale * code;
      result += residual * residual;
    }
    return result;
  }

  std::vector<double> magnitudes_;
  std::array<Boundary, kLimit> heap_;
};

template <int Bits, bool MatrixB>
inline void quantize_matrix(const ColMatrix &matrix, uint8_t *out, float *factors) {
  constexpr int offset = 1 << (Bits - 1);
  constexpr int divisor = 8 / Bits;
  const int rows = matrix.rows() - int(MatrixB);
  ColumnFitter<Bits> fitter;
  for (int col = 0; col < matrix.cols(); ++col) {
    const float *values = matrix.col(col).data() + int(MatrixB);
    const float factor = fitter.fit(values, rows);
    factors[col] = factor;
    uint8_t *packed = out + col * rows / divisor;
    if constexpr (Bits == 8) {
      for (int row = 0; row < rows; ++row)
        packed[row] = uint8_t(fitter.encode(values[row], factor) + offset);
    } else {
      // The A kernel packs within 32-coordinate blocks; B packs across its
      // complete quantized latent dimension (16, 32, or 64 coordinates).
      const int block = MatrixB ? rows : 32;
      for (int start = 0; start < rows; start += block) {
        for (int row = 0; row < block / 2; ++row) {
          const int low = fitter.encode(values[start + row], factor) + offset;
          const int high = fitter.encode(values[start + block / 2 + row], factor) + offset;
          packed[start / 2 + row] = uint8_t(low | (high << 4));
        }
      }
    }
  }
}

}  // namespace joint_quantization
}  // namespace Lorann
