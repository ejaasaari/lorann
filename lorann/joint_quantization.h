#pragma once

// Joint quantization fits integer codes and scales for the low-rank model A*B,
// then refines them using the error in the reconstructed model product.
//
// First, fit the quantized part of each A/B column v by minimizing
// ||v - s*k||^2, where k contains integer codes and s is the decoding scale.
// SQ4 uses all codes in [-8, 7] and searches both signs of s, allowing the extra
// level to serve either tail. The fitted scale can trade clipping of large
// coefficients for smaller rounding errors in the remaining coefficients.
// For fixed nonzero codes, the optimal scale is s = dot(v,k)/dot(k,k). Sweeping the
// intervals where rounding leaves the codes unchanged finds candidate scales
// analytically; candidates are checked with the stored floating-point factors.
//
// Next, hold the reconstructed A fixed and jointly refine each B column's codes,
// scale, and existing floating-point correction to reduce
// ||A*B - A_hat*B_hat||_F^2, where hats denote the reconstructed factors. This lets
// B compensate for quantization errors in A and accounts for interactions between
// latent coordinates. Up to four bounded coordinate-descent sweeps retain only
// improvements to the product error. This refinement is a local optimization;
// it does not guarantee a global minimum.
//
// The stored factors are 1/s for compatibility with the integer query kernels.
// The protected coefficients remain floating point, and the packed index layout
// is unchanged. All fitting and refinement take place during index construction.

#include <array>
#include <cmath>
#include <limits>

#include "utils.h"

namespace Lorann {
namespace joint_quantization {

// Symmetric column fitter, also used to preserve exact SQ4 grids when tied.
// Sorted magnitudes give one boundary stream per integer level; scratch storage
// is reused across columns. FullRangeFitter below handles the asymmetric SQ4 range.
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
        const double threshold = (tail_energy - best_error) / (tail_sum + std::sqrt(discriminant));
        if (i == 1 || threshold >= magnitudes_[i - 2]) {
          lower = std::nextafter(threshold / kLimit, 0.);
          break;
        }
      }

      size_t heap_size = 0;
      for (int level = 0; level < kLimit; ++level) {
        const double inverse = 1. / (level + 0.5);
        const auto first =
            std::upper_bound(magnitudes_.begin(), magnitudes_.end(), lower,
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
        int code =
            lower > 0 ? int(std::clamp(std::ceil(x / lower - 0.5), 0., double(kLimit))) : kLimit;
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

// Use all sixteen SQ4 codes, choosing which tail receives the extra level.
// Each orientation is solved by sweeping every relevant rounding interval.
class FullRangeFitter {
  static constexpr std::array<double, 8> kBoundaryInverse = {2.,      2. / 3.,  2. / 5.,  2. / 7.,
                                                             2. / 9., 2. / 11., 2. / 13., 2. / 15.};

 public:
  static int encode(float value, float factor) {
    return int(std::clamp(std::floor(double(value) * factor + 0.5), -8., 7.));
  }

  float fit(const float *values, int size) {
    double maximum = 0;
    for (int i = 0; i < size; ++i) {
      if (!std::isfinite(values[i]))
        throw std::invalid_argument("Model quantization requires finite values");
      maximum = std::max(maximum, std::abs(double(values[i])));
    }
    if (maximum == 0) return 1.f;
    auto bound_factor = [](double factor) {
      return float(std::clamp(factor, double(ColumnFitter<4>::kMinFactor),
                              double(ColumnFitter<4>::kMaxFactor)));
    };
    float best = bound_factor(7. / maximum);
    double best_loss = stored_loss(values, size, best);
    if (best_loss == 0) return best;
    // A cheap feasible bound suffices: the two full-range sweeps below already
    // solve the complete problem, including every symmetric candidate.
    float initial = best;
    for (int iteration = 0; iteration < 3; ++iteration) {
      double dot = 0, norm = 0;
      for (int i = 0; i < size; ++i) {
        const int code = encode(values[i], initial);
        dot += double(values[i]) * code;
        norm += code * code;
      }
      if (dot == 0) break;
      const float next = bound_factor(norm / dot);
      if (next == initial) break;
      initial = next;
      const double error = stored_loss(values, size, initial);
      if (error < best_loss) {
        best_loss = error;
        best = initial;
      }
    }
    auto preserve_exact_grid = [&](float factor) {
      // Exact columns can admit several grids. Keep the original symmetric
      // grid when it is also exact, since B refinement depends on its spacing.
      ColumnFitter<4> symmetric;
      const float original = symmetric.fit(values, size);
      return stored_loss(values, size, original) == 0 ? original : factor;
    };
    if (best_loss == 0) return preserve_exact_grid(best);
    auto consider = [&](double ideal) {
      const double bound = ColumnFitter<4>::kMaxFactor;
      const float rounded = float(std::clamp(ideal, -bound, bound));
      for (float factor : {rounded, std::nextafter(rounded, 0.f),
                           std::nextafter(rounded, std::copysign(INFINITY, rounded))}) {
        if (!std::isfinite(factor) || std::abs(factor) < ColumnFitter<4>::kMinFactor ||
            std::abs(factor) > ColumnFitter<4>::kMaxFactor)
          continue;
        const double candidate = stored_loss(values, size, factor);
        if (candidate < best_loss) {
          best_loss = candidate;
          best = factor;
        }
      }
    };
    ordered_.clear();
    double energy = 0;
    for (int i = 0; i < size; ++i) {
      if (values[i] == 0) continue;
      const double x = std::abs(double(values[i])) / maximum;
      ordered_.push_back({x, 0, values[i] < 0 ? 8 : 7});
      energy += x * x;
    }
    std::sort(ordered_.begin(), ordered_.end(),
              [](const Value &a, const Value &b) { return a.x < b.x; });
    for (int sign : {1, -1}) {
      if (sign < 0)
        for (auto &v : ordered_) v.limit = 15 - v.limit;
      values_.clear();
      // Merge the two sign groups in descending saturation order. Their
      // magnitudes are already sorted, so neither orientation needs a re-sort.
      int index7 = int(ordered_.size()), index8 = index7;
      auto advance = [&](int &index, int limit) {
        while (--index >= 0)
          if (ordered_[index].limit == limit) return ordered_[index].x / limit;
        return -1.;
      };
      double next7 = advance(index7, 7), next8 = advance(index8, 8);
      while (index7 >= 0 || index8 >= 0) {
        if (next7 >= next8) {
          values_.push_back({ordered_[index7].x, next7, 7});
          next7 = advance(index7, 7);
        } else {
          values_.push_back({ordered_[index8].x, next8, 8});
          next8 = advance(index8, 8);
        }
      }
      double best_error = best_loss / maximum / maximum;
      double best_scale = 1. / std::abs(double(best)) / maximum;
      // Saturated coordinates bound the admissible decoding scales from below.
      double xx = 0, xk = 0, kk = 0, lower = 0;
      for (size_t i = 0; i < values_.size(); ++i) {
        const auto v = values_[i];
        xx += v.x * v.x;
        xk += v.x * v.limit;
        kk += v.limit * v.limit;
        if (xx <= best_error) continue;
        const double root =
            (xx - best_error) / (xk + std::sqrt(std::max(0., xk * xk - kk * (xx - best_error))));
        if (i + 1 == values_.size() || root >= values_[i + 1].saturation) {
          lower = std::nextafter(root, 0.);
          break;
        }
      }
      double dot = 0, norm = 0, zero_energy = 0;
      for (const auto v : values_) {
        int code = lower > 0 ? int(std::clamp(std::ceil(v.x / lower - 0.5), 0., double(v.limit)))
                             : v.limit;
        while (code > 0 && v.x * kBoundaryInverse[code - 1] <= lower) --code;
        while (code < v.limit && v.x * kBoundaryInverse[code] > lower) ++code;
        dot += v.x * code;
        norm += code * code;
        if (code == 0) zero_energy += v.x * v.x;
      }
      // Seven boundary streams use every magnitude; the eighth uses only
      // coordinates assigned the extra negative code. Merge them lazily rather
      // than allocating and sorting all coefficient/level pairs.
      std::array<Stream, 8> heap;
      size_t heap_size = 0;
      for (int level = 0; level < 8; ++level) {
        const double inverse = kBoundaryInverse[level];
        size_t index = std::upper_bound(ordered_.begin(), ordered_.end(), lower,
                                        [inverse](double bound, const Value &v) {
                                          return bound < v.x * inverse;
                                        }) -
                       ordered_.begin();
        if (level == 7)
          while (index < ordered_.size() && ordered_[index].limit != 8) ++index;
        if (index < ordered_.size())
          heap[heap_size++] = {ordered_[index].x * inverse, index, level};
      }
      std::make_heap(heap.begin(), heap.begin() + heap_size,
                     [](const Stream &a, const Stream &b) { return a.scale > b.scale; });
      bool improved = false;
      for (;;) {
        const double upper = heap_size ? heap[0].scale : INFINITY;
        if (norm > 0 && dot >= lower * norm && dot <= upper * norm &&
            dot * dot / norm >= energy - best_error) {
          const double scale = dot / norm;
          double error = 0;
          for (const auto v : values_) {
            const double code = std::min(double(v.limit), std::floor(v.x / scale + 0.5));
            const double residual = v.x - scale * code;
            error += residual * residual;
          }
          if (error < best_error) {
            best_error = error;
            best_scale = scale;
            improved = true;
          }
        }
        if (heap_size == 0 || zero_energy >= best_error) break;
        Stream next = heap[0];
        const double x = ordered_[next.index].x;
        dot -= x;
        norm -= 2 * next.level + 1;
        lower = next.scale;
        if (next.level == 0) zero_energy += x * x;
        ++next.index;
        if (next.level == 7)
          while (next.index < ordered_.size() && ordered_[next.index].limit != 8) ++next.index;
        if (next.index == ordered_.size()) {
          next = heap[--heap_size];
          if (heap_size == 0) continue;
        } else {
          next.scale = ordered_[next.index].x * kBoundaryInverse[next.level];
        }
        size_t parent = 0, child = 1;
        while (child < heap_size) {
          if (child + 1 < heap_size && heap[child + 1].scale < heap[child].scale) ++child;
          if (next.scale <= heap[child].scale) break;
          heap[parent] = heap[child];
          parent = child;
          child = 2 * parent + 1;
        }
        heap[parent] = next;
      }
      if (improved) consider(sign / (best_scale * maximum));
      // The other orientation can improve the stored codes at the same scale.
      consider(sign * std::abs(double(best)));
    }
    return best_loss == 0 ? preserve_exact_grid(best) : best;
  }

 private:
  struct Value {
    double x, saturation;
    int limit;
  };
  struct Stream {
    double scale;
    size_t index;
    int level;
  };
  static double stored_loss(const float *values, int size, float factor) {
    double sum = 0;
    for (int i = 0; i < size; ++i) {
      const double error = double(values[i]) - encode(values[i], factor) / double(factor);
      sum += error * error;
    }
    return sum;
  }
  std::vector<Value> values_;
  std::vector<Value> ordered_;
};

template <int Bits, bool MatrixB>
inline void quantize_matrix(const ColMatrix &matrix, uint8_t *out, float *factors) {
  constexpr int offset = 1 << (Bits - 1);
  constexpr int divisor = 8 / Bits;
  const int rows = matrix.rows() - int(MatrixB);
  std::conditional_t<Bits == 4, FullRangeFitter, ColumnFitter<Bits>> fitter;
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

// Jointly fit B's codes, decoding scale and protected coefficient to the
// original model product, holding the quantized first-stage model fixed.
inline void refit_correction(const ColMatrix &a, const ColMatrix &b, const ColMatrixUInt8 &qa,
                             ColMatrixUInt8 &qb, const Vector &ca, Vector &cb) {
  Eigen::MatrixXd ah(a.rows(), a.cols());
  for (int j = 0; j < a.cols(); ++j) {
    for (int i = 0; i < a.rows(); ++i) {
      const int byte = (i / 32) * 16 + i % 16;
      ah(i, j) = (((qa(byte, j) >> (i % 32 < 16 ? 0 : 4)) & 15) - 8) / double(ca[j]);
    }
    ah(0, j) = ca[a.cols() + j];
  }
  const int rank = b.rows() - 1;
  const Eigen::MatrixXd gram = ah.transpose() * ah;
  const double norm = gram(0, 0);
  if (!std::isfinite(norm) || norm <= 1e-12 * gram.trace()) return;
  Eigen::MatrixXd original = a.cast<double>();
  original.row(0) = ca.tail(a.cols()).cast<double>();
  const Eigen::MatrixXd target = (ah.transpose() * original) * b.cast<double>();
  const Eigen::VectorXd cross = gram.col(0).tail(rank);
  const Eigen::MatrixXd reduced =
      gram.bottomRightCorner(rank, rank) - cross * cross.transpose() / norm;
  const double diagonal_threshold = 1e-12 * reduced.trace();
  Eigen::VectorXd z(rank), best(rank), rhs(rank), gradient(rank), reduced_z(rank),
      decoded(rank + 1), gram_decoded(rank + 1);
  for (int j = 0; j < b.cols(); ++j) {
    for (int i = 0; i < rank; ++i)
      z[i] = ((qb(i % (rank / 2), j) >> (i < rank / 2 ? 0 : 4)) & 15) - 8;
    rhs = target.col(j).tail(rank) - cross * (target(0, j) / norm);
    const double initial_scale = 1. / cb[j];
    double scale = initial_scale;
    best = z;
    float best_factor = cb[j], best_correction = cb[b.cols() + j];
    auto objective = [&](const Eigen::VectorXd &codes, float factor, float correction) {
      decoded[0] = correction;
      decoded.tail(rank) = codes / double(factor);
      gram_decoded.noalias() = gram * decoded;
      return decoded.dot(gram_decoded) - 2 * decoded.dot(target.col(j));
    };
    double best_loss = objective(z, best_factor, best_correction);
    auto retain = [&]() {
      const float factor = float(1. / scale);
      if (!std::isfinite(factor) || std::abs(factor) < ColumnFitter<4>::kMinFactor ||
          std::abs(factor) > ColumnFitter<4>::kMaxFactor)
        return;
      const float correction = float((target(0, j) - cross.dot(z) / double(factor)) / norm);
      if (!std::isfinite(correction)) return;
      const double error = objective(z, factor, correction);
      if (error < best_loss) {
        best_loss = error;
        best = z;
        best_factor = factor;
        best_correction = correction;
      }
    };
    retain();
    reduced_z.noalias() = reduced * z;
    for (int iteration = 0; iteration < 4; ++iteration) {
      gradient = reduced_z - rhs / scale;
      bool changed = false;
      for (int i = 0; i < rank; ++i) {
        if (reduced(i, i) <= diagonal_threshold) continue;
        const double code =
            std::clamp(std::floor(z[i] - gradient[i] / reduced(i, i) + .5), -8., 7.);
        const double delta = code - z[i];
        if (delta == 0) continue;
        changed = true;
        z[i] = code;
        gradient += delta * reduced.col(i);
      }
      if (changed) retain();
      if (changed) reduced_z.noalias() = reduced * z;
      const double denominator = z.dot(reduced_z);
      if (denominator <= 0) break;
      const double multiplier = std::clamp(z.dot(rhs) / denominator / initial_scale, .5, 2.);
      const double next_scale = initial_scale * multiplier;
      // Once both codes and scale are unchanged, the remaining sweeps would
      // repeat exactly the same updates and stored-candidate evaluations.
      if (next_scale == scale) {
        if (!changed) break;
        continue;
      }
      scale = next_scale;
      retain();
    }
    cb[j] = best_factor;
    cb[b.cols() + j] = best_correction;
    for (int i = 0; i < rank / 2; ++i)
      qb(i, j) = uint8_t(int(best[i]) + 8) | (uint8_t(int(best[i + rank / 2]) + 8) << 4);
  }
}

}  // namespace joint_quantization
}  // namespace Lorann
