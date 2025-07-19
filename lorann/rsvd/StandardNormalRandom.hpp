#ifndef RSVD_STANDARD_NORMAL_RANDOM_HPP_
#define RSVD_STANDARD_NORMAL_RANDOM_HPP_

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <random>

namespace Rsvd {

namespace Internal {

namespace {
/// \brief Shortcut for the underlying \a real type used in the matrix.
///
/// \long E.g. if the matrix is of type \c std::complex<double>, this typedef equals to \c double.
template <typename MatrixType>
using RealType = typename Eigen::NumTraits<typename MatrixType::Scalar>::Real;
} // namespace

/// \brief Helper struct to do partial specialization in the matrix scalar type
///
/// \tparam MatrixType Eigen matrix type.
///
/// \tparam ScalarType Underlying scalar type used in \c MatrixType, can be complex or real.
///
/// \tparam RandomEngineType Type of the random engine, e.g. \c std::default_random_engine or \c
/// std::mt19937_64.
template <typename MatrixType, typename ScalarType, typename RandomEngineType>
struct StandardNormalRandomHelper {
  static inline MatrixType generate(Eigen::Index numRows, Eigen::Index numCols,
                                    RandomEngineType &engine);
};

/// \brief Partial specialization for real matrices.
template <typename MatrixType, typename RandomEngineType>
struct StandardNormalRandomHelper<MatrixType, RealType<MatrixType>, RandomEngineType> {
  static inline MatrixType generate(const Eigen::Index numRows, const Eigen::Index numCols,
                                    RandomEngineType &engine) {
    // Create a standard normal distribution with zero mean (mu = 0) and unity variance (sigma^2 =
    // 1)
    std::normal_distribution<RealType<MatrixType>> distribution{0, 1};
    const auto normal{[&](typename MatrixType::Scalar) { return distribution(engine); }};
    return MatrixType::NullaryExpr(numRows, numCols, normal);
  }
};

/// \brief Partial specialization for complex matrices.
template <typename MatrixType, typename RandomEngineType>
struct StandardNormalRandomHelper<MatrixType, std::complex<RealType<MatrixType>>,
                                  RandomEngineType> {
  static inline MatrixType generate(const Eigen::Index numRows, const Eigen::Index numCols,
                                    RandomEngineType &engine) {
    // A complex standard normal distribution has half of its variance in the real variable and
    // half in the complex. Hence, the each variance is 1/2 and each standard deviation is
    // 1/sqrt(2)
    constexpr RealType<MatrixType> stdDev{0.707106781186547};
    std::normal_distribution<RealType<MatrixType>> distribution{0, stdDev};
    const auto complexNormal{[&](typename MatrixType::Scalar) {
      return std::complex<RealType<MatrixType>>(distribution(engine), distribution(engine));
    }};
    return MatrixType::NullaryExpr(numRows, numCols, complexNormal);
  }
};

/// \brief Generate a matrix with iid elements from standard normal
/// distribution.
///
/// \long The matrix elements are drawn from the standard normal distribution (zero mean and unit
/// variance), i.e. \f$ [ \Omega ]_{i,j} \sim \mathcal{N}(0, 1) \f$ for the real case and \f$ [
/// \Omega ]_{i,j} \sim \mathcal{N}\left(0, \frac{1}{2} \right) + \mathcal{N}\left(0, \frac{1}{2}
/// \right) \cdot \mathrm{i}\f$ for the complex case.
///
/// \tparam MatrixType Eigen matrix type.
/// \tparam RandomEngineType Type of the random engine, e.g. \c std::default_random_engine or \c
/// std::mt19937_64.
///
/// \param numRows Number of matrix rows \f$ m \f$.
/// \param numCols Number of matrix columns \f$ n \f$.
/// \param engine Reference to the random engine.
///
/// \return Matrix \f$ \Omega \in \mathbb{F}^{m \times n} \f$ with \f$\mathbb{F} \in \{ \mathbb{R},
/// \mathbb{C} \} \f$ with normally distributed elements.
template <typename MatrixType, typename RandomEngineType>
inline MatrixType standardNormalRandom(const Eigen::Index numRows, const Eigen::Index numCols,
                                       RandomEngineType &engine) {
  return StandardNormalRandomHelper<MatrixType, typename MatrixType::Scalar,
                                    RandomEngineType>::generate(numRows, numCols, engine);
}

} // namespace Internal

} // namespace Rsvd

#endif
