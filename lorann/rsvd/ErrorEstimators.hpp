#ifndef RSVD_ERROR_ESTIMATORS_HPP_
#define RSVD_ERROR_ESTIMATORS_HPP_

#include <Eigen/Dense>

namespace Rsvd {

/// \brief Compute the relative Frobenius error of a matrix approximation.
///
/// \long The formula for the relative error is
/// \f[ \varepsilon = \frac{ \| \tilde{A} - A \|_{\mathrm{F}} }{ \| A \|_{\mathrm{F}} }. \f]
///
/// Both matrices can be real or complex.
///
/// \tparam MatrixType Eigen matrix type.
///
/// \param reference Reference matrix \f$ A \f$. It is assumed that this matrix has positive
/// Frobenius norm.
///
/// \param approximation Approximation \f$ \tilde{A} \f$ of the reference matrix.
///
/// \return Relative Frobenius error \f$ \varepsilon \f$ of the approximation (non-negative real).
template <typename MatrixType>
typename Eigen::NumTraits<typename MatrixType::Scalar>::Real
relativeFrobeniusNormError(const MatrixType &reference, const MatrixType &approximation) {
  const auto differenceNorm{(approximation - reference).norm()};
  const auto referenceNorm{reference.norm()};

  assert(referenceNorm > 0);

  return differenceNorm / referenceNorm;
}

} // namespace Rsvd

#endif
