#ifndef RSVD_GRAMSCHMIDT_HPP_
#define RSVD_GRAMSCHMIDT_HPP_

#include <Eigen/Dense>
#include <algorithm>
#include <limits>

namespace Rsvd {

namespace Internal {

/// \brief Orthonormalize matrix columns inplace. Deflate if needed.
///
/// \warning This is a self-implemented algorithm, use it with caution. Use
/// #Rsvd::SubspaceIterationConditioner::Lu or #Rsvd::SubspaceIterationConditioner::Qr if numerical
/// stability and implementation quality is important.
///
/// \long This function implements the modified Gram--Schmidt process with deflation.
///
/// The deflation is implemented as follows: Let \f$i\f$ be the index of the current matrix column.
/// Let \f$n_i\f$ be its norm after projection onto a subspace orthogonal to the subspace spanned
/// by the previous columns \f$1, \ldots, n - 1\f$. Obviously, if \f$n_i\f$ is equal to zero, then
/// the \f$i\f$-th column is a linear combination of previous columns. Due to the limited precision
/// of the floating-point computations, \f$n_i\f$ will be small but greater than zero. Hence, we
/// use the following heuristic rule: If \f[ n_i < \max\{n_1, \ldots, n_{i - 1}\} \cdot
/// \varepsilon_{\mathrm{mach}} \cdot 100, \f] then the \f$i\f$-th column is deemed to be linearly
/// dependent and is filled with zeros.
///
/// Here, \f$ \varepsilon_{\mathrm{mach}} \f$ is the machine epsilon.
///
/// \note Although this function deflates if needed, it asserts that the matrix has fewer columns
/// than rows. This might help during debugging. If compiled without assertions, this function will
/// silently deflate on column rank loss.
///
/// \tparam MatrixType Eigen matrix type.
///
/// \param a Matrix whose columns should be orthonormalized inplace. The matrix can be over a real
/// or complex field.
template <typename MatrixType> void modifiedGramSchmidt(MatrixType &a) {
  using RealType = typename Eigen::NumTraits<typename MatrixType::Scalar>::Real;

  RealType largestNormSeen{0};
  // 100 is just an educated guess...
  const RealType tol{100 * std::numeric_limits<RealType>::epsilon()};

  // If a matrix has fewer rows than columns then the columns are linearly dependent
  assert(a.cols() <= a.rows());

  Eigen::Index currCol;
  for (currCol = 0; currCol < a.cols(); ++currCol) {
    for (Eigen::Index prevCol{0}; prevCol < currCol; ++prevCol) {
      /// \note Implementation detail: The order in the dot product is important for vectors over
      /// complex fields!
      a.col(currCol) -= a.col(prevCol).dot(a.col(currCol)) * a.col(prevCol);
    }

    // If the current column has near zero norm, it is a linear combination of previous columns
    const auto currColNorm{a.col(currCol).norm()};
    if (currColNorm < tol * largestNormSeen) {
      // Deflate
      a.col(currCol).setZero();
    } else {
      // Normalize
      a.col(currCol) /= currColNorm;
      largestNormSeen = std::max(largestNormSeen, currColNorm);
    }
  }
}

} // namespace Internal

} // namespace Rsvd

#endif
