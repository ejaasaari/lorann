#ifndef RSVD_RANDOMIZED_RANGE_FINDER_HPP_
#define RSVD_RANDOMIZED_RANGE_FINDER_HPP_

#include <Eigen/Dense>
#include <cassert>
#include <rsvd/Constants.hpp>
#include <rsvd/GramSchmidt.hpp>
#include <rsvd/StandardNormalRandom.hpp>
#include <vector>

namespace Rsvd {

namespace Internal {

/// \brief Orthonormalize a tall matrix with CholeskyQR2 and a rank-deficient QR fallback.
template <typename MatrixType> void orthonormalize(MatrixType &a) {
  for (unsigned int pass{0U}; pass < 2U; ++pass) {
    const MatrixType gram{a.adjoint() * a};
    Eigen::LLT<MatrixType> llt(gram);
    if (llt.info() != Eigen::Success) {
      Eigen::ColPivHouseholderQR<Eigen::Ref<MatrixType>> qr(a);
      a.noalias() = qr.householderQ() * MatrixType::Identity(a.rows(), a.cols());
      return;
    }

    const MatrixType rTranspose{llt.matrixU().transpose()};
    auto aTranspose{a.transpose()};
    rTranspose.template triangularView<Eigen::Lower>().solveInPlace(aTranspose);
  }
}

/// \brief Reusable packed left operand for repeated matrix products.
template <typename MatrixType, bool IsRowMajor = MatrixType::IsRowMajor>
class ReusableLeftProduct;

template <typename MatrixType> class ReusableLeftProduct<MatrixType, false> {
public:
  using Scalar = typename MatrixType::Scalar;
  using Traits = Eigen::internal::gebp_traits<Scalar, Scalar>;
  using LhsMapper =
      Eigen::internal::const_blas_data_mapper<Scalar, Eigen::Index, Eigen::ColMajor>;
  using RhsMapper =
      Eigen::internal::const_blas_data_mapper<Scalar, Eigen::Index, Eigen::ColMajor>;
  using ResultMapper = Eigen::internal::blas_data_mapper<
      Scalar, Eigen::Index, Eigen::ColMajor, Eigen::Unaligned, 1>;
  using PackLhs = Eigen::internal::gemm_pack_lhs<
      Scalar, Eigen::Index, LhsMapper, Traits::mr, Traits::LhsProgress,
      typename Traits::LhsPacket4Packing, Eigen::ColMajor>;
  using PackRhs = Eigen::internal::gemm_pack_rhs<
      Scalar, Eigen::Index, RhsMapper, Traits::nr, Eigen::ColMajor>;
  using Kernel = Eigen::internal::gebp_kernel<
      Scalar, Scalar, Eigen::Index, ResultMapper, Traits::mr, Traits::nr, false, false>;
  using PackedVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  explicit ReusableLeftProduct(const MatrixType &a)
      : m_numRows{a.rows()}, m_depth{a.cols()}, m_packedLhs(m_numRows * m_depth) {
    const LhsMapper mapper(a.data(), a.outerStride());
    PackLhs{}(m_packedLhs.data(), mapper, m_depth, m_numRows);
  }

  void multiply(const MatrixType &rhs, MatrixType &result) {
    const Eigen::Index numCols{rhs.cols()};
    m_packedRhs.resize(m_depth * numCols);
    const RhsMapper rhsMapper(rhs.data(), rhs.outerStride());
    PackRhs{}(m_packedRhs.data(), rhsMapper, m_depth, numCols);

    result.setZero(m_numRows, numCols);
    const ResultMapper resultMapper(result.data(), result.outerStride(), result.innerStride());
    Kernel{}(resultMapper, m_packedLhs.data(), m_packedRhs.data(), m_numRows, m_depth,
             numCols, Scalar{1});
  }

private:
  Eigen::Index m_numRows;
  Eigen::Index m_depth;
  PackedVector m_packedLhs;
  PackedVector m_packedRhs;
};

template <typename MatrixType> class ReusableLeftProduct<MatrixType, true> {
public:
  explicit ReusableLeftProduct(const MatrixType &a) : m_a{a} {}

  void multiply(const MatrixType &rhs, MatrixType &result) const {
    result.noalias() = m_a * rhs;
  }

private:
  const MatrixType &m_a;
};

/// \brief Reusable packed adjoint operand for repeated matrix products.
template <typename MatrixType, bool IsRowMajor = MatrixType::IsRowMajor>
class ReusableAdjointProduct;

template <typename MatrixType> class ReusableAdjointProduct<MatrixType, false> {
public:
  using Scalar = typename MatrixType::Scalar;
  using Traits = Eigen::internal::gebp_traits<Scalar, Scalar>;
  using LhsMapper =
      Eigen::internal::const_blas_data_mapper<Scalar, Eigen::Index, Eigen::RowMajor>;
  using RhsMapper =
      Eigen::internal::const_blas_data_mapper<Scalar, Eigen::Index, Eigen::ColMajor>;
  using ResultMapper = Eigen::internal::blas_data_mapper<
      Scalar, Eigen::Index, Eigen::ColMajor, Eigen::Unaligned, 1>;
  using PackLhs = Eigen::internal::gemm_pack_lhs<
      Scalar, Eigen::Index, LhsMapper, Traits::mr, Traits::LhsProgress,
      typename Traits::LhsPacket4Packing, Eigen::RowMajor>;
  using PackRhs = Eigen::internal::gemm_pack_rhs<
      Scalar, Eigen::Index, RhsMapper, Traits::nr, Eigen::ColMajor>;
  using Kernel = Eigen::internal::gebp_kernel<
      Scalar, Scalar, Eigen::Index, ResultMapper, Traits::mr, Traits::nr, true, false>;
  using PackedVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  explicit ReusableAdjointProduct(const MatrixType &a)
      : m_numRows{a.cols()}, m_depth{a.rows()}, m_packedLhs(m_numRows * m_depth) {
    const LhsMapper mapper(a.data(), a.outerStride());
    PackLhs{}(m_packedLhs.data(), mapper, m_depth, m_numRows);
  }

  void multiply(const MatrixType &rhs, MatrixType &result) {
    const Eigen::Index numCols{rhs.cols()};
    m_packedRhs.resize(m_depth * numCols);
    const RhsMapper rhsMapper(rhs.data(), rhs.outerStride());
    PackRhs{}(m_packedRhs.data(), rhsMapper, m_depth, numCols);

    result.setZero(m_numRows, numCols);
    const ResultMapper resultMapper(result.data(), result.outerStride(), result.innerStride());
    Kernel{}(resultMapper, m_packedLhs.data(), m_packedRhs.data(), m_numRows, m_depth,
             numCols, Scalar{1});
  }

private:
  Eigen::Index m_numRows;
  Eigen::Index m_depth;
  PackedVector m_packedLhs;
  PackedVector m_packedRhs;
};

template <typename MatrixType> class ReusableAdjointProduct<MatrixType, true> {
public:
  explicit ReusableAdjointProduct(const MatrixType &a) : m_a{a} {}

  void multiply(const MatrixType &rhs, MatrixType &result) const {
    result.noalias() = m_a.adjoint() * rhs;
  }

private:
  const MatrixType &m_a;
};

/// \brief Single-shot randomized range approximation.
///
/// \long This function implements Algorithm 4.3. It computes an approximate range of the given
/// matrix \f$ A \f$ by a single random sampling.
///
/// \tparam MatrixType Eigen matrix type.
/// \tparam RandomEngineType Type of the random engine, e.g. \c std::default_random_engine or \c
/// std::mt19937_64.
///
/// \param a Matrix \f$A \in \mathbb{F}^{m \times n}\f$ whose range should be approximated.
/// \param dim Dimension \f$r\f$ (number of columns) of the range approximation.
/// \param engine Random engine to use for sampling from standard normal distribution.
///
/// \return Matrix \f$ Q \in \mathbb{F}^{m \times r} \f$ whose columns build an orthonormal basis
/// of the approximate range of \f$ A \f$.
template <typename MatrixType, typename RandomEngineType>
MatrixType singleShot(const MatrixType &a, const Eigen::Index dim, RandomEngineType &engine) {

  const auto numCols{a.cols()};

  MatrixType result{a * standardNormalRandom<MatrixType, RandomEngineType>(numCols, dim, engine)};

  orthonormalize(result);

  return result;
}

/// \brief Helper struct for randomized subspace iterations.
///
/// \tparam MatrixType Eigen matrix type.
/// \tparam RandomEngineType Type of the random engine, e.g. \c std::default_random_engine or \c
/// std::mt19937_64.
/// \tparam Conditioner Which conditioner to use for subspace iterations, see
/// #SubspaceIterationConditioner.
template <typename MatrixType, typename RandomEngineType, SubspaceIterationConditioner Conditioner>
struct RandomizedSubspaceIterations {
  /// \brief Randomized subspace iterations for range approximation.
  ///
  /// \long Compute an approximate range of the given matrix \f$ A \f$.
  ///
  /// \param a Matrix \f$A \in \mathbb{F}^{m \times n}\f$ whose range should be approximated.
  /// \param dim Dimension \f$r\f$ (number of columns) of the range approximation.
  /// \param numIter Number of subspace iterations.
  /// \param engine Random engine to use for sampling from standard normal distribution.
  ///
  /// \return Matrix \f$ Q \in \mathbb{F}^{m \times r} \f$ whose columns are an orthonormal basis
  /// of the approximate range of \f$ A \f$.
  static MatrixType compute(const MatrixType &a, Eigen::Index dim, unsigned int numIter,
                            RandomEngineType &engine);
};

/// \brief Replace a tall matrix with the row-unpermuted unit lower factor from a partial-pivot LU.
template <typename MatrixType> void partialPivotLuCondition(MatrixType &a) {
  assert(a.cols() <= a.rows());

  const Eigen::Index numRows{a.rows()};
  const Eigen::Index numCols{a.cols()};
  std::vector<Eigen::Index> pivots(static_cast<std::size_t>(numCols));

  for (Eigen::Index j{0}; j < numCols; ++j) {
    Eigen::Index relativePivot{0};
    const auto pivotMagnitude{
        a.col(j).tail(numRows - j).cwiseAbs().maxCoeff(&relativePivot)};
    const Eigen::Index pivot{j + relativePivot};
    pivots[static_cast<std::size_t>(j)] = pivot;
    if (pivot != j) {
      a.row(j).swap(a.row(pivot));
    }

    if (pivotMagnitude != 0 && j + 1 < numRows) {
      a.col(j).tail(numRows - j - 1) /= a(j, j);
      if (j + 1 < numCols) {
        a.bottomRightCorner(numRows - j - 1, numCols - j - 1).noalias() -=
            a.col(j).tail(numRows - j - 1) * a.row(j).tail(numCols - j - 1);
      }
    }
  }

  a.diagonal().setOnes();
  a.template triangularView<Eigen::StrictlyUpper>().setZero();
  for (Eigen::Index j{numCols}; j-- > 0;) {
    const Eigen::Index pivot{pivots[static_cast<std::size_t>(j)]};
    if (pivot != j) {
      a.row(j).swap(a.row(pivot));
    }
  }
}

/// \brief Partial specialization for subspace iterations without a conditioner.
template <typename MatrixType, typename RandomEngineType>
struct RandomizedSubspaceIterations<MatrixType, RandomEngineType,
                                    SubspaceIterationConditioner::None> {
  static MatrixType compute(const MatrixType &a, const Eigen::Index dim,
                            const unsigned int numIter, RandomEngineType &engine) {
    assert(numIter > 0);

    const auto numRows{a.rows()};
    const auto numCols{a.cols()};

    MatrixType tmpRows{a *
                       standardNormalRandom<MatrixType, RandomEngineType>(numCols, dim, engine)};
    MatrixType tmpCols(numCols, dim);

    for (unsigned int j{0U}; j < numIter; ++j) {
      tmpCols.noalias() = a.adjoint() * tmpRows;
      tmpRows.noalias() = a * tmpCols;
    }

    orthonormalize(tmpRows);

    return tmpRows;
  }
};

/// \brief Partial specialization for subspace iterations with the partially pivoted
/// LU decomposition for conditioning.
///
/// \long To improve numerical stability, the temporary matrix \f$M\f$ is decomposed as follows:
/// \f$ M = P^{-1} L U \f$, where \f$P\f$ is a row permutation matrix; \f$L\f$ is a unit lower
/// triangular matrix; \f$U\f$ is an upper triangular matrix.
///
/// After the decomposition, \f$P^{-1} L\f$ is used for further iterations instead of \f$M\f$.
template <typename MatrixType, typename RandomEngineType>
struct RandomizedSubspaceIterations<MatrixType, RandomEngineType,
                                    SubspaceIterationConditioner::Lu> {
  static MatrixType computeRightBasis(const MatrixType &a, const Eigen::Index dim,
                                      const unsigned int numIter, RandomEngineType &engine,
                                      MatrixType &image) {
    assert(numIter > 0);

    const auto numRows{a.rows()};
    const auto numCols{a.cols()};

    MatrixType tmpCols{standardNormalRandom<MatrixType, RandomEngineType>(numCols, dim, engine)};
    MatrixType tmpRows(numRows, dim);
    ReusableLeftProduct<MatrixType> forwardProduct(a);
    ReusableAdjointProduct<MatrixType> adjointProduct(a);

    for (unsigned int j{0U}; j < numIter; ++j) {
      forwardProduct.multiply(tmpCols, tmpRows);
      partialPivotLuCondition(tmpRows);

      adjointProduct.multiply(tmpRows, tmpCols);
      if (j + 1U < numIter) {
        partialPivotLuCondition(tmpCols);
      }
    }

    orthonormalize(tmpCols);
    forwardProduct.multiply(tmpCols, image);
    return tmpCols;
  }

  static MatrixType compute(const MatrixType &a, const Eigen::Index dim,
                            const unsigned int numIter, RandomEngineType &engine) {
    assert(numIter > 0);

    const auto numRows{a.rows()};
    const auto numCols{a.cols()};

    MatrixType tmpCols{standardNormalRandom<MatrixType, RandomEngineType>(numCols, dim, engine)};
    MatrixType tmpRows(numRows, dim);
    ReusableLeftProduct<MatrixType> forwardProduct(a);
    ReusableAdjointProduct<MatrixType> adjointProduct(a);

    for (unsigned int j{0U}; j < numIter; ++j) {
      forwardProduct.multiply(tmpCols, tmpRows);
      partialPivotLuCondition(tmpRows);

      adjointProduct.multiply(tmpRows, tmpCols);
      partialPivotLuCondition(tmpCols);
    }

    forwardProduct.multiply(tmpCols, tmpRows);
    orthonormalize(tmpRows);

    return tmpRows;
  }
};

/// \brief Partial specialization for subspace iterations with the modified Gram--Schmidt process
/// for conditioning.
template <typename MatrixType, typename RandomEngineType>
struct RandomizedSubspaceIterations<MatrixType, RandomEngineType,
                                    SubspaceIterationConditioner::Mgs> {
  static MatrixType compute(const MatrixType &a, const Eigen::Index dim,
                            const unsigned int numIter, RandomEngineType &engine) {
    assert(numIter > 0);

    const auto numRows{a.rows()};
    const auto numCols{a.cols()};

    MatrixType tmpCols{standardNormalRandom<MatrixType, RandomEngineType>(numCols, dim, engine)};
    MatrixType tmpRows(numRows, dim);

    for (unsigned int j{0U}; j < numIter; ++j) {
      tmpRows.noalias() = a * tmpCols;
      modifiedGramSchmidt(tmpRows);

      tmpCols.noalias() = a.adjoint() * tmpRows;
      modifiedGramSchmidt(tmpCols);
    }

    tmpRows.noalias() = a * tmpCols;
    orthonormalize(tmpRows);

    return tmpRows;
  }
};

/// \brief Partial specialization for subspace iterations with the QR decomposition for
/// conditioning.
template <typename MatrixType, typename RandomEngineType>
struct RandomizedSubspaceIterations<MatrixType, RandomEngineType,
                                    SubspaceIterationConditioner::Qr> {
  static MatrixType compute(const MatrixType &a, const Eigen::Index dim,
                            const unsigned int numIter, RandomEngineType &engine) {
    assert(numIter > 0);

    const auto numRows{a.rows()};
    const auto numCols{a.cols()};

    MatrixType tmpCols{standardNormalRandom<MatrixType, RandomEngineType>(numCols, dim, engine)};
    MatrixType tmpRows(numRows, dim);

    for (unsigned int j{0U}; j < numIter; ++j) {
      tmpRows.noalias() = a * tmpCols;
      Eigen::ColPivHouseholderQR<Eigen::Ref<MatrixType>> qrRows(tmpRows);

      tmpCols.noalias() = a.adjoint() * qrRows.householderQ();
      Eigen::ColPivHouseholderQR<Eigen::Ref<MatrixType>> qrCols(tmpCols);
      tmpCols.noalias() = qrCols.householderQ() * MatrixType::Identity(numCols, dim);
    }

    tmpRows.noalias() = a * tmpCols;
    orthonormalize(tmpRows);

    return tmpRows;
  }
};

} // namespace Internal

} // namespace Rsvd

#endif
