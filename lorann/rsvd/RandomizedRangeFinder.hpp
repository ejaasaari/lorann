#ifndef RSVD_RANDOMIZED_RANGE_FINDER_HPP_
#define RSVD_RANDOMIZED_RANGE_FINDER_HPP_

#include <Eigen/Dense>
#include <rsvd/Constants.hpp>
#include <rsvd/GramSchmidt.hpp>
#include <rsvd/StandardNormalRandom.hpp>

namespace Rsvd {

namespace Internal {

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

  const auto numRows{a.rows()};
  const auto numCols{a.cols()};

  MatrixType result{a * standardNormalRandom<MatrixType, RandomEngineType>(numCols, dim, engine)};

  Eigen::ColPivHouseholderQR<Eigen::Ref<MatrixType>> qr(result);
  result.noalias() = qr.householderQ() * MatrixType::Identity(numRows, dim);

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

    Eigen::ColPivHouseholderQR<Eigen::Ref<MatrixType>> qr(tmpRows);
    tmpRows.noalias() = qr.householderQ() * MatrixType::Identity(numRows, dim);

    return tmpRows;
  }
};

/// \brief Partial specialization for subspace iterations with the fully pivoted
/// LU decomposition for conditioning.
///
/// \long To improve numerical stability, the temporary matrix \f$M\f$ is decomposed as follows:
/// \f$ M = P^{-1} L U Q^{-1} \f$, where \f$P\f$ and \f$Q\f$ are permutation matrices; \f$L\f$ is a
/// unit lower triangular matrix; \f$U\f$ is an upper triangular matrix.
///
/// After the decomposition, \f$P^{-1} L\f$ is used for further iterations instead of \f$M\f$.
template <typename MatrixType, typename RandomEngineType>
struct RandomizedSubspaceIterations<MatrixType, RandomEngineType,
                                    SubspaceIterationConditioner::Lu> {
  static MatrixType compute(const MatrixType &a, const Eigen::Index dim,
                            const unsigned int numIter, RandomEngineType &engine) {
    assert(numIter > 0);

    const auto numRows{a.rows()};
    const auto numCols{a.cols()};

    MatrixType tmpCols{standardNormalRandom<MatrixType, RandomEngineType>(numCols, dim, engine)};
    MatrixType tmpRows(numRows, dim);

    for (unsigned int j{0U}; j < numIter; ++j) {
      tmpRows.noalias() = a * tmpCols;
      Eigen::FullPivLU<Eigen::Ref<MatrixType>> luRows(tmpRows);
      tmpRows.diagonal().setOnes();
      tmpRows.template triangularView<Eigen::StrictlyUpper>().setZero();

      tmpCols.noalias() = a.adjoint() * luRows.permutationP().inverse() * tmpRows;
      Eigen::FullPivLU<Eigen::Ref<MatrixType>> luCols(tmpCols);
      tmpCols.diagonal().setOnes();
      tmpCols.template triangularView<Eigen::StrictlyUpper>().setZero();

      /// \todo Can we avoid intermediate allocation here?
      tmpCols = luCols.permutationP().inverse() * tmpCols;
    }

    tmpRows.noalias() = a * tmpCols;
    Eigen::ColPivHouseholderQR<Eigen::Ref<MatrixType>> qr(tmpRows);
    tmpRows.noalias() = qr.householderQ() * MatrixType::Identity(numRows, dim);

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
    Eigen::ColPivHouseholderQR<Eigen::Ref<MatrixType>> qr(tmpRows);
    tmpRows.noalias() = qr.householderQ() * MatrixType::Identity(numRows, dim);

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
    Eigen::ColPivHouseholderQR<Eigen::Ref<MatrixType>> qr(tmpRows);
    tmpRows.noalias() = qr.householderQ() * MatrixType::Identity(numRows, dim);

    return tmpRows;
  }
};

} // namespace Internal

} // namespace Rsvd

#endif
