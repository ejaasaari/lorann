#ifndef RSVD_RANDOMIZED_SVD_HPP_
#define RSVD_RANDOMIZED_SVD_HPP_

// BSD 3-Clause License
//
// Copyright (c) 2018, Mikhail Pak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <Eigen/Dense>
#include <algorithm>
#include <rsvd/Constants.hpp>
#include <rsvd/RandomizedRangeFinder.hpp>

namespace Rsvd {

/// \brief Randomized singular value decomposition.
///
/// \long Let \f$\mathbb{F}\f$ be a field of complex or real numbers, \f$\mathbb{F}=\mathbb{C}\f$
/// or \f$\mathbb{F}=\mathbb{R}\f$.
///
/// Let \f$ A \in \mathbb{F}^{m \times n}\f$, \f$\mathrm{rank}(A) = r\f$. Its economic singular
/// value decomposition (SVD) is given by \f[ A = U \Sigma V^* \f] with \f$ U \in \mathbb{F}^{m
/// \times r} \f$, \f$ \Sigma \in \mathbb{F}^{r \times r} \f$, and \f$ V \in \mathbb{F}^{n \times
/// r} \f$. The left and right singular vector matrices have orthonormal columns: \f$ U^* U = I_{r
/// \times r} \f$, \f$ V^* V = I_{r \times r} \f$. The singular values matrix \f$\Sigma\f$ is
/// diagonal and has sorted singular values on its principal diagonal.
///
/// When \f$m\f$ and \f$n\f$ are very large, we want to approximate this SVD using the randomized
/// range approximation \f$Q \in \mathbb{F}^{m \times r}\f$ s.t. \f$ \| A - Q Q^* A \|_2 <
/// \varepsilon \f$. Then, the problem can be projected onto a smaller subspace as follows: \f$ B =
/// Q^* A\f$, \f$ B \in \mathbb{F}^{r \times n}\f$. After the SVD of the smaller problem \f$ B =
/// \tilde{U} \Sigma V^* \f$, the solution to the original problem can be recovered as \f$ U = Q
/// \tilde{U} \f$.
///
/// The range \f$Q\f$ is approximated using random sampling. In order to capture the largest
/// singular values, randomized subspace iterations can be used. However, as any power iteration
/// methods, they suffer from numerical problems. To mitigate this problem, the user can select an
/// appropriate conditioner based on the modified Gram--Schmidt process, the LU decomposition, and
/// the QR decomposition. The conditioner choice is a trade-off between runtime and numerical
/// properties, see #SubspaceIterationConditioner.
///
/// Usage example:
/// ```cpp
/// const MatrixXd a = ... ;
///
/// std::mt19937_64 randomEngine{};
/// randomEngine.seed(777);
///
/// Rsvd::RandomizedSvd<MatrixXd, std::mt19937_64, Rsvd::SubspaceIterationConditioner::Lu>
/// rsvd(randomEngine);
///
/// rsvd.compute(a, reducedRank);
///
/// // Recover matrix a
/// const MatrixXd rsvdApprox =
///     rsvd.matrixU() * rsvd.singularValues().asDiagonal() *
///     rsvd.matrixV().adjoint();
/// ```
///
/// \tparam MatrixType Eigen matrix type.
/// \tparam RandomEngineType Type of the random engine, e.g. \c std::default_random_engine or \c
/// std::mt19937_64.
/// \tparam Conditioner Which conditioner to use for subspace iterations, see
/// #Rsvd::SubspaceIterationConditioner.
template <typename MatrixType, typename RandomEngineType, SubspaceIterationConditioner Conditioner>
class RandomizedSvd {
public:
  /// \brief Create an object for the randomized singular value decomposition.
  ///
  /// \param engine Random engine to use for sampling from standard normal distribution.
  explicit RandomizedSvd(RandomEngineType &engine) : m_randomEngine{engine} {};

  /// \brief Return the vector of singular values.
  MatrixType singularValues() const { return m_singularValues; }
  /// \brief Return the matrix with the left singular vectors.
  MatrixType matrixU() const { return m_leftSingularVectors; }
  /// \brief Return the matrix with the right singular vectors.
  MatrixType matrixV() const { return m_rightSingularVectors; }

  /// \brief Compute the randomized singular value decomposition.
  ///
  /// \param a Matrix \f$A\f$ to be decomposed, \f$A = U \Sigma V^*\f$.
  /// \param rank Rank of the decomposition.
  /// \param oversamples Number of additionally sampled range directions. Increase it to improve
  /// the approximation. Default value: 5.
  /// \param numIter Number of randomized subspace iterations. Increase it to improve
  /// the approximation. Default value: 2.
  void compute(const MatrixType &a, const Eigen::Index rank, const Eigen::Index oversamples = 5,
               const unsigned int numIter = 2U) {
    /// \todo Handle matrices with m < n correctly.

    const Eigen::Index matrixShortSize{std::min(a.rows(), a.cols())};
    const Eigen::Index rangeApproximationDim{std::min(matrixShortSize, rank + oversamples)};

    const MatrixType q{
        (numIter == 0U)
            ? Internal::singleShot<MatrixType, RandomEngineType>(a, rangeApproximationDim,
                                                                 m_randomEngine)
            : Internal::RandomizedSubspaceIterations<
                  MatrixType, RandomEngineType, Conditioner>::compute(a, rangeApproximationDim,
                                                                      numIter, m_randomEngine)};

    const auto b{q.adjoint() * a};
    Eigen::JacobiSVD<MatrixType> svd(b, Eigen::ComputeThinU | Eigen::ComputeThinV);

    m_leftSingularVectors.noalias() = q * svd.matrixU().leftCols(rank);
    m_singularValues = svd.singularValues().head(rank);
    m_rightSingularVectors = svd.matrixV().leftCols(rank);
  }

private:
  RandomEngineType &m_randomEngine;
  MatrixType m_leftSingularVectors{};
  MatrixType m_singularValues{};
  MatrixType m_rightSingularVectors{};
};

} // namespace Rsvd

#endif
