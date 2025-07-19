#ifndef RSVD_CONSTANTS_HPP_
#define RSVD_CONSTANTS_HPP_

namespace Rsvd {

/// \brief Enum for conditioners to be used during subspace iterations
///
/// \warning Modified Gram--Schmidt is a self-implemented algorithm, use it with caution. Use
/// #Rsvd::SubspaceIterationConditioner::Lu or #Rsvd::SubspaceIterationConditioner::Qr if numerical
/// stability and implementation quality is important.
enum class SubspaceIterationConditioner {
  None, ///< No conditioner, fastest, can cause numerical problems
  Lu,   ///< Fully pivoted LU decompostion, fast, acceptable numerical stability
  Mgs,  ///< Modified Gram--Schmidt orthonormalization, slow, better numerical stability
  Qr,   ///< QR decomposition (Householder reflections), slowest, best numerical stability
};

} // namespace Rsvd

#endif
