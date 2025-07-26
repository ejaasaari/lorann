#pragma once

#include <Eigen/Core>
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

namespace cereal {

template <class Archive, class Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_,
          int MaxCols_>
inline typename std::enable_if<traits::is_output_serializable<BinaryData<Scalar_>, Archive>::value,
                               void>::type
save(Archive &ar,
     const Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_> &matrix) {
  const std::int32_t rows = static_cast<std::int32_t>(matrix.rows());
  const std::int32_t cols = static_cast<std::int32_t>(matrix.cols());
  ar(rows);
  ar(cols);
  ar(binary_data(matrix.data(), rows * cols * sizeof(Scalar_)));
};

template <class Archive, class Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_,
          int MaxCols_>
inline typename std::enable_if<traits::is_input_serializable<BinaryData<Scalar_>, Archive>::value,
                               void>::type
load(Archive &ar, Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_> &matrix) {
  std::int32_t rows;
  std::int32_t cols;
  ar(rows);
  ar(cols);

  matrix.resize(rows, cols);
  ar(binary_data(matrix.data(), static_cast<std::size_t>(rows * cols * sizeof(Scalar_))));
};

} /* namespace cereal */
