#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "lorann.h"

typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXuRm;
typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, 1, Eigen::RowMajor> VectorXuRm;

static uint32_t read_uint32(std::ifstream &ifs) {
  uint32_t value = 0;
  for (int i = 0; i < 4; ++i) {
    uint8_t byte;
    if (!ifs.read(reinterpret_cast<char *>(&byte), 1))
      throw std::runtime_error("Unexpected EOF while reading header");
    value = (value << 8) | byte;
  }
  return value;
}

static MatrixXuRm load_mnist_images(const std::string &file) {
  std::ifstream ifs(file, std::ios::binary);
  if (!ifs) throw std::runtime_error("Cannot open " + file);

  if (read_uint32(ifs) != 0x00000803) throw std::runtime_error("Invalid magic in " + file);

  const uint32_t n_imgs = read_uint32(ifs);
  const uint32_t n_rows = read_uint32(ifs);
  const uint32_t n_cols = read_uint32(ifs);
  const uint32_t img_sz = n_rows * n_cols;

  MatrixXuRm images(n_imgs, img_sz);
  std::vector<uint8_t> buffer(img_sz);

  for (uint32_t i = 0; i < n_imgs; ++i) {
    if (!ifs.read(reinterpret_cast<char *>(buffer.data()), img_sz))
      throw std::runtime_error("Unexpected EOF in " + file);
    std::memcpy(images.row(i).data(), buffer.data(), img_sz);
  }
  return images;
}

int main() {
  MatrixXuRm X = load_mnist_images("mnist/train-images-idx3-ubyte");
  MatrixXuRm Q = load_mnist_images("mnist/t10k-images-idx3-ubyte");

  const int n_clusters = 256;
  const int global_dim = 128;
  const int rank = 32;
  const int train_size = 5;
  const Lorann::Distance distance = Lorann::L2;

  const int clusters_to_search = 8;
  const int points_to_rerank = 40;

  const int k = 10;

  std::cout << "Building the index..." << std::endl;
  std::unique_ptr<Lorann::LorannBase<uint8_t>> index =
      std::make_unique<Lorann::Lorann<uint8_t, Lorann::SQ4Quantizer>>(
          X.data(), X.rows(), X.cols(), n_clusters, global_dim, rank, train_size, distance);
  index->build();

  // Alternatively, load an index from a file:
  // std::unique_ptr<Lorann::LorannBase<uint8_t>> index;
  // std::ifstream input_file("index.bin", std::ios::binary);
  // cereal::BinaryInputArchive input_archive(input_file);
  // input_archive(index);

  Eigen::VectorXi indices(k), indices_exact(k);

  std::cout << "Querying the index using exact search..." << std::endl;
  index->exact_search(Q.row(0).data(), k, indices_exact.data());
  std::cout << indices_exact.transpose() << std::endl;

  std::cout << "Querying the index using approximate search..." << std::endl;
  index->search(Q.row(0).data(), k, clusters_to_search, points_to_rerank, indices.data());
  std::cout << indices.transpose() << std::endl;

  std::cout << "Saving the index to disk..." << std::endl;
  std::ofstream output_file("index.bin", std::ios::binary);
  cereal::BinaryOutputArchive output_archive(output_file);
  output_archive(index);

  return 0;
}
