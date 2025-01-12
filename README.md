<h1 align="center">LoRANN</h1>
<div align="center">
Approximate nearest neighbor search library implementing <a href="https://arxiv.org/abs/2410.18926">reduced-rank regression</a> (NeurIPS '24), a technique enabling extremely fast queries, tiny memory usage, and rapid indexing on modern embedding datasets.
</div>
<br/>

<div align="center">
    <a href="https://github.com/ejaasaari/lorann/actions/workflows/build.yml"><img src="https://github.com/ejaasaari/lorann/actions/workflows/build.yml/badge.svg" alt="Build status" /></a>
    <a href="https://arxiv.org/abs/2410.18926"><img src="https://img.shields.io/badge/Paper-NeurIPS%3A_LoRANN-salmon" alt="Paper" /></a>
    <a href="https://ejaasaari.github.io/lorann"><img src="https://img.shields.io/badge/api-reference-blue.svg" alt="Documentation" /></a>
    <a href="https://pypi.org/project/lorann/"><img src="https://img.shields.io/pypi/v/lorann?color=blue" alt="PyPI" /></a>
    <a href="https://github.com/ejaasaari/lorann/blob/master/LICENSE"><img src="https://img.shields.io/github/license/ejaasaari/lorann" alt="License" /></a>
    <a href="https://github.com/ejaasaari/lorann/stargazers"><img src="https://img.shields.io/github/stars/ejaasaari/lorann" alt="GitHub stars" /></a>
</div>

---

- Lightweight header-only C++17 library with Python bindings
- Query speed matching state-of-the-art graph methods but with tiny memory usage
- Optimized for modern high-dimensional (d > 100) embedding data sets
- Optimized for modern CPU architectures with acceleration for AVX2, AVX-512, and ARM NEON
- State-of-the-art query speed for GPU batch queries (experimental)
- Supported distances: (negative) inner product, Euclidean distance, cosine distance
- Support for index serialization

## Getting started

### Python

Install the module with `pip install lorann`

> [!TIP]
> On macOS, it is highly recommended to use the Homebrew version of Clang as the compiler:

```shell script
brew install llvm libomp
CC=/opt/homebrew/opt/llvm/bin/clang CXX=/opt/homebrew/opt/llvm/bin/clang++ LDFLAGS=-L/opt/homebrew/opt/llvm/lib pip install lorann
```

A minimal example for indexing and querying a dataset using LoRANN is provided below:

```python
import lorann
import numpy as np
from sklearn.datasets import fetch_openml  # scikit-learn is used only for loading the data

X, _ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = np.ascontiguousarray(X, dtype=np.float32)

data = X[:60_000]
index = lorann.LorannIndex(
    data=data,
    n_clusters=256,
    global_dim=128,
    quantization_bits=8,
    euclidean=True,
)

index.build()

k = 10
approximate = index.search(X[-1], k, clusters_to_search=8, points_to_rerank=100)
exact       = index.exact_search(X[-1], k))

print('Approximate:', approximate)
print('Exact:', exact)
```

For a more detailed example, see [examples/example.py](examples/example.py).

[Python documentation](https://eliasjaasaari.com/lorann/python.html)

### C++

LoRANN is a header-only library so no installation is required: just include the header `lorann/lorann.h`. A C++ compiler with C++17 support (e.g. gcc/g++ >= 7) is required.

> [!TIP]
> It is recommended to target the correct instruction set (e.g., by using `-march=native`) as LoRANN makes heavy use of SIMD intrinsics. The quantized version of LoRANN can use AVX-512 VNNI instructions, available in Intel Cascade Lake (and later) and AMD Zen 4, for improved performance.

Usage is similar to the Python version:

```cpp
Lorann::Lorann<Lorann::SQ8Quantizer> index(data, n_samples, dim, n_clusters, global_dim);
index.build();

index.search(query, k, clusters_to_search, points_to_rerank, output);
```

For a complete example, see [examples/cpp](examples/cpp).

[C++ documentation](https://eliasjaasaari.com/lorann/cpp.html)

### GPU

Hardware accelerators such as GPUs and TPUs can be used to speed up ANN search for queries that arrive in batches. GPU/TPU support in LoRANN is experimental and available only as a Python module. Currently, the GPU is used only for queries and not for index building.

To use it, install the Python package and import `Lorann` from the desired submodule:

```python
from lorann_gpu.jax import Lorann
```

The available GPU submodules are `jax`, `torch`, `cupy`, and `mlx` with corresponding dependencies [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/), [CuPy](https://cupy.dev/), or [MLX](https://github.com/ml-explore/mlx). The Jax implementation will probably be the fastest, at least on NVIDIA GPUs.

For a complete example, see [examples/gpu_example.py](examples/gpu_example.py)

## Citation

If you use the library in an academic context, please consider citing the following paper:

> Jääsaari, E., Hyvönen, V., & Roos, T. (2024). LoRANN: Low-Rank Matrix Factorization for Approximate Nearest Neighbor Search. Advances in Neural Information Processing Systems, 37.

~~~~
@article{Jaasaari2024,
  title={LoRANN: Low-Rank Matrix Factorization for Approximate Nearest Neighbor Search},
  author={J{\"a}{\"a}saari, Elias and Hyv{\"o}nen, Ville and Roos, Teemu},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  year={2024}
}
~~~~

## License

LoRANN is available under the MIT License (see [LICENSE](LICENSE)). Note that third-party libraries in the [lorann](lorann) folder may be distributed under other open source licenses (see [licenses](licenses)).
