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

- Header-only C++17 library with Python bindings
- Query speed matching state-of-the-art graph methods but with tiny memory usage
- Optimized for modern high-dimensional (d > 100) embedding data sets
- Optimized for modern CPU architectures with acceleration for AVX2, AVX-512, and ARM NEON
- State-of-the-art query speed for GPU batch queries (experimental)
- Supported data types: float32, float16, bfloat16, uint8, binary
- Supported distances: (negative) inner product, Euclidean distance, cosine distance
- Support for index serialization

## Getting started

### Python

Install the module with `pip install lorann`

> [!TIP]
> On macOS, it is recommended to use the Homebrew version of Clang as the compiler:

```shell script
brew install llvm libomp
CC=/opt/homebrew/opt/llvm/bin/clang CXX=/opt/homebrew/opt/llvm/bin/clang++ pip install lorann
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
    distance=lorann.L2,
)

index.build(verbose=False)

k = 10
approximate = index.search(X[-1], k, clusters_to_search=8, points_to_rerank=100)
exact       = index.exact_search(X[-1], k)

print('Approximate:', approximate)
print('Exact:', exact)
```

The data matrix should have type `float32`, `float16`, `uint8`, or `uint16` ([for bfloat16](https://github.com/ashvardanian/SimSIMD?tab=readme-ov-file#half-precision-brain-float-numbers)). For binary data (packed into uint8 bytes using e.g. `numpy.packbits`), use `LorannBinaryIndex`.

The distance should be either `lorann.L2` (for squared Euclidean distance), `lorann.IP` (for inner product), or `lorann.HAMMING` (only for binary data). For cosine distance, normalize vectors to unit norm and use `lorann.IP`.

For a more detailed example, see [examples/example.py](examples/example.py).

[Python documentation](https://eliasjaasaari.com/lorann/python.html)

### C++

LoRANN is a header-only library so no installation is required: just include the header `lorann/lorann.h`. A C++ compiler with C++17 support (e.g. gcc/g++ >= 7) is required.

> [!TIP]
> It is recommended to target the correct instruction set (e.g., by using `-march=native`) as LoRANN makes heavy use of SIMD intrinsics. The quantized version of LoRANN can use AVX-512 VNNI instructions, available in Intel Cascade Lake (and later) and AMD Zen 4, for improved performance.

Usage is similar to the Python version:

```cpp
Lorann::Lorann<float, Lorann::SQ8Quantizer> index(data, n_samples, dim, n_clusters, global_dim);
index.build();

index.search(query, k, clusters_to_search, points_to_rerank, output);
```

For a complete example, see [examples/cpp](examples/cpp).

[C++ documentation](https://eliasjaasaari.com/lorann/cpp.html)

### GPU

Hardware accelerators such as GPUs and TPUs can be used to speed up ANN search for queries that arrive in batches. GPU support in LoRANN is experimental and available only as a Python module. For index building, the GPU is currently used only partially.

Usage is similar to the CPU version, just with a different import:

```python
from lorann_gpu import LorannIndex
```

The GPU submodule is implemented with and depends on [PyTorch](https://pytorch.org/), but a native CUDA implementation would provide better performance and will be available in the future. All index structures are by default saved as `torch.float32` to ensure numerical stability. However, for most embedding datasets, better performance can be obtained by passing `dtype=torch.float16` to the index constructor.

For a complete example, see [examples/gpu_example.py](examples/gpu_example.py)

## Citation

If you use the library in an academic context, please consider citing the following paper:

> Jääsaari, E., Hyvönen, V., Roos, T. LoRANN: Low-Rank Matrix Factorization for Approximate Nearest Neighbor Search. Advances in Neural Information Processing Systems 37 (2024): 102121-102153.

~~~~
@article{Jaasaari2024lorann,
  title={{LoRANN}: Low-Rank Matrix Factorization for Approximate Nearest Neighbor Search},
  author={J{\"a}{\"a}saari, Elias and Hyv{\"o}nen, Ville and Roos, Teemu},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={102121--102153},
  year={2024}
}
~~~~

## License

LoRANN is available under the MIT License (see [LICENSE](LICENSE)). Note that third-party libraries in the [lorann](lorann) folder may be distributed under other open source licenses (see [licenses](licenses)).
