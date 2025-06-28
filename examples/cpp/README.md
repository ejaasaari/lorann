This is an example C++ project that uses LoRANN to index 1 million 300-dimensional FastText word vectors.

Run `make prepare-data` to download the [data](https://fasttext.cc/docs/en/english-vectors.html). The download size is approximately 2.2 GB.

Run `make` to build the executable and run it using `./example`. The example will take a few minutes to run.

Note that you can use 8-bit quantization by changing the type of the index to be `Lorann::Lorann<SQ8Quantizer>`.

The full C++ documentation is available [here](https://ejaasaari.github.io/lorann/cpp.html).
