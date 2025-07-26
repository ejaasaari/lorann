CXX ?= c++
CC  ?= cc

ifeq ($(findstring Apple clang,$(shell $(CXX) --version)),Apple clang)
    OPENMP_FLAGS := -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
    OPENMP_LIB   := -L/opt/homebrew/opt/libomp/lib -lomp
else
    OPENMP_FLAGS := -fopenmp
    ifeq ($(findstring clang,$(shell $(CXX) --version)),clang)
        OPENMP_LIB := -fopenmp -lomp
    else
        OPENMP_LIB := -fopenmp
    endif
endif

ARCH := $(shell uname -m)
ifeq ($(ARCH),x86_64)
    ARCH_FLAGS = -march=native -mtune=native
else ifeq ($(ARCH),aarch64)
    ARCH_FLAGS = -mcpu=native
else ifeq ($(ARCH),arm64)
    ARCH_FLAGS = -mcpu=native
else
    $(error Unsupported architecture: $(ARCH))
endif

CXXFLAGS += -std=c++17 -O3 $(ARCH_FLAGS) -DNDEBUG \
            -Wno-unknown-pragmas -Wno-unknown-warning-option -Wno-unused-function \
            -I../../lorann -flax-vector-conversions $(OPENMP_FLAGS)
LDLIBS   += $(OPENMP_LIB)

all: example

example: example.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

prepare-data:
	@echo "Preparing data…"
	@mkdir mnist
	@wget "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz" -O mnist/train-images-idx3-ubyte.gz -q --show-progress
	@wget "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz" -O mnist/t10k-images-idx3-ubyte.gz -q --show-progress
	@gunzip mnist/train-images-idx3-ubyte.gz
	@gunzip mnist/t10k-images-idx3-ubyte.gz

clean:
	@rm -rf example mnist index.bin

.PHONY: all prepare-data clean
