UNAME_S := $(shell uname -s)

# On macOS: prefer Homebrew's LLVM if it exists ----------------
ifeq ($(UNAME_S),Darwin)
    HOMEBREW_LLVM_PREFIX := $(shell brew --prefix llvm 2>/dev/null)
    ifneq ($(HOMEBREW_LLVM_PREFIX),)
        CXX = $(HOMEBREW_LLVM_PREFIX)/bin/clang++
        CC  = $(HOMEBREW_LLVM_PREFIX)/bin/clang
        LLVM_LIBDIR := $(HOMEBREW_LLVM_PREFIX)/lib
        OPENMP_LDFLAGS := -L$(LLVM_LIBDIR) -Wl,-rpath,$(LLVM_LIBDIR)
    endif
endif

CXX ?= c++
CC  ?= cc

ifeq ($(findstring clang,$(shell $(CXX) --version)),clang)
    OPENMP_FLAGS  := -fopenmp
    OPENMP_LIB    := -lomp
else
    OPENMP_FLAGS  := -fopenmp
    OPENMP_LIB    := -lgomp
endif

CXXFLAGS += -std=c++17 -O3 -march=native -DNDEBUG \
            -I../../lorann -flax-vector-conversions $(OPENMP_FLAGS)
LDFLAGS  += $(OPENMP_LDFLAGS)
LDLIBS   += $(OPENMP_LIB)

all: example

example: example.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

prepare-data:
	@echo "Preparing data…"
	@wget "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip" -q --show-progress
	@unzip -q wiki-news-300d-1M.vec.zip
	@rm wiki-news-300d-1M.vec.zip

clean:
	@rm -f example wiki-news-300d-1M.vec index.bin

.PHONY: all prepare-data clean
