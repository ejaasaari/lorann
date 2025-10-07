#define PY_SSIZE_T_CLEAN
#define EIGEN_DONT_PARALLELIZE

#ifdef _OPENMP
#include <omp.h>
#endif

#include <sys/stat.h>
#include <sys/types.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "Python.h"

#ifndef _WIN32
#include <sys/mman.h>
#endif

#include <simsimd/simsimd.h>

#include <Eigen/Dense>
#include <cereal/archives/binary.hpp>

#include "lorann.h"
#include "numpy/arrayobject.h"

template <typename T>
struct npy_lorann_dist;
template <>
struct npy_lorann_dist<float> {
  static constexpr int value = NPY_FLOAT32;
};
template <>
struct npy_lorann_dist<double> {
  static constexpr int value = NPY_FLOAT64;
};

template <typename T>
constexpr int npy_lorann_dist_v = npy_lorann_dist<T>::value;

typedef struct {
  PyObject_HEAD std::unique_ptr<Lorann::KMeans> index;
  PyArrayObject *py_data;
} KMeansIndex;

template <typename T>
struct LorannIndex {
  PyObject_HEAD std::unique_ptr<Lorann::LorannBase<T>> index;
  PyArrayObject *py_data;
};

static PyObject *KMeans_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  KMeansIndex *self = reinterpret_cast<KMeansIndex *>(type->tp_alloc(type, 0));

  if (self != NULL) {
    self->py_data = NULL;
  }

  return reinterpret_cast<PyObject *>(self);
}

static int KMeans_init(KMeansIndex *self, PyObject *args, PyObject *kwds) {
  int n_clusters, iters, balanced, max_balance_diff;
  Lorann::Distance distance;
  float penalty_factor;

  if (!PyArg_ParseTuple(args, "iiiiif", &n_clusters, &iters, &distance, &balanced,
                        &max_balance_diff, &penalty_factor)) {
    return -1;
  }

  try {
    self->index = std::make_unique<Lorann::KMeans>(n_clusters, iters, distance, balanced,
                                                   max_balance_diff, penalty_factor);
  } catch (const std::exception &e) {
    PyErr_Format(PyExc_RuntimeError, "Failed to initialize KMeans index: %s", e.what());
    return -1;
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "Unknown error during KMeans index initialization");
    return -1;
  }
  return 0;
}

static void kmeans_dealloc(KMeansIndex *self) {
  if (self->index) {
    self->index.reset();
  }

  Py_XDECREF(self->py_data);
  self->py_data = NULL;

  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

static PyObject *kmeans_train(KMeansIndex *self, PyObject *args) {
  PyArrayObject *py_data;
  int n, dim, verbose, n_threads;

  if (!PyArg_ParseTuple(args, "O!iiii", &PyArray_Type, &py_data, &n, &dim, &verbose, &n_threads))
    return NULL;

#ifdef _OPENMP
  if (n_threads <= 0) {
    n_threads = omp_get_max_threads();
  }
#endif

  Py_INCREF(py_data);
  self->py_data = py_data;

  float *data = reinterpret_cast<float *>(PyArray_DATA(py_data));

  std::vector<std::vector<int>> idxs;

  std::exception_ptr eptr;
  Py_BEGIN_ALLOW_THREADS try {
    idxs = self->index->train(data, n, dim, verbose, n_threads);
  } catch (...) {
    eptr = std::current_exception();
  }
  Py_END_ALLOW_THREADS

      if (eptr) {
    try {
      std::rethrow_exception(eptr);
    } catch (const std::exception &e) {
      Py_DECREF(py_data);
      self->py_data = NULL;
      PyErr_Format(PyExc_RuntimeError, "Failed to train KMeans index: %s", e.what());
      return NULL;
    } catch (...) {
      Py_DECREF(py_data);
      self->py_data = NULL;
      PyErr_SetString(PyExc_RuntimeError, "Unknown error during KMeans training");
      return NULL;
    }
  }

  PyObject *list = PyList_New(idxs.size());
  for (size_t i = 0; i < idxs.size(); ++i) {
    npy_intp dims[1] = {static_cast<npy_intp>(idxs[i].size())};
    PyObject *cs = PyArray_SimpleNew(1, dims, NPY_INT);
    int *outdata = reinterpret_cast<int *>(PyArray_DATA((PyArrayObject *)cs));
    std::memcpy(outdata, idxs[i].data(), idxs[i].size() * sizeof(int));
    PyList_SetItem(list, i, cs);
  }

  return list;
}

static PyObject *kmeans_get_n_clusters(KMeansIndex *self, PyObject *args) {
  return PyLong_FromLong(static_cast<long>(self->index->get_n_clusters()));
}

static PyObject *kmeans_get_iters(KMeansIndex *self, PyObject *args) {
  return PyLong_FromLong(static_cast<long>(self->index->get_iters()));
}

static PyObject *kmeans_get_balanced(KMeansIndex *self, PyObject *args) {
  return PyBool_FromLong(static_cast<long>(self->index->is_balanced()));
}

static PyObject *kmeans_get_centroids(KMeansIndex *self, PyObject *args) {
  try {
    const Lorann::RowMatrix centroids = self->index->get_centroids();
    const int n_clusters = centroids.rows();
    const int dim = centroids.cols();

    npy_intp dims[2] = {n_clusters, dim};
    PyObject *ret = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    float *outdata = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)ret));
    std::memcpy(outdata, centroids.data(), n_clusters * dim * sizeof(float));

    return ret;
  } catch (const std::exception &e) {
    PyErr_Format(PyExc_RuntimeError, "Failed to get centroids: %s", e.what());
    return NULL;
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "Unknown error while getting centroids");
    return NULL;
  }
}

static PyObject *kmeans_assign(KMeansIndex *self, PyObject *args) {
  PyObject *py_data;
  int n, dim, k;

  if (!PyArg_ParseTuple(args, "Oiii", &py_data, &n, &dim, &k)) return NULL;

  float *data = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)py_data));

  std::vector<std::vector<int>> idxs;
  try {
    idxs = self->index->assign(data, n, k);
  } catch (const std::exception &e) {
    PyErr_Format(PyExc_RuntimeError, "Failed to assign clusters: %s", e.what());
    return NULL;
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "Unknown error during cluster assignment");
    return NULL;
  }

  PyObject *list = PyList_New(idxs.size());
  for (size_t i = 0; i < idxs.size(); ++i) {
    npy_intp dims[1] = {static_cast<npy_intp>(idxs[i].size())};
    PyObject *cs = PyArray_SimpleNew(1, dims, NPY_INT);
    int *outdata = reinterpret_cast<int *>(PyArray_DATA((PyArrayObject *)cs));
    std::memcpy(outdata, idxs[i].data(), idxs[i].size() * sizeof(int));
    PyList_SetItem(list, i, cs);
  }

  return list;
}

template <typename T>
static PyObject *Lorann_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  LorannIndex<T> *self = reinterpret_cast<LorannIndex<T> *>(type->tp_alloc(type, 0));

  if (self != NULL) {
    self->py_data = NULL;
  }

  return reinterpret_cast<PyObject *>(self);
}

template <typename T>
static int Lorann_init(LorannIndex<T> *self, PyObject *args, PyObject *kwds) {
  PyArrayObject *py_data;
  int n, dim, quantization_bits, n_clusters, global_dim, rank, train_size, balanced, copy;
  Lorann::Distance distance;

  if (!PyArg_ParseTuple(args, "O!iiiiiiiiii", &PyArray_Type, &py_data, &n, &dim, &quantization_bits,
                        &n_clusters, &global_dim, &rank, &train_size, &distance, &balanced,
                        &copy)) {
    return -1;
  }

  if (!copy) {
    Py_INCREF(py_data);
    self->py_data = py_data;
  }

  T *data = reinterpret_cast<T *>(PyArray_DATA(py_data));
  try {
    if (quantization_bits == 4) {
      self->index = std::make_unique<Lorann::Lorann<T, Lorann::SQ4Quantizer>>(
          data, n, dim, n_clusters, global_dim, rank, train_size, distance, balanced);
    } else if (quantization_bits == 8) {
      self->index = std::make_unique<Lorann::Lorann<T, Lorann::SQ8Quantizer>>(
          data, n, dim, n_clusters, global_dim, rank, train_size, distance, balanced);
    } else {
      self->index = std::make_unique<Lorann::LorannFP<T>>(data, n, dim, n_clusters, global_dim,
                                                          rank, train_size, distance, balanced);
    }
  } catch (const std::exception &e) {
    if (!copy) {
      Py_DECREF(py_data);
      self->py_data = NULL;
    }
    PyErr_Format(PyExc_RuntimeError, "Failed to initialize Lorann index: %s", e.what());
    return -1;
  } catch (...) {
    if (!copy) {
      Py_DECREF(py_data);
      self->py_data = NULL;
    }
    PyErr_SetString(PyExc_RuntimeError, "Unknown error during Lorann index initialization");
    return -1;
  }

  return 0;
}

template <typename T>
static void lorann_dealloc(LorannIndex<T> *self) {
  if (self->index) {
    self->index.reset();
  }

  if (self->py_data) {
    Py_XDECREF(self->py_data);
    self->py_data = NULL;
  }

  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

template <typename T>
static PyObject *lorann_get_n_samples(LorannIndex<T> *self, PyObject *args) {
  return PyLong_FromLong(static_cast<long>(self->index->get_n_samples()));
}

template <typename T>
static PyObject *lorann_get_dim(LorannIndex<T> *self, PyObject *args) {
  return PyLong_FromLong(static_cast<long>(self->index->get_dim()));
}

template <typename T>
static PyObject *lorann_get_n_clusters(LorannIndex<T> *self, PyObject *args) {
  return PyLong_FromLong(static_cast<long>(self->index->get_n_clusters()));
}

template <typename T>
static PyObject *lorann_get_dissimilarity(LorannIndex<T> *self, PyObject *args) {
  PyArrayObject *u;
  PyArrayObject *v;
  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &u, &PyArray_Type, &v)) return NULL;

  T *u_data = reinterpret_cast<T *>(PyArray_DATA(u));
  T *v_data = reinterpret_cast<T *>(PyArray_DATA(v));

  const lorann_dist_t dissimilarity = self->index->get_dissimilarity(u_data, v_data);
  return PyFloat_FromDouble(static_cast<double>(dissimilarity));
}

template <typename T>
static PyObject *lorann_build(LorannIndex<T> *self, PyObject *args) {
  int approximate = 1;
  int verbose = 0;
  int n_threads = -1;
  PyArrayObject *Q = NULL;

  if (!PyArg_ParseTuple(args, "iii|O!", &approximate, &verbose, &n_threads, &PyArray_Type, &Q))
    return NULL;

#ifdef _OPENMP
  if (n_threads <= 0) {
    n_threads = omp_get_max_threads();
  }
#endif

  std::exception_ptr eptr;

  if (Q != NULL) {
    T *indata = reinterpret_cast<T *>(PyArray_DATA(Q));
    int n = PyArray_DIM(Q, 0);
    Py_BEGIN_ALLOW_THREADS try {
      self->index->build(indata, n, approximate, verbose, n_threads);
    } catch (...) {
      eptr = std::current_exception();
    }
    Py_END_ALLOW_THREADS
  } else {
    Py_BEGIN_ALLOW_THREADS try {
      self->index->build(approximate, verbose, n_threads);
    } catch (...) {
      eptr = std::current_exception();
    }
    Py_END_ALLOW_THREADS
  }

  if (eptr) {
    try {
      std::rethrow_exception(eptr);
    } catch (const std::exception &e) {
      PyErr_Format(PyExc_RuntimeError, "Failed to build index: %s", e.what());
      return NULL;
    } catch (...) {
      PyErr_SetString(PyExc_RuntimeError, "Unknown error during index build");
      return NULL;
    }
  }

  Py_RETURN_NONE;
}

template <typename T>
static PyObject *lorann_search(LorannIndex<T> *self, PyObject *args) {
  PyArrayObject *v;
  int k, dim, n, clusters_to_search, points_to_rerank, return_distances, n_threads;

  if (!PyArg_ParseTuple(args, "O!iiiii", &PyArray_Type, &v, &k, &clusters_to_search,
                        &points_to_rerank, &return_distances, &n_threads))
    return NULL;

#ifdef _OPENMP
  if (n_threads <= 0) {
    n_threads = omp_get_max_threads();
  }
#endif

  T *indata = reinterpret_cast<T *>(PyArray_DATA(v));
  PyObject *nearest;

  if (PyArray_NDIM(v) == 1) {
    dim = PyArray_DIM(v, 0);

    npy_intp dims[1] = {k};
    nearest = PyArray_SimpleNew(1, dims, NPY_INT);
    int *out_idx = reinterpret_cast<int *>(PyArray_DATA((PyArrayObject *)nearest));

    if (return_distances) {
      PyObject *distances = PyArray_SimpleNew(1, dims, npy_lorann_dist_v<lorann_dist_t>);
      lorann_dist_t *out_distances =
          reinterpret_cast<lorann_dist_t *>(PyArray_DATA((PyArrayObject *)distances));

      Py_BEGIN_ALLOW_THREADS;
      self->index->search(indata, k, clusters_to_search, points_to_rerank, out_idx, out_distances);
      Py_END_ALLOW_THREADS;

      PyObject *out_tuple = PyTuple_New(2);
      PyTuple_SetItem(out_tuple, 0, nearest);
      PyTuple_SetItem(out_tuple, 1, distances);
      return out_tuple;
    } else {
      Py_BEGIN_ALLOW_THREADS;
      self->index->search(indata, k, clusters_to_search, points_to_rerank, out_idx);
      Py_END_ALLOW_THREADS;
      return nearest;
    }
  } else {
    n = PyArray_DIM(v, 0);
    dim = PyArray_DIM(v, 1);

    npy_intp dims[2] = {n, k};
    nearest = PyArray_SimpleNew(2, dims, NPY_INT);
    int *out_idx = reinterpret_cast<int *>(PyArray_DATA((PyArrayObject *)nearest));

    if (return_distances) {
      npy_intp dims[2] = {n, k};
      PyObject *distances = PyArray_SimpleNew(2, dims, npy_lorann_dist_v<lorann_dist_t>);
      lorann_dist_t *out_distances =
          reinterpret_cast<lorann_dist_t *>(PyArray_DATA((PyArrayObject *)distances));

      Py_BEGIN_ALLOW_THREADS;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
      for (int i = 0; i < n; ++i) {
        self->index->search(indata + i * dim, k, clusters_to_search, points_to_rerank,
                            out_idx + i * k, out_distances + i * k);
      }
      Py_END_ALLOW_THREADS;

      PyObject *out_tuple = PyTuple_New(2);
      PyTuple_SetItem(out_tuple, 0, nearest);
      PyTuple_SetItem(out_tuple, 1, distances);
      return out_tuple;
    } else {
      Py_BEGIN_ALLOW_THREADS;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
      for (int i = 0; i < n; ++i) {
        self->index->search(indata + i * dim, k, clusters_to_search, points_to_rerank,
                            out_idx + i * k);
      }
      Py_END_ALLOW_THREADS;
      return nearest;
    }
  }
}

template <typename T>
static PyObject *lorann_exact_search(LorannIndex<T> *self, PyObject *args) {
  PyArrayObject *v;
  int k, dim, n, return_distances, n_threads;

  if (!PyArg_ParseTuple(args, "O!iii", &PyArray_Type, &v, &k, &return_distances, &n_threads))
    return NULL;

#ifdef _OPENMP
  if (n_threads <= 0) {
    n_threads = omp_get_max_threads();
  }
#endif

  T *indata = reinterpret_cast<T *>(PyArray_DATA(v));
  PyObject *nearest;

  if (PyArray_NDIM(v) == 1) {
    dim = PyArray_DIM(v, 0);

    npy_intp dims[1] = {k};
    nearest = PyArray_SimpleNew(1, dims, NPY_INT);
    int *out_idx = reinterpret_cast<int *>(PyArray_DATA((PyArrayObject *)nearest));

    if (return_distances) {
      PyObject *distances = PyArray_SimpleNew(1, dims, npy_lorann_dist_v<lorann_dist_t>);
      lorann_dist_t *out_distances =
          reinterpret_cast<lorann_dist_t *>(PyArray_DATA((PyArrayObject *)distances));

      Py_BEGIN_ALLOW_THREADS;
      self->index->exact_search(indata, k, out_idx, out_distances);
      Py_END_ALLOW_THREADS;

      PyObject *out_tuple = PyTuple_New(2);
      PyTuple_SetItem(out_tuple, 0, nearest);
      PyTuple_SetItem(out_tuple, 1, distances);
      return out_tuple;
    } else {
      Py_BEGIN_ALLOW_THREADS;
      self->index->exact_search(indata, k, out_idx);
      Py_END_ALLOW_THREADS;
      return nearest;
    }
  } else {
    n = PyArray_DIM(v, 0);
    dim = PyArray_DIM(v, 1);

    npy_intp dims[2] = {n, k};
    nearest = PyArray_SimpleNew(2, dims, NPY_INT);
    int *out_idx = reinterpret_cast<int *>(PyArray_DATA((PyArrayObject *)nearest));

    if (return_distances) {
      npy_intp dims[2] = {n, k};
      PyObject *distances = PyArray_SimpleNew(2, dims, npy_lorann_dist_v<lorann_dist_t>);
      lorann_dist_t *out_distances =
          reinterpret_cast<lorann_dist_t *>(PyArray_DATA((PyArrayObject *)distances));

      Py_BEGIN_ALLOW_THREADS;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
      for (int i = 0; i < n; ++i) {
        self->index->exact_search(indata + i * dim, k, out_idx + i * k, out_distances + i * k);
      }
      Py_END_ALLOW_THREADS;

      PyObject *out_tuple = PyTuple_New(2);
      PyTuple_SetItem(out_tuple, 0, nearest);
      PyTuple_SetItem(out_tuple, 1, distances);
      return out_tuple;
    } else {
      Py_BEGIN_ALLOW_THREADS;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
      for (int i = 0; i < n; ++i) {
        self->index->exact_search(indata + i * dim, k, out_idx + i * k);
      }
      Py_END_ALLOW_THREADS;
      return nearest;
    }
  }
}

template <typename T>
static PyObject *lorann_save(LorannIndex<T> *self, PyObject *args) {
  const char *fname;
  if (!PyArg_ParseTuple(args, "s", &fname)) return NULL;

  try {
    std::ofstream output_file(fname, std::ios::binary);
    output_file << self->index->get_type_marker();
    cereal::BinaryOutputArchive output_archive(output_file);
    output_archive(self->index);
  } catch (...) {
    PyErr_Format(PyExc_IOError, "Failed to write to file '%s'", fname);
    return NULL;
  }

  Py_RETURN_NONE;
}

#define DECLARE_LORANN_PYTYPE(NAME, TYPE)                                                   \
  static PyMethodDef NAME##LorannMethods[] = {                                              \
      {"exact_search", (PyCFunction)lorann_exact_search<TYPE>, METH_VARARGS, ""},           \
      {"search", (PyCFunction)lorann_search<TYPE>, METH_VARARGS, ""},                       \
      {"build", (PyCFunction)lorann_build<TYPE>, METH_VARARGS, ""},                         \
      {"save", (PyCFunction)lorann_save<TYPE>, METH_VARARGS, ""},                           \
      {"get_n_samples", (PyCFunction)lorann_get_n_samples<TYPE>, METH_NOARGS, ""},          \
      {"get_dim", (PyCFunction)lorann_get_dim<TYPE>, METH_NOARGS, ""},                      \
      {"get_n_clusters", (PyCFunction)lorann_get_n_clusters<TYPE>, METH_NOARGS, ""},        \
      {"get_dissimilarity", (PyCFunction)lorann_get_dissimilarity<TYPE>, METH_VARARGS, ""}, \
      {NULL, NULL, 0, NULL}};                                                               \
  static PyTypeObject NAME##LorannIndexType = {                                             \
      PyVarObject_HEAD_INIT(NULL, 0) "lorann." #NAME "LorannIndex", /* tp_name  */          \
      sizeof(LorannIndex<TYPE>),                                    /* tp_basicsize */      \
      0,                                                            /* tp_itemsize  */      \
      (destructor)lorann_dealloc<TYPE>,                             /* tp_dealloc   */      \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags     */                          \
      #NAME "Lorann index object",              /* tp_doc       */                          \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      NAME##LorannMethods, /* tp_methods   */                                               \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      0,                                                                                    \
      (initproc)Lorann_init<TYPE>, /* tp_init      */                                       \
      0,                           /* tp_alloc     */                                       \
      Lorann_new<TYPE>             /* tp_new       */                                       \
  }

DECLARE_LORANN_PYTYPE(FP32, float);
#if SIMSIMD_NATIVE_F16
DECLARE_LORANN_PYTYPE(FP16, simsimd_f16_t);
#endif
#if SIMSIMD_NATIVE_BF16
DECLARE_LORANN_PYTYPE(BF16, simsimd_bf16_t);
#endif
DECLARE_LORANN_PYTYPE(U8, uint8_t);
DECLARE_LORANN_PYTYPE(Binary, Lorann::BinaryType);

static PyMethodDef KMeansMethods[] = {
    {"train", (PyCFunction)kmeans_train, METH_VARARGS, ""},
    {"get_n_clusters", (PyCFunction)kmeans_get_n_clusters, METH_NOARGS, ""},
    {"get_iters", (PyCFunction)kmeans_get_iters, METH_NOARGS, ""},
    {"get_balanced", (PyCFunction)kmeans_get_balanced, METH_NOARGS, ""},
    {"get_centroids", (PyCFunction)kmeans_get_centroids, METH_VARARGS, ""},
    {"assign", (PyCFunction)kmeans_assign, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyTypeObject KMeansIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0) "lorann.KMeansIndex", /* tp_name */
    sizeof(KMeansIndex),                                 /* tp_basicsize */
    0,                                                   /* tp_itemsize */
    (destructor)kmeans_dealloc,                          /* tp_dealloc */
    0,                                                   /* tp_vectorcall_offset */
    0,                                                   /* tp_getattr */
    0,                                                   /* tp_setattr */
    0,                                                   /* tp_as_async */
    0,                                                   /* tp_repr */
    0,                                                   /* tp_as_number */
    0,                                                   /* tp_as_sequence */
    0,                                                   /* tp_as_mapping */
    0,                                                   /* tp_hash */
    0,                                                   /* tp_call */
    0,                                                   /* tp_str */
    0,                                                   /* tp_getattro */
    0,                                                   /* tp_setattro */
    0,                                                   /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,            /* tp_flags */
    "KMeans index object",                               /* tp_doc */
    0,                                                   /* tp_traverse */
    0,                                                   /* tp_clear */
    0,                                                   /* tp_richcompare */
    0,                                                   /* tp_weaklistoffset */
    0,                                                   /* tp_iter */
    0,                                                   /* tp_iternext */
    KMeansMethods,                                       /* tp_methods */
    0,                                                   /* tp_members */
    0,                                                   /* tp_getset */
    0,                                                   /* tp_base */
    0,                                                   /* tp_dict */
    0,                                                   /* tp_descr_get */
    0,                                                   /* tp_descr_set */
    0,                                                   /* tp_dictoffset */
    (initproc)KMeans_init,                               /* tp_init */
    0,                                                   /* tp_alloc */
    KMeans_new,                                          /* tp_new */
};

template <typename T>
static PyObject *load_impl(PyTypeObject *type, std::ifstream &input_file, const char *fname) {
  LorannIndex<T> *self = reinterpret_cast<LorannIndex<T> *>(type->tp_alloc(type, 0));
  if (!self) return nullptr;

  try {
    cereal::BinaryInputArchive ar(input_file);
    ar(self->index);
  } catch (...) {
    Py_DECREF(reinterpret_cast<PyObject *>(self));
    PyErr_Format(PyExc_IOError, "Failed to load index from file '%s'", fname);
    return nullptr;
  }

  self->py_data = nullptr;
  return reinterpret_cast<PyObject *>(self);
}

static PyObject *lorann_load(PyObject *self, PyObject *args) {
  const char *fname;
  if (!PyArg_ParseTuple(args, "s", &fname)) return nullptr;

  std::ifstream input_file(fname, std::ios::binary);
  if (!input_file) {
    PyErr_Format(PyExc_FileNotFoundError, "Cannot open '%s' for reading", fname);
    return nullptr;
  }

  int type_marker;
  input_file >> type_marker;

  switch (type_marker) {
    case Lorann::detail::FLOAT32:
      return load_impl<float>(&FP32LorannIndexType, input_file, fname);
#if SIMSIMD_NATIVE_F16
    case Lorann::detail::FLOAT16:
      return load_impl<simsimd_f16_t>(&FP16LorannIndexType, input_file, fname);
#endif
#if SIMSIMD_NATIVE_BF16
    case Lorann::detail::BFLOAT16:
      return load_impl<simsimd_bf16_t>(&BF16LorannIndexType, input_file, fname);
#endif
    case Lorann::detail::UINT8:
      return load_impl<uint8_t>(&U8LorannIndexType, input_file, fname);
    case Lorann::detail::BINARY:
      return load_impl<Lorann::BinaryType>(&BinaryLorannIndexType, input_file, fname);
    default:
      PyErr_Format(PyExc_ValueError, "Unknown type marker %d in file '%s'", type_marker, fname);
      return nullptr;
  }
}

static PyObject *lorann_compute_V(PyObject *self, PyObject *args) {
  PyArrayObject *A;
  int rank;

  if (!PyArg_ParseTuple(args, "O!i|i", &PyArray_Type, &A, &rank)) return NULL;

  const int n = PyArray_DIM(A, 0);
  const int d = PyArray_DIM(A, 1);
  float *A_data = reinterpret_cast<float *>(PyArray_DATA(A));

  try {
    Eigen::MatrixXf Y = Eigen::Map<Lorann::RowMatrix>(A_data, n, d);
    Eigen::MatrixXf V = Lorann::compute_V(Y, rank);

    PyObject *ret;
    npy_intp dims[2] = {V.cols(), V.rows()};
    ret = PyArray_SimpleNew(2, dims, NPY_FLOAT32);

    float *outdata = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)ret));
    std::memcpy(outdata, V.data(), V.rows() * V.cols() * sizeof(float));

    return ret;
  } catch (const std::exception &e) {
    PyErr_Format(PyExc_RuntimeError, "Failed to compute V: %s", e.what());
    return NULL;
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "Unknown error while computing V");
    return NULL;
  }
}

static PyMethodDef module_methods[] = {
    {"compute_V", (PyCFunction)lorann_compute_V, METH_VARARGS, ""},
    {"load_index", (PyCFunction)lorann_load, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "lorannlib",    /* m_name */
    "",             /* m_doc */
    -1,             /* m_size */
    module_methods, /* m_methods */
    NULL,           /* m_slots */
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL            /* m_free */
};

#define ADD_LORANN_INDEX_TYPE(m, TYPE)                                        \
  do {                                                                        \
    Py_INCREF(&TYPE##LorannIndexType);                                        \
    PyModule_AddObject(m, #TYPE "LorannIndex",                                \
                       reinterpret_cast<PyObject *>(&TYPE##LorannIndexType)); \
  } while (0)

PyMODINIT_FUNC PyInit_lorannlib(void) {
  PyObject *m;
  if (PyType_Ready(&FP32LorannIndexType) < 0) return NULL;
#if SIMSIMD_NATIVE_F16
  if (PyType_Ready(&FP16LorannIndexType) < 0) return NULL;
#endif
#if SIMSIMD_NATIVE_BF16
  if (PyType_Ready(&BF16LorannIndexType) < 0) return NULL;
#endif
  if (PyType_Ready(&U8LorannIndexType) < 0) return NULL;
  if (PyType_Ready(&BinaryLorannIndexType) < 0) return NULL;
  if (PyType_Ready(&KMeansIndexType) < 0) return NULL;

  m = PyModule_Create(&moduledef);

  if (m == NULL) return NULL;

  import_array();

  ADD_LORANN_INDEX_TYPE(m, FP32);
#if SIMSIMD_NATIVE_F16
  ADD_LORANN_INDEX_TYPE(m, FP16);
#endif
#if SIMSIMD_NATIVE_BF16
  ADD_LORANN_INDEX_TYPE(m, BF16);
#endif
  ADD_LORANN_INDEX_TYPE(m, U8);
  ADD_LORANN_INDEX_TYPE(m, Binary);

  Py_INCREF(&KMeansIndexType);
  PyModule_AddObject(m, "KMeans", reinterpret_cast<PyObject *>(&KMeansIndexType));

  PyModule_AddIntConstant(m, "IP", Lorann::IP);
  PyModule_AddIntConstant(m, "L2", Lorann::L2);
  PyModule_AddIntConstant(m, "HAMMING", Lorann::HAMMING);

  return m;
}
