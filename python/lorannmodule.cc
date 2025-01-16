#define PY_SSIZE_T_CLEAN

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

#include <Eigen/Dense>
#include <cereal/archives/binary.hpp>

#include "lorann.h"
#include "lorann_base.h"
#include "lorann_fp.h"
#include "numpy/arrayobject.h"
#include "quant.h"
#include "utils.h"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrix;

typedef struct {
  PyObject_HEAD std::unique_ptr<Lorann::KMeans> index;
  PyArrayObject *py_data;
} KMeansIndex;

typedef struct {
  PyObject_HEAD std::unique_ptr<Lorann::LorannBase> index;
  PyArrayObject *py_data;
} LorannIndex;

static PyObject *KMeans_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  KMeansIndex *self = reinterpret_cast<KMeansIndex *>(type->tp_alloc(type, 0));

  if (self != NULL) {
    self->py_data = NULL;
  }

  return reinterpret_cast<PyObject *>(self);
}

static int KMeans_init(KMeansIndex *self, PyObject *args, PyObject *kwds) {
  int n_clusters, iters, euclidean, balanced, max_balance_diff, verbose;

  if (!PyArg_ParseTuple(args, "iiiiii", &n_clusters, &iters, &euclidean, &balanced,
                        &max_balance_diff, &verbose)) {
    return -1;
  }

  self->index = std::make_unique<Lorann::KMeans>(n_clusters, iters, euclidean, balanced,
                                                 max_balance_diff, verbose);
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
  int n, dim, n_threads;

  if (!PyArg_ParseTuple(args, "O!iii", &PyArray_Type, &py_data, &n, &dim, &n_threads)) return NULL;

#ifdef _OPENMP
  if (n_threads <= 0) {
    n_threads = omp_get_max_threads();
  }
#endif

  Py_INCREF(py_data);
  self->py_data = py_data;

  float *data = reinterpret_cast<float *>(PyArray_DATA(py_data));

  std::vector<std::vector<int>> idxs;
  Py_BEGIN_ALLOW_THREADS;
  idxs = self->index->train(data, n, dim, n_threads);
  Py_END_ALLOW_THREADS;

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

static PyObject *kmeans_get_euclidean(KMeansIndex *self, PyObject *args) {
  return PyBool_FromLong(static_cast<long>(self->index->get_euclidean()));
}

static PyObject *kmeans_get_balanced(KMeansIndex *self, PyObject *args) {
  return PyBool_FromLong(static_cast<long>(self->index->is_balanced()));
}

static PyObject *kmeans_get_centroids(KMeansIndex *self, PyObject *args) {
  const RowMatrix centroids = self->index->get_centroids();
  const int n_clusters = centroids.rows();
  const int dim = centroids.cols();

  npy_intp dims[2] = {n_clusters, dim};
  PyObject *ret = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
  float *outdata = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)ret));
  std::memcpy(outdata, centroids.data(), n_clusters * dim * sizeof(float));

  return ret;
}

static PyObject *kmeans_assign(KMeansIndex *self, PyObject *args) {
  PyObject *py_data;
  int n, dim, k;

  if (!PyArg_ParseTuple(args, "Oiii", &py_data, &n, &dim, &k)) return NULL;

  float *data = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)py_data));

  std::vector<std::vector<int>> idxs = self->index->assign(data, n, k);

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

static PyObject *Lorann_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  LorannIndex *self = reinterpret_cast<LorannIndex *>(type->tp_alloc(type, 0));

  if (self != NULL) {
    self->py_data = NULL;
  }

  return reinterpret_cast<PyObject *>(self);
}

static int Lorann_init(LorannIndex *self, PyObject *args, PyObject *kwds) {
  PyArrayObject *py_data;
  int n, dim, quantization_bits, n_clusters, global_dim, rank, train_size, euclidean, balanced;

  if (!PyArg_ParseTuple(args, "O!iiiiiiiii", &PyArray_Type, &py_data, &n, &dim, &quantization_bits,
                        &n_clusters, &global_dim, &rank, &train_size, &euclidean, &balanced)) {
    return -1;
  }

  Py_INCREF(py_data);
  self->py_data = py_data;

  float *data = reinterpret_cast<float *>(PyArray_DATA(py_data));
  if (quantization_bits == 4) {
    self->index = std::make_unique<Lorann::Lorann<Lorann::SQ4Quantizer>>(
        data, n, dim, n_clusters, global_dim, rank, train_size, euclidean, balanced);
  } else if (quantization_bits == 8) {
    self->index = std::make_unique<Lorann::Lorann<Lorann::SQ8Quantizer>>(
        data, n, dim, n_clusters, global_dim, rank, train_size, euclidean, balanced);
  } else {
    self->index = std::make_unique<Lorann::LorannFP>(data, n, dim, n_clusters, global_dim, rank,
                                                     train_size, euclidean, balanced);
  }

  return 0;
}

static void lorann_dealloc(LorannIndex *self) {
  if (self->index) {
    self->index.reset();
  }

  if (self->py_data) {
    Py_XDECREF(self->py_data);
    self->py_data = NULL;
  }

  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

static PyObject *lorann_get_n_samples(LorannIndex *self, PyObject *args) {
  return PyLong_FromLong(static_cast<long>(self->index->get_n_samples()));
}

static PyObject *lorann_get_dim(LorannIndex *self, PyObject *args) {
  return PyLong_FromLong(static_cast<long>(self->index->get_dim()));
}

static PyObject *lorann_get_n_clusters(LorannIndex *self, PyObject *args) {
  return PyLong_FromLong(static_cast<long>(self->index->get_n_clusters()));
}

static PyObject *lorann_get_vector(LorannIndex *self, PyObject *args) {
  int idx;
  if (!PyArg_ParseTuple(args, "i", &idx)) return NULL;

  npy_intp dims[1] = {self->index->get_dim()};
  PyObject *vector = PyArray_SimpleNew(1, dims, NPY_FLOAT32);

  float *out = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)vector));

  self->index->get_vector(idx, out);
  return vector;
}

static PyObject *lorann_get_dissimilarity(LorannIndex *self, PyObject *args) {
  PyArrayObject *u;
  PyArrayObject *v;
  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &u, &PyArray_Type, &v)) return NULL;

  float *u_data = reinterpret_cast<float *>(PyArray_DATA(u));
  float *v_data = reinterpret_cast<float *>(PyArray_DATA(v));

  const float dissimilarity = self->index->get_dissimilarity(u_data, v_data);
  return PyFloat_FromDouble(static_cast<double>(dissimilarity));
}

static PyObject *lorann_build(LorannIndex *self, PyObject *args) {
  int approximate = 1;
  int n_threads = -1;
  PyArrayObject *Q = NULL;

  if (!PyArg_ParseTuple(args, "|iiO!", &approximate, &n_threads, &PyArray_Type, &Q)) return NULL;

#ifdef _OPENMP
  if (n_threads <= 0) {
    n_threads = omp_get_max_threads();
  }
#endif

  if (Q != NULL) {
    float *indata = reinterpret_cast<float *>(PyArray_DATA(Q));
    int n = PyArray_DIM(Q, 0);
    Py_BEGIN_ALLOW_THREADS;
    self->index->build(indata, n, approximate, n_threads);
    Py_END_ALLOW_THREADS;
  } else {
    Py_BEGIN_ALLOW_THREADS;
    self->index->build(approximate, n_threads);
    Py_END_ALLOW_THREADS;
  }

  Py_RETURN_NONE;
}

static PyObject *lorann_search(LorannIndex *self, PyObject *args) {
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

  float *indata = reinterpret_cast<float *>(PyArray_DATA(v));
  PyObject *nearest;

  if (PyArray_NDIM(v) == 1) {
    dim = PyArray_DIM(v, 0);

    npy_intp dims[1] = {k};
    nearest = PyArray_SimpleNew(1, dims, NPY_INT);
    int *out_idx = reinterpret_cast<int *>(PyArray_DATA((PyArrayObject *)nearest));

    if (return_distances) {
      PyObject *distances = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
      float *out_distances = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)distances));

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
      PyObject *distances = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
      float *out_distances = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)distances));

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

static PyObject *lorann_exact_search(LorannIndex *self, PyObject *args) {
  PyArrayObject *v;
  int k, dim, n, return_distances, n_threads;

  if (!PyArg_ParseTuple(args, "O!iii", &PyArray_Type, &v, &k, &return_distances, &n_threads))
    return NULL;

#ifdef _OPENMP
  if (n_threads <= 0) {
    n_threads = omp_get_max_threads();
  }
#endif

  float *indata = reinterpret_cast<float *>(PyArray_DATA(v));
  PyObject *nearest;

  if (PyArray_NDIM(v) == 1) {
    dim = PyArray_DIM(v, 0);

    npy_intp dims[1] = {k};
    nearest = PyArray_SimpleNew(1, dims, NPY_INT);
    int *out_idx = reinterpret_cast<int *>(PyArray_DATA((PyArrayObject *)nearest));

    if (return_distances) {
      PyObject *distances = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
      float *out_distances = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)distances));

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
      PyObject *distances = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
      float *out_distances = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)distances));

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

static PyObject *lorann_save(LorannIndex *self, PyObject *args) {
  const char *fname;
  if (!PyArg_ParseTuple(args, "s", &fname)) return NULL;

  try {
    std::ofstream output_file(fname, std::ios::binary);
    cereal::BinaryOutputArchive output_archive(output_file);
    output_archive(self->index);
  } catch (...) {
    PyErr_Format(PyExc_IOError, "Failed to write to file '%s'", fname);
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject *lorann_load(PyObject *cls, PyObject *args) {
  const char *fname;
  if (!PyArg_ParseTuple(args, "s", &fname)) return NULL;

  LorannIndex *new_obj =
      reinterpret_cast<LorannIndex *>(((PyTypeObject *)cls)->tp_alloc((PyTypeObject *)cls, 0));

  if (!new_obj) return NULL;

  try {
    std::ifstream input_file(fname, std::ios::binary);
    cereal::BinaryInputArchive input_archive(input_file);
    input_archive(new_obj->index);
  } catch (...) {
    PyErr_Format(PyExc_IOError, "Failed to load index from file '%s'", fname);
    return NULL;
  }

  new_obj->py_data = NULL;

  return (PyObject *)new_obj;
}

static PyObject *lorann_compute_V(PyObject *self, PyObject *args) {
  PyArrayObject *A;
  int rank, approximate = 1;

  if (!PyArg_ParseTuple(args, "O!i|i", &PyArray_Type, &A, &rank, &approximate)) return NULL;

  const int n = PyArray_DIM(A, 0);
  const int d = PyArray_DIM(A, 1);
  float *A_data = reinterpret_cast<float *>(PyArray_DATA(A));

  Eigen::MatrixXf Y = Eigen::Map<RowMatrix>(A_data, n, d);
  Eigen::MatrixXf V = Lorann::compute_V(Y, rank, approximate);

  PyObject *ret;
  npy_intp dims[2] = {V.cols(), V.rows()};
  ret = PyArray_SimpleNew(2, dims, NPY_FLOAT32);

  float *outdata = reinterpret_cast<float *>(PyArray_DATA((PyArrayObject *)ret));
  std::memcpy(outdata, V.data(), V.rows() * V.cols() * sizeof(float));

  return ret;
}

static PyMethodDef LorannMethods[] = {
    {"exact_search", (PyCFunction)lorann_exact_search, METH_VARARGS, ""},
    {"search", (PyCFunction)lorann_search, METH_VARARGS, ""},
    {"build", (PyCFunction)lorann_build, METH_VARARGS, ""},
    {"save", (PyCFunction)lorann_save, METH_VARARGS, ""},
    {"load", (PyCFunction)lorann_load, METH_VARARGS | METH_CLASS, ""},
    {"get_n_samples", (PyCFunction)lorann_get_n_samples, METH_NOARGS, ""},
    {"get_dim", (PyCFunction)lorann_get_dim, METH_NOARGS, ""},
    {"get_n_clusters", (PyCFunction)lorann_get_n_clusters, METH_NOARGS, ""},
    {"get_vector", (PyCFunction)lorann_get_vector, METH_VARARGS, ""},
    {"get_dissimilarity", (PyCFunction)lorann_get_dissimilarity, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyMethodDef KMeansMethods[] = {
    {"train", (PyCFunction)kmeans_train, METH_VARARGS, ""},
    {"get_n_clusters", (PyCFunction)kmeans_get_n_clusters, METH_NOARGS, ""},
    {"get_iters", (PyCFunction)kmeans_get_iters, METH_NOARGS, ""},
    {"get_euclidean", (PyCFunction)kmeans_get_euclidean, METH_NOARGS, ""},
    {"get_balanced", (PyCFunction)kmeans_get_balanced, METH_NOARGS, ""},
    {"get_centroids", (PyCFunction)kmeans_get_centroids, METH_VARARGS, ""},
    {"assign", (PyCFunction)kmeans_assign, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyTypeObject LorannIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0) "lorann.LorannIndex", /* tp_name */
    sizeof(LorannIndex),                                 /* tp_basicsize */
    0,                                                   /* tp_itemsize */
    (destructor)lorann_dealloc,                          /* tp_dealloc */
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
    "Lorann index object",                               /* tp_doc */
    0,                                                   /* tp_traverse */
    0,                                                   /* tp_clear */
    0,                                                   /* tp_richcompare */
    0,                                                   /* tp_weaklistoffset */
    0,                                                   /* tp_iter */
    0,                                                   /* tp_iternext */
    LorannMethods,                                       /* tp_methods */
    0,                                                   /* tp_members */
    0,                                                   /* tp_getset */
    0,                                                   /* tp_base */
    0,                                                   /* tp_dict */
    0,                                                   /* tp_descr_get */
    0,                                                   /* tp_descr_set */
    0,                                                   /* tp_dictoffset */
    (initproc)Lorann_init,                               /* tp_init */
    0,                                                   /* tp_alloc */
    Lorann_new,                                          /* tp_new */
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

static PyMethodDef module_methods[] = {
    {"compute_V", (PyCFunction)lorann_compute_V, METH_VARARGS, ""},
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

PyMODINIT_FUNC PyInit_lorannlib(void) {
  PyObject *m;
  if (PyType_Ready(&LorannIndexType) < 0) return NULL;
  if (PyType_Ready(&KMeansIndexType) < 0) return NULL;

  m = PyModule_Create(&moduledef);

  if (m == NULL) return NULL;

  import_array();

  Py_INCREF(&LorannIndexType);
  PyModule_AddObject(m, "LorannIndex", reinterpret_cast<PyObject *>(&LorannIndexType));

  Py_INCREF(&KMeansIndexType);
  PyModule_AddObject(m, "KMeans", reinterpret_cast<PyObject *>(&KMeansIndexType));

  return m;
}
