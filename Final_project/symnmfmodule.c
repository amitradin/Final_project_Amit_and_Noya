/*
 * Python C extension: exposes sym, ddg, norm, symnmf to Python.
 * Converts list-of-lists to/from double** and calls the symnmf.c implementations.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"
#include "matrix_ops.h"
#include <stdlib.h>

/* Copy Python list of lists into a C matrix; caller frees with free_matrix. */
static double **py_to_matrix(PyObject *py_mat, int *rows, int *cols)
{
    PyObject *row_obj;
    Py_ssize_t i, j, n, m;
    double **M;

    if (!PyList_Check(py_mat)) return NULL;
    n = PyList_Size(py_mat);
    if (n < 1) return NULL;
    row_obj = PyList_GetItem(py_mat, 0);
    if (!PyList_Check(row_obj)) return NULL;
    m = PyList_Size(row_obj);
    if (m < 1) return NULL;
    *rows = (int)n;
    *cols = (int)m;
    M = allocate_matrix((int)n, (int)m);
    if (M == NULL) return NULL;
    for (i = 0; i < n; i++) {
        row_obj = PyList_GetItem(py_mat, i);
        if (!PyList_Check(row_obj) || PyList_Size(row_obj) != m) {
            free_matrix(M, (int)n);
            return NULL;
        }
        for (j = 0; j < m; j++) {
            PyObject *val = PyList_GetItem(row_obj, j);
            M[i][j] = PyFloat_AsDouble(val);
            if (PyErr_Occurred()) {
                free_matrix(M, (int)n);
                return NULL;
            }
        }
    }
    return M;
}

/* Turn a C matrix into a Python list of lists. */
static PyObject *matrix_to_py(double **M, int rows, int cols)
{
    PyObject *result, *row;
    int i, j;

    result = PyList_New(rows);
    if (result == NULL) return NULL;
    for (i = 0; i < rows; i++) {
        row = PyList_New(cols);
        if (row == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        for (j = 0; j < cols; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(M[i][j]));
        }
        PyList_SetItem(result, i, row);
    }
    return result;
}

/* Python entry points: each unpacks args, calls C, packs result back. */
static PyObject *symnmf_sym(PyObject *self, PyObject *args)
{
    PyObject *py_points, *result_obj;
    double **points = NULL, **A = NULL;
    int n, d;

    (void)self;
    if (!PyArg_ParseTuple(args, "O", &py_points)) return NULL;
    points = py_to_matrix(py_points, &n, &d);
    if (points == NULL) { PyErr_SetString(PyExc_ValueError, "invalid points"); return NULL; }
    A = sym((const double **)points, n, d);
    free_matrix(points, n);
    if (A == NULL) { PyErr_NoMemory(); return NULL; }
    result_obj = matrix_to_py(A, n, n);
    free_matrix(A, n);
    return result_obj;
}

static PyObject *symnmf_ddg(PyObject *self, PyObject *args)
{
    PyObject *py_points, *result_obj;
    double **points = NULL, **D = NULL;
    int n, d;

    (void)self;
    if (!PyArg_ParseTuple(args, "O", &py_points)) return NULL;
    points = py_to_matrix(py_points, &n, &d);
    if (points == NULL) { PyErr_SetString(PyExc_ValueError, "invalid points"); return NULL; }
    D = ddg((const double **)points, n, d);
    free_matrix(points, n);
    if (D == NULL) { PyErr_NoMemory(); return NULL; }
    result_obj = matrix_to_py(D, n, n);
    free_matrix(D, n);
    return result_obj;
}

static PyObject *symnmf_norm(PyObject *self, PyObject *args)
{
    PyObject *py_points, *result_obj;
    double **points = NULL, **W = NULL;
    int n, d;

    (void)self;
    if (!PyArg_ParseTuple(args, "O", &py_points)) return NULL;
    points = py_to_matrix(py_points, &n, &d);
    if (points == NULL) { PyErr_SetString(PyExc_ValueError, "invalid points"); return NULL; }
    W = norm((const double **)points, n, d);
    free_matrix(points, n);
    if (W == NULL) { PyErr_NoMemory(); return NULL; }
    result_obj = matrix_to_py(W, n, n);
    free_matrix(W, n);
    return result_obj;
}

static PyObject *symnmf_symnmf(PyObject *self, PyObject *args)
{
    PyObject *py_H, *py_W, *result_obj;
    double **H = NULL, **W = NULL;
    int n, k;
    double epsilon;
    int max_iter;

    (void)self;
    if (!PyArg_ParseTuple(args, "OOdi", &py_H, &py_W, &epsilon, &max_iter)) return NULL;
    H = py_to_matrix(py_H, &n, &k);
    if (H == NULL) { PyErr_SetString(PyExc_ValueError, "invalid H"); return NULL; }
    W = py_to_matrix(py_W, &n, &n);
    if (W == NULL) { free_matrix(H, n); PyErr_SetString(PyExc_ValueError, "invalid W"); return NULL; }
    symnmf_run(W, H, n, k, epsilon, max_iter);
    free_matrix(W, n);
    result_obj = matrix_to_py(H, n, k);
    free_matrix(H, n);
    return result_obj;
}

static PyMethodDef symnmf_methods[] = {
    {"sym", symnmf_sym, METH_VARARGS, "Compute similarity matrix A."},
    {"ddg", symnmf_ddg, METH_VARARGS, "Compute diagonal degree matrix D."},
    {"norm", symnmf_norm, METH_VARARGS, "Compute normalized similarity matrix W."},
    {"symnmf", symnmf_symnmf, METH_VARARGS, "Run SymNMF update on W and H."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "_symnmf",
    NULL,
    -1,
    symnmf_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__symnmf(void)
{
    return PyModule_Create(&symnmfmodule);
}
