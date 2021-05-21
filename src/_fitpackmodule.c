/*
    Multipack project.
    This file is generated by setmodules.py. Do not modify it.
 */

#include <Python.h>
#include "numpy/arrayobject.h"

#define PyInt_AsLong PyLong_AsLong

static PyObject *fitpack_error;
#include "__fitpack.h"

#ifdef HAVE_ILP64

#define F_INT npy_int64
#define F_INT_NPY NPY_INT64

#if NPY_BITSOF_SHORT == 64
#define F_INT_PYFMT   "h"
#elif NPY_BITSOF_INT == 64
#define F_INT_PYFMT   "i"
#elif NPY_BITSOF_LONG == 64
#define F_INT_PYFMT   "l"
#elif NPY_BITSOF_LONGLONG == 64
#define F_INT_PYFMT   "L"
#else
#error No compatible 64-bit integer size. \
       Please contact NumPy maintainers and give detailed information about your \
       compiler and platform, or set NPY_USE_BLAS64_=0
#endif

#else

#define F_INT int
#define F_INT_NPY NPY_INT
#define F_INT_PYFMT   "i"

#endif


/*
 * Functions moved verbatim from __fitpack.h
 */


/*
 * Python-C wrapper of FITPACK (by P. Dierckx) (in netlib known as dierckx)
 * Author: Pearu Peterson <pearu@ioc.ee>
 * June 1.-4., 1999
 * June 7. 1999
 * $Revision$
 * $Date$
 */

/*  module_methods:
 * {"_spl_", fitpack_spl_, METH_VARARGS, doc_spl_},
 * {"_parcur", fitpack_parcur, METH_VARARGS, doc_parcur},
 */

/* link libraries: (one item per line)
   ddierckx
 */
/* python files: (to be imported to Multipack.py)
   fitpack.py
 */

#if defined(UPPERCASE_FORTRAN)
	#if defined(NO_APPEND_FORTRAN)
	/* nothing to do */
	#else
		#define SPLDER SPLDER_
		#define SPLEV  SPLEV_
		#define PARCUR PARCUR_
		#define CLOCUR CLOCUR_
	#endif
#else
	#if defined(NO_APPEND_FORTRAN)
		#define SPLDER splder
		#define SPLEV splev
		#define PARCUR parcur
		#define CLOCUR clocur
	#else
		#define SPLDER splder_
		#define SPLEV splev_
		#define PARCUR parcur_
		#define CLOCUR clocur_
	#endif
#endif

void SPLDER(double*,F_INT*,double*,F_INT*,F_INT*,double*,
        double*,F_INT*,F_INT*,double*,F_INT*);
void SPLEV(double*,F_INT*,double*,F_INT*,double*,double*,F_INT*,F_INT*,F_INT*);
void PARCUR(F_INT*,F_INT*,F_INT*,F_INT*,double*,F_INT*,double*,
        double*,double*,double*,F_INT*,double*,F_INT*,F_INT*,
        double*,F_INT*,double*,double*,double*,F_INT*,F_INT*,F_INT*);
void CLOCUR(F_INT*,F_INT*,F_INT*,F_INT*,double*,F_INT*,double*,
        double*,F_INT*,double*,F_INT*,F_INT*,double*,F_INT*,
        double*,double*,double*,F_INT*,F_INT*,F_INT*);

/* Note that curev, cualde need no interface. */

static char doc_parcur[] = " [t,c,o] = _parcur(x,w,u,ub,ue,k,iopt,ipar,s,t,nest,wrk,iwrk,per)";
static PyObject *
fitpack_parcur(PyObject *dummy, PyObject *args)
{
    F_INT k, iopt, ipar, nest, *iwrk, idim, m, mx, no=0, nc, ier, lwa, lwrk, i, per;
    F_INT n=0,  lc;
    npy_intp dims[1];
    double *x, *w, *u, *c, *t, *wrk, *wa=NULL, ub, ue, fp, s;
    PyObject *x_py = NULL, *u_py = NULL, *w_py = NULL, *t_py = NULL;
    PyObject *wrk_py=NULL, *iwrk_py=NULL;
    PyArrayObject *ap_x = NULL, *ap_u = NULL, *ap_w = NULL, *ap_t = NULL, *ap_c = NULL;
    PyArrayObject *ap_wrk = NULL, *ap_iwrk = NULL;

    if (!PyArg_ParseTuple(args, ("OOOdd" F_INT_PYFMT F_INT_PYFMT F_INT_PYFMT
                                 "dO" F_INT_PYFMT "OO" F_INT_PYFMT),
                          &x_py, &w_py, &u_py, &ub, &ue, &k, &iopt, &ipar,
                          &s, &t_py, &nest, &wrk_py, &iwrk_py, &per)) {
        return NULL;
    }
    ap_x = (PyArrayObject *)PyArray_ContiguousFromObject(x_py, NPY_DOUBLE, 0, 1);
    ap_u = (PyArrayObject *)PyArray_ContiguousFromObject(u_py, NPY_DOUBLE, 0, 1);
    ap_w = (PyArrayObject *)PyArray_ContiguousFromObject(w_py, NPY_DOUBLE, 0, 1);
    ap_wrk=(PyArrayObject *)PyArray_ContiguousFromObject(wrk_py, NPY_DOUBLE, 0, 1);
    ap_iwrk=(PyArrayObject *)PyArray_ContiguousFromObject(iwrk_py, F_INT_NPY, 0, 1);
    if (ap_x == NULL
            || ap_u == NULL
            || ap_w == NULL
            || ap_wrk == NULL
            || ap_iwrk == NULL) {
        goto fail;
    }
    x = (double *) ap_x->data;
    u = (double *) ap_u->data;
    w = (double *) ap_w->data;
    m = ap_w->dimensions[0];
    mx = ap_x->dimensions[0];
    idim = mx/m;
    if (per) {
        lwrk = m*(k + 1) + nest*(7 + idim + 5*k);
    }
    else {
        lwrk = m*(k + 1) + nest*(6 + idim + 3*k);
    }
    nc = idim*nest;
    lwa = nc + 2*nest + lwrk;
    if ((wa = malloc(lwa*sizeof(double))) == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    t = wa;
    c = t + nest;
    wrk = c + nc;
    iwrk = (F_INT *)(wrk + lwrk);
    if (iopt) {
        ap_t=(PyArrayObject *)PyArray_ContiguousFromObject(t_py, NPY_DOUBLE, 0, 1);
        if (ap_t == NULL) {
            goto fail;
        }
        n = no = ap_t->dimensions[0];
        memcpy(t, ap_t->data, n*sizeof(double));
        Py_DECREF(ap_t);
        ap_t = NULL;
    }
    if (iopt == 1) {
        memcpy(wrk, ap_wrk->data, n*sizeof(double));
        memcpy(iwrk, ap_iwrk->data, n*sizeof(F_INT));
    }
    if (per) {
        CLOCUR(&iopt, &ipar, &idim, &m, u, &mx, x, w, &k, &s, &nest,
                &n, t, &nc, c, &fp, wrk, &lwrk, iwrk, &ier);
    }
    else {
        PARCUR(&iopt, &ipar, &idim, &m, u, &mx, x, w, &ub, &ue, &k,
                &s, &nest, &n, t, &nc, c, &fp, wrk, &lwrk, iwrk, &ier);
    }
    if (ier == 10) {
        PyErr_SetString(PyExc_ValueError, "Invalid inputs.");
        goto fail;
    }
    if (ier > 0 && n == 0) {
        n = 1;
    }
    lc = (n - k - 1)*idim;
    dims[0] = n;
    ap_t = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    dims[0] = lc;
    ap_c = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (ap_t == NULL || ap_c == NULL) {
        goto fail;
    }
    if (iopt != 1|| n > no) {
        Py_XDECREF(ap_wrk);
        ap_wrk = NULL;
        Py_XDECREF(ap_iwrk);
        ap_iwrk = NULL;

        dims[0] = n;
        ap_wrk = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (ap_wrk == NULL) {
            goto fail;
        }
        ap_iwrk = (PyArrayObject *)PyArray_SimpleNew(1, dims, F_INT_NPY);
        if (ap_iwrk == NULL) {
            goto fail;
        }
    }
    memcpy(ap_t->data, t, n*sizeof(double));
    for (i = 0; i < idim; i++)
        memcpy((double *)ap_c->data + i*(n - k - 1), c + i*n, (n - k - 1)*sizeof(double));
    memcpy(ap_wrk->data, wrk, n*sizeof(double));
    memcpy(ap_iwrk->data, iwrk, n*sizeof(F_INT));
    free(wa);
    Py_DECREF(ap_x);
    Py_DECREF(ap_w);
    return Py_BuildValue(("NN{s:N,s:d,s:d,s:N,s:N,s:" F_INT_PYFMT ",s:d}"), PyArray_Return(ap_t),
            PyArray_Return(ap_c), "u", PyArray_Return(ap_u), "ub", ub, "ue", ue,
            "wrk", PyArray_Return(ap_wrk), "iwrk", PyArray_Return(ap_iwrk),
            "ier", ier, "fp",fp);
fail:
    free(wa);
    Py_XDECREF(ap_x);
    Py_XDECREF(ap_u);
    Py_XDECREF(ap_w);
    Py_XDECREF(ap_t);
    Py_XDECREF(ap_wrk);
    Py_XDECREF(ap_iwrk);
    return NULL;
}

static char doc_spl_[] = " [y,ier] = _spl_(x,nu,t,c,k,e)";
static PyObject *
fitpack_spl_(PyObject *dummy, PyObject *args)
{
    F_INT n, nu, ier, k, m, e=0;
    npy_intp dims[1];
    double *x, *y, *t, *c, *wrk = NULL;
    PyArrayObject *ap_x = NULL, *ap_y = NULL, *ap_t = NULL, *ap_c = NULL;
    PyObject *x_py = NULL, *t_py = NULL, *c_py = NULL;

    if (!PyArg_ParseTuple(args, ("O" F_INT_PYFMT "OO" F_INT_PYFMT F_INT_PYFMT),
                          &x_py, &nu, &t_py, &c_py, &k, &e)) {
        return NULL;
    }
    ap_x = (PyArrayObject *)PyArray_ContiguousFromObject(x_py, NPY_DOUBLE, 0, 1);
    ap_t = (PyArrayObject *)PyArray_ContiguousFromObject(t_py, NPY_DOUBLE, 0, 1);
    ap_c = (PyArrayObject *)PyArray_ContiguousFromObject(c_py, NPY_DOUBLE, 0, 1);
    if ((ap_x == NULL || ap_t == NULL || ap_c == NULL)) {
        goto fail;
    }
    x = (double *)ap_x->data;
    m = ap_x->dimensions[0];
    t = (double *)ap_t->data;
    c = (double *)ap_c->data;
    n = ap_t->dimensions[0];
    dims[0] = m;
    ap_y = (PyArrayObject *)PyArray_SimpleNew(1,dims,NPY_DOUBLE);
    if (ap_y == NULL) {
        goto fail;
    }
    y = (double *)ap_y->data;
    if ((wrk = malloc(n*sizeof(double))) == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    if (nu) {
        SPLDER(t, &n, c, &k, &nu, x, y, &m, &e, wrk, &ier);
    }
    else {
        SPLEV(t, &n, c, &k, x, y, &m, &e, &ier);
    }
    free(wrk);
    Py_DECREF(ap_x);
    Py_DECREF(ap_c);
    Py_DECREF(ap_t);
    return Py_BuildValue(("N" F_INT_PYFMT), PyArray_Return(ap_y), ier);

fail:
    free(wrk);
    Py_XDECREF(ap_x);
    Py_XDECREF(ap_c);
    Py_XDECREF(ap_t);
    return NULL;
}

/* End of functions moved verbatim from __fitpack.h */



static struct PyMethodDef fitpack_module_methods[] = {
{"_spl_",
    fitpack_spl_,
    METH_VARARGS, doc_spl_},
{"_parcur",
    fitpack_parcur,
    METH_VARARGS, doc_parcur},
{NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_fitpack",
    NULL,
    -1,
    fitpack_module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyObject *PyInit__fitpack(void)
{
    PyObject *m, *d, *s;

    m = PyModule_Create(&moduledef);
    import_array();

    d = PyModule_GetDict(m);

    s = PyUnicode_FromString(" 1.7 ");
    PyDict_SetItemString(d, "__version__", s);
    fitpack_error = PyErr_NewException ("fitpack.error", NULL, NULL);
    Py_DECREF(s);
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module fitpack");
    }

    return m;
}