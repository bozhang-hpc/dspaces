#include <Python.h>
#include <mpi4py/mpi4py.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL ds
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

#include <dspaces.h>
#include <dspaces-ops.h>
#include <dspaces-server.h>

#include <stdio.h>

PyObject *wrapper_dspaces_init(int rank)
{
    dspaces_client_t *clientp;

    import_array();
    import_mpi4py();

    clientp = malloc(sizeof(*clientp));

    dspaces_init(rank, clientp);

    PyObject *client = PyLong_FromVoidPtr((void *)clientp);

    return (client);
}

PyObject *wrapper_dspaces_init_mpi(PyObject *commpy)
{
    MPI_Comm *comm_p = NULL;
    dspaces_client_t *clientp;

    import_array();
    import_mpi4py();

    comm_p = PyMPIComm_Get(commpy);
    if(!comm_p) {
        return(NULL);
    }
    clientp = malloc(sizeof(*clientp));

    dspaces_init_mpi(*comm_p, clientp);

    PyObject *client = PyLong_FromVoidPtr((void *)clientp);

    return (client);
}

PyObject *wrapper_dspaces_init_wan(const char *listen_str, const char *conn, int rank)
{
    dspaces_client_t *clientp;

    import_array();
    import_mpi4py();

    clientp = malloc(sizeof(*clientp));

    dspaces_init_wan(listen_str, conn, rank, clientp);

    PyObject *client = PyLong_FromVoidPtr((void *)clientp);

    return (client);
}

PyObject *wrapper_dspaces_init_wan_mpi(const char *listen_str, const char *conn, PyObject *commpy)
{
    MPI_Comm *comm_p = NULL;
    dspaces_client_t *clientp;

    import_array();
    import_mpi4py();

    comm_p = PyMPIComm_Get(commpy);
    if(!comm_p) {
        return(NULL);
    }
    clientp = malloc(sizeof(*clientp));

    dspaces_init_wan_mpi(listen_str, conn, *comm_p, clientp);

    PyObject *client = PyLong_FromVoidPtr((void *)clientp);

    return(client);
}

PyObject *wrapper_dspaces_server_init(const char *listen_str, PyObject *commpy,
                                const char *conf)
{
    MPI_Comm *comm_p = NULL;
    dspaces_provider_t *serverp;

    import_array();
    import_mpi4py();

    comm_p = PyMPIComm_Get(commpy);
    if(!comm_p) {
        return(NULL);
    }
    serverp = malloc(sizeof(*serverp));

    dspaces_server_init(listen_str, *comm_p, conf, serverp);

    PyObject *server = PyLong_FromVoidPtr((void *)serverp);

    return (server);
}

void wrapper_dspaces_fini(PyObject *clientppy)
{
    dspaces_client_t *clientp = PyLong_AsVoidPtr(clientppy);

    dspaces_fini(*clientp);

    free(clientp);
}

void wrapper_dspaces_server_fini(PyObject *serverppy)
{
    dspaces_provider_t *serverp = PyLong_AsVoidPtr(serverppy);

    dspaces_server_fini(*serverp);

    free(serverp);
}

void wrapper_dspaces_kill(PyObject *clientppy)
{
    dspaces_client_t *clientp = PyLong_AsVoidPtr(clientppy);

    dspaces_kill(*clientp);
}

void wrapper_dspaces_put(PyObject *clientppy, PyObject *obj, const char *name,
                         int version, PyObject *offset)
{
    dspaces_client_t *clientp = PyLong_AsVoidPtr(clientppy);
    PyArrayObject *arr = (PyArrayObject *)obj;
    int size = PyArray_ITEMSIZE(arr);
    int ndim = PyArray_NDIM(arr);
    void *data = PyArray_DATA(arr);
    uint64_t lb[ndim];
    uint64_t ub[ndim];
    npy_intp *shape = PyArray_DIMS(arr);
    PyObject *item;
    int i;

    for(i = 0; i < ndim; i++) {
        item = PyTuple_GetItem(offset, i);
        lb[i] = PyLong_AsLong(item);
        ub[i] = lb[i] + ((long)shape[i] - 1);
    }
    dspaces_put(*clientp, name, version, size, ndim, lb, ub, data);

    return;
}

PyObject *wrapper_dspaces_get(PyObject *clientppy, const char *name,
                              int version, PyObject *lbt, PyObject *ubt,
                              PyObject *dtype, int timeout)
{
    dspaces_client_t *clientp = PyLong_AsVoidPtr(clientppy);
    int ndim = PyTuple_GET_SIZE(lbt);
    uint64_t lb[ndim];
    uint64_t ub[ndim];
    void *data;
    PyObject *item;
    PyObject *arr;
    PyArray_Descr *descr = PyArray_DescrNew((PyArray_Descr *)dtype);
    npy_intp dims[ndim];
    int i;

    for(i = 0; i < ndim; i++) {
        item = PyTuple_GetItem(lbt, i);
        lb[i] = PyLong_AsLong(item);
        item = PyTuple_GetItem(ubt, i);
        ub[i] = PyLong_AsLong(item);
        dims[i] = (ub[i] - lb[i]) + 1;
    }

    dspaces_aget(*clientp, name, version, ndim, lb, ub, &data, timeout);

    arr = PyArray_NewFromDescr(&PyArray_Type, descr, ndim, dims, NULL, data, 0,
                               NULL);

    return (arr);
}

void wrapper_dspaces_define_gdim(PyObject *clientppy, const char *name, PyObject *gdimt)
{
    dspaces_client_t *clientp = PyLong_AsVoidPtr(clientppy);
    int ndim = PyTuple_GET_SIZE(gdimt);
    uint64_t gdim[ndim];
    PyObject *item;
    int i;

    for(i = 0; i < ndim; i++) {
        item = PyTuple_GetItem(gdimt, i);
        gdim[i] = PyLong_AsLong(item);    
    }

    dspaces_define_gdim(*clientp, name, ndim, gdim);
}

PyObject *wrapper_dspaces_ops_new_iconst(long val)
{
    ds_expr_t *exprp;

    exprp = malloc(sizeof(*exprp));

    *exprp = dspaces_op_new_iconst(val);

    PyObject *expr = PyLong_FromVoidPtr((void *)exprp);

    return(expr);
}

PyObject *wrapper_dspaces_ops_new_rconst(double val)
{
    ds_expr_t *exprp;

    exprp = malloc(sizeof(*exprp));

    *exprp = dspaces_op_new_rconst(val);

    PyObject *expr = PyLong_FromVoidPtr((void *)exprp);

    return(expr);
}

PyObject *wrapper_dspaces_ops_new_obj(PyObject *clientppy, const char *name, int version, PyObject *lbt, PyObject *ubt, PyObject *dtype)
{
    dspaces_client_t *clientp = PyLong_AsVoidPtr(clientppy);
    ds_expr_t *exprp;
    int ndim = PyTuple_GET_SIZE(lbt);
    uint64_t lb[ndim];
    uint64_t ub[ndim];
    PyObject *item, *item_utf, *expr;
    char *type_str;
    int val_type;
    int i;

    item = PyObject_GetAttrString(dtype, "__name__");
    item_utf = PyUnicode_EncodeLocale(item, "strict");
    type_str = PyBytes_AsString(item_utf);

    if(strcmp(type_str, "float") == 0) {
        val_type = DS_VAL_REAL;
    } else if(strcmp(type_str, "int") == 0) {
        val_type = DS_VAL_INT;
    } else {
        PyErr_SetString(PyExc_TypeError, "type must be int or float");
        return(NULL);
    }

    for(i = 0; i < ndim; i++) {
        item = PyTuple_GetItem(lbt, i);
        lb[i] = PyLong_AsLong(item);
        item = PyTuple_GetItem(ubt, i);
        ub[i] = PyLong_AsLong(item);
    }

    exprp = malloc(sizeof(*exprp));
    *exprp = dspaces_op_new_obj(*clientp, name, version, val_type, ndim, lb, ub);
    expr = PyLong_FromVoidPtr((void *)exprp);

    return(expr);
}

PyObject *wrapper_dspaces_op_new_add(PyObject *exprppy1, PyObject *exprppy2)
{
    ds_expr_t *exprp1, *exprp2, *resp;
    PyObject *res;

    exprp1 = PyLong_AsVoidPtr(exprppy1);
    exprp2 = PyLong_AsVoidPtr(exprppy2);

    resp = malloc(sizeof(*resp));
    *resp = dspaces_op_new_add(*exprp1, *exprp2);
    res = PyLong_FromVoidPtr((void *)resp);

    return(res);
}

PyObject *wrapper_dspaces_ops_calc(PyObject *clientppy, PyObject *exprppy)
{
    void *result_buf;
    dspaces_client_t *clientp = PyLong_AsVoidPtr(clientppy);
    ds_expr_t *exprp = PyLong_AsVoidPtr(exprppy);
    int typenum;
    int ndim;
    uint64_t *dims;
    ds_val_t etype;
    npy_intp *array_dims;
    long int_res;
    double real_res;
    PyObject *arr;
    int i;

    dspaces_op_calc(*clientp, *exprp, &result_buf);
    dspaces_op_get_result_size(*exprp, &ndim, &dims);
    etype = dspaces_op_get_result_type(*exprp);
    if(ndim == 0) {
        if(etype == DS_VAL_INT) {
            int_res = *(long *)result_buf;
            free(result_buf);
            return(PyLong_FromLong(int_res));
        } else if(etype == DS_VAL_REAL) {
            real_res = *(double *)result_buf;
            free(result_buf);
            return(PyFloat_FromDouble(real_res));
        } else {
            PyErr_SetString(PyExc_TypeError, "invalid type assigned to expression (corruption?)");
            return(NULL);
        }
    }
    array_dims = malloc(sizeof(*array_dims) * ndim);
    for(i = 0; i < ndim; i++) {
        array_dims[i] = dims[i];
    }
    free(dims);
    if(etype == DS_VAL_INT) {
        typenum = NPY_INT64;
    } else if(etype == DS_VAL_REAL) {
        typenum = NPY_FLOAT64;
    } else {
        PyErr_SetString(PyExc_TypeError, "invalid type assigned to expression (corruption?)");
        return(NULL);
    }
    arr = PyArray_SimpleNewFromData(ndim, array_dims, typenum, result_buf);
    return(arr);
}
