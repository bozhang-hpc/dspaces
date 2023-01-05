#ifndef _IDX1_WRAPPER_
#define _IDX1_WRAPPER_

#include <margo.h>
#include <mercury.h>
#include <mercury_atomic.h>
#include <mercury_bulk.h>
#include <mercury_macros.h>
#include <mercury_proc_string.h>

typedef struct {
    char dir[128];
    char filename[128];
    int resolution;
    // idx1 has up to 5D support
    int lb[5];
    int ub[5];
    // maybe add compression params later
} idx1_params;

typedef struct {
    size_t idx1p_size;
    size_t gdim_size;
    char *raw_idx1p;
    char *raw_gdim;

} idx1p_hdr_with_gdim;

static inline hg_return_t hg_proc_idx1p_hdr_with_gdim(hg_proc_t proc, void *arg)
{
    hg_return_t ret;
    idx1p_hdr_with_gdim *in = (idx1p_hdr_with_gdim *)arg;
    ret = hg_proc_hg_size_t(proc, &in->idx1p_size);
    ret = hg_proc_hg_size_t(proc, &in->gdim_size);
    if(ret != HG_SUCCESS)
        return ret;
    if(in->idx1p_size) {
        switch(hg_proc_get_op(proc)) {
        case HG_ENCODE:
            ret = hg_proc_raw(proc, in->raw_idx1p, in->idx1p_size);
            if(ret != HG_SUCCESS)
                return ret;
            ret = hg_proc_raw(proc, in->raw_gdim, in->gdim_size);
            if(ret != HG_SUCCESS)
                return ret;
            break;
        case HG_DECODE:
            in->raw_idx1p = (char *)malloc(in->idx1p_size);
            ret = hg_proc_raw(proc, in->raw_idx1p, in->idx1p_size);
            if(ret != HG_SUCCESS)
                return ret;
            in->raw_gdim = (char *)malloc(in->gdim_size);
            ret = hg_proc_raw(proc, in->raw_gdim, in->gdim_size);
            if(ret != HG_SUCCESS)
                return ret;
            break;
        case HG_FREE:
            free(in->raw_idx1p);
            free(in->raw_gdim);
            break;
        default:
            break;
        }
    }
    return HG_SUCCESS;
}

MERCURY_GEN_PROC(idx1p_gdim_t, ((idx1p_hdr_with_gdim)(idx1p_gdim))((int32_t)(param)))

void* read_idx1(char* filepath, char* fieldname, int ndims,
                size_t elemsize, uint64_t* lb, uint64_t* ub, 
                unsigned int ts, int resolution);

#endif