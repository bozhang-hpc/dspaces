#ifndef _IDX2_WRAPPER_
#define _IDX2_WRAPPER_

#include <margo.h>
#include <mercury.h>
#include <mercury_atomic.h>
#include <mercury_bulk.h>
#include <mercury_macros.h>
#include <mercury_proc_string.h>

struct idx_file;
struct idx_params;
struct idx_buffer;
struct idx_grid;
struct idx_extent;

typedef struct {
    size_t idxp_size;
    size_t gdim_size;
    char *raw_idxp;
    char *raw_gdim;

} idxp_hdr_with_gdim;

static inline hg_return_t hg_proc_idxp_hdr_with_gdim(hg_proc_t proc, void *arg)
{
    hg_return_t ret;
    idxp_hdr_with_gdim *in = (idxp_hdr_with_gdim *)arg;
    ret = hg_proc_hg_size_t(proc, &in->idxp_size);
    ret = hg_proc_hg_size_t(proc, &in->gdim_size);
    if(ret != HG_SUCCESS)
        return ret;
    if(in->idxp_size) {
        switch(hg_proc_get_op(proc)) {
        case HG_ENCODE:
            ret = hg_proc_raw(proc, in->raw_idxp, in->idxp_size);
            if(ret != HG_SUCCESS)
                return ret;
            ret = hg_proc_raw(proc, in->raw_gdim, in->gdim_size);
            if(ret != HG_SUCCESS)
                return ret;
            break;
        case HG_DECODE:
            in->raw_idxp = (char *)malloc(in->idxp_size);
            ret = hg_proc_raw(proc, in->raw_idxp, in->idxp_size);
            if(ret != HG_SUCCESS)
                return ret;
            in->raw_gdim = (char *)malloc(in->gdim_size);
            ret = hg_proc_raw(proc, in->raw_gdim, in->gdim_size);
            if(ret != HG_SUCCESS)
                return ret;
            break;
        case HG_FREE:
            free(in->raw_idxp);
            free(in->raw_gdim);
            break;
        default:
            break;
        }
    }
    return HG_SUCCESS;
}

MERCURY_GEN_PROC(idxp_gdim_t, ((idxp_hdr_with_gdim)(idxp_gdim))((int32_t)(param)))

idx_file_t idx_init_c(struct idx_params *p);
int idx2_get_dtype_size(idx_handle_t idx2);
idx_grid_t idx2_get_output_grid_c(const idx_handle_t idx2,
                                    const struct idx_params *p);
int64_t idx2_get_grid_volume(idx_grid_t grid);
int idx2_decode_c(const idx_handle_t idx2, const struct idx_params *p,
                  idx_buffer_t out_buf);
idx_extent_t idx2_extent_from_triple(int32_t x, int32_t y, int32_t z);
idx_extent_t idx2_bounding_box(idx_extent_t ext1,
                                idx_extent_t ext2);
struct idx_params* idx2_new_params();
void idx2_param_set_input_file(struct idx_params *p,
                                const char *infile);
void idx2_param_set_in_dir(struct idx_params *p, const char *indir);
void idx2_param_set_downsampling_factor(struct idx_params *p, int32_t x,
                                        int32_t y, int32_t z);
void idx2_param_set_decode_accuracy(sturct idx_params *p, double accuracy);
void idx2_param_set_extent(struct idx_params *p, idx_extent_t extent);
void idx2_free_params(struct idx_params *p);
int idx2_destroy_c(idx_handle_t idx2);
void idx2_free_grid(idx_grid_t grid);
void idx2_free_extent(idx_extent_t extent);

#endif
