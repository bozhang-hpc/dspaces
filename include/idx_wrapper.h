#ifndef _IDX2_WRAPPER_
#define _IDX2_WRAPPER_

struct idx_file;
struct idx_params;
struct idx_buffer;
struct idx_grid;
struct idx_extent;

struct idx_file *idx_init_c(struct idx_params *p);
int idx2_get_dtype_size(struct idx_handle *idx2);
struct idx_grid *idx2_get_output_grid_c(const struct idx_handle *idx2,
                                        const struct idx_params *p);
int64_t idx2_get_grid_volume(struct idx_grid *grid);
int idx2_decode_c(const struct idx_handle *idx2, const struct idx_params *p,
                  struct idx_buffer *out_buf);
struct idx_extent *idx2_extent_from_triple(int32_t x, int32_t y, int32_t z);
struct idx_extent *idx2_bounding_box(struct idx_extent *ext1,
                                     struct idx_extent *ext2);
struct idx_params *
idx2_new_params() void idx2_param_set_input_file(struct idx_params *p,
                                                 const char *infile);
void idx2_param_set_in_dir(struct idx_params *p, const char *indir);
void idx2_param_set_downsampling_factor(struct idx_params *p, int32_t x,
                                        int32_t y, int32_t z);
void idx2_param_set_decode_accuracy(struct idx_params *p, double accuracy);
void idx2_param_set_extent(struct idx_params *p, struct idx_extent *extent);
void idx2_free_params(struct idx_params *p);
int idx2_destroy_c(struct idx_handle *idx2);
void idx2_free_grid(struct idx_grid *grid);
void idx2_free_extent(struct idx_extent *extent);

#endif
