#define idx2_Implementation
#include <idx2.hpp>

#include <cstdint>

struct idx_file {
    idx2::idx2_file Idx2;
};

struct idx_params {
    idx2::params Params;
};

struct idx_buffer {
    idx2::buffer Buffer;
};

struct idx_grid {
    idx_grid(const idx2::idx2_file &Idx2, const idx2::params &P)
    {
        Grid = idx2::GetOutputGrid(Idx2, P);
    }
    idx2::grid Grid;
};

struct idx_extent {
    idx_extent(idx2::extent Extent_) : Extent(Extent_) {}
    idx_extent(const idx2::v3i &Dim3) { Extent = idx2::extent(Dim3); }
    idx2::extent Extent;
};

extern "C" struct idx_file *idx_init_c(struct idx_params *p)
{
    struct idx_file *idx2 = new struct idx_handle();

    idx2::error Err = idx2::Init(&idx2->Idx2, p->Params);

    if(ErrorExists(Err)) {
        Dealloc(&idx2->Idx2);
        delete idx2;
        return (NULL);
    } else {
        return (idx2);
    }
};

extern "C" int idx2_get_dtype_size(struct idx_handle *idx2)
{
    return (idx2::SizeOf(idx2->Idx2.DType));
}

extern "C" struct idx_grid *
idx2_get_output_grid_c(const struct idx_handle *idx2,
                       const struct idx_params *p)
{
    return (new struct idx_grid(idx2->Idx2, p->Params));
}

extern "C" int64_t idx2_get_grid_volume(struct idx_grid *grid)
{
    return (idx2::Prod<idx2::i64>(idx2::Dims(grid->Grid)));
}

extern "C" int idx2_decode_c(const struct idx_handle *idx2,
                             const struct idx_params *p,
                             struct idx_buffer *out_buf)
{
    idx2::error Err = idx2::Decode(idx2->Idx2, p->Params, &out_buf->Buffer);
    if(ErrorExists(Err)) {
        return (-1);
    }
    return (0);
}

extern "C" struct idx_extent *idx2_extent_from_triple(int32_t x, int32_t y,
                                                      int32_t z)
{
    idx2::v3i Dims3(x, y, z);
    return (new struct idx_extent(Dims3));
}

extern "C" struct idx_extent *idx2_bounding_box(struct idx_extent *ext1,
                                                struct idx_extent *ext2)
{
    return (
        new struct idx_extent(idx2::BoundingBox(ext1->Extent, ext2->Extent)));
}

extern "C" struct idx_params *idx2_new_params()
{
    return (new struct idx_params());
}

extern "C" void idx2_param_set_input_file(struct idx_params *p,
                                          const char *infile)
{
    p->Params.InputFile = strdup(infile);
}

extern "C" void idx2_param_set_in_dir(struct idx_params *p, const char *indir)
{
    p->Params.InDir = strdup(indir);
}

extern "C" void idx2_param_set_downsampling_factor(struct idx_params *p,
                                                   int32_t x, int32_t y,
                                                   int32_t z)
{
    idx2::v3i Factor3(x, y, z);
    p->Params.DownsamplingFactor3 = Factor3;
}

extern "C" void idx2_param_set_decode_accuracy(struct idx_params *p,
                                               double accuracy)
{
    p->Params.DecodeAccuracy = accuracy;
}

extern "C" void idx2_param_set_extent(struct idx_params *p,
                                      struct idx_extent *extent)
{
    p->Params.DecodeExtent = extent->Extent;
}

extern "C" void idx2_free_params(struct idx_params *p) { delete p; }

extern "C" int idx2_destroy_c(struct idx_handle *idx2)
{
    idx2::error Err = idx2::Destroy(&idx2->Idx2);
    if(ErrorExists(Err)) {
        return (-1);
    }
    delete idx2;
    return (0);
}

extern "C" void idx2_free_grid(struct idx_grid *grid) { delete grid; }

extern "C" void idx2_free_extent(struct idx_extent *extent) { delete extent; }
