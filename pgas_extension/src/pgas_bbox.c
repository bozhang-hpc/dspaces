#include "pgas_bbox.h"

/* 
  transform n-D bbox to 1D bbox
  compute the number of 1D bbox 
*/
uint64_t bbox1D_num(struct bbox *bb)
{
    uint64_t n = 1;
    int ndims = bb->num_dims;
    int i;

    for(i=1; i < ndims; i++) {
        n = n * coord_dist(&bb->lb, &bb->ub, i);
    }

    return n;
}