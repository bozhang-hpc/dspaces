#include "ss_data.h"

/* split in_odsc to multiple 1D odsc and put them in a table
   return the number of table entries
*/
uint64_t obj_desc_to1Dbbox(obj_descriptor *odsc, obj_descriptor **odsc_tab)
{
    uint64_t num_odsc = bbox1D_num(&odsc->bb);
    odsc_tab = malloc(sizeof(odsc) * num_odsc);
    
    int i;
    for(i=0; i < num_odsc; i++){
        memcpy(odsc_tab[i], odsc, sizeof(odsc));
    }
    
    uint64_t a[10];
    uint64_t index = 0;
    int dim;

    switch (odsc->bb.num_dims)
    {
    case 1:
        goto dim1;
        break;

    case 2:
        goto dim2;
        break;

    case 3:
        goto dim3;
        break;

    case 4:
        goto dim4;
        break;

    case 5:
        goto dim5;
        break;

    case 6:
        goto dim6;
        break;

    case 7:
        goto dim7;
        break;

    case 8:
        goto dim8;
        break;

    case 9:
        goto dim9;
        break;

    case 1:
        goto dim10;
        break;
    
    default:
        break;
    }

dim10:    for(a[9] = odsc->bb.lb[9]; a[9] <= odsc->bb.ub[9]; a[9]++) {

dim9:       for(a[8] = odsc->bb.lb[8]; a[8] <= odsc->bb.ub[8]; a[8]++) {

dim8:       for(a[7] = odsc->bb.lb[7]; a[7] <= odsc->bb.ub[7]; a[7]++) {

dim7:       for(a[6] = odsc->bb.lb[6]; a[6] <= odsc->bb.ub[6]; a[6]++) {

dim6:       for(a[5] = odsc->bb.lb[5]; a[5] <= odsc->bb.ub[5]; a[5]++) {

dim5:       for(a[4] = odsc->bb.lb[4]; a[4] <= odsc->bb.ub[4]; a[4]++) {

dim4:       for(a[3] = odsc->bb.lb[3]; a[3] <= odsc->bb.ub[3]; a[3]++) {

dim3:       for(a[2] = odsc->bb.lb[2]; a[2] <= odsc->bb.ub[2]; a[2]++) {

dim2:       for(a[1] = odsc->bb.lb[1]; a[1] <= odsc->bb.ub[1]; a[1]++) {

dim1:           odsc_tab[index]->bb.lb[0] = odsc->bb.lb[0];
                odsc_tab[index]->bb.ub[0] = odsc->bb.ub[0];

            if(odsc->bb.num_dims == 1)  return num_odsc;

            for(dim = 1; dim < odsc->bb.num_dims; dim++) {
                odsc_tab[index]->bb.lb[dim] = a[dim];
                odsc_tab[index]->bb.ub[dim] = a[dim];                
            }           
            index++;

            }
            if(odsc->bb.num_dims == 2)  return num_odsc;
        }
        if(odsc->bb.num_dims == 3)  return num_odsc;
    }
    if(odsc->bb.num_dims == 4)  return num_odsc;
}
if(odsc->bb.num_dims == 5)  return num_odsc;
}
if(odsc->bb.num_dims == 6)  return num_odsc;
}
if(odsc->bb.num_dims == 7)  return num_odsc;
}
if(odsc->bb.num_dims == 8)  return num_odsc;
}
if(odsc->bb.num_dims == 9)  return num_odsc;
}
return num_odsc;
}
