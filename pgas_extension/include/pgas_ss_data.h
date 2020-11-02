/*
 * Copyright (c) 2020, Rutgers Discovery Informatics Institute, Rutgers University
 *
 * See COPYRIGHT in top-level directory.
 */

#ifndef __PGAS_SS_DATA_H_
#define __PGAS_SS_DATA_H_

#include <stdlib.h>

#include "bbox.h"
#include "list.h"
#include "ss_data.h"


typedef struct{
        char                    name[154];

        enum storage_type       st;

        unsigned int            version;



        /* Size of one element of a data object. */
        size_t                  size;
} pgas_obj_descriptor;

/* split in_odsc to multiple 1D odsc and put them in a table
   return the number of table entries
*/
uint64_t obj_desc_to1Dbbox(obj_descriptor *odsc, obj_descriptor **odsc_tab);

#endif /* __PGAS_SS_DATA_H_ */