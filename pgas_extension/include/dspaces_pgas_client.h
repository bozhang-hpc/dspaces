/*
 * Copyright (c) 2020, Rutgers Discovery Informatics Institute, Rutgers University
 *
 * See COPYRIGHT in top-level directory.
 */


#ifndef __DSPACES_PGAS_CLIENT_H
#define __DSPACES_PGAS_CLIENT_H

#include <dspaces-common.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct dspaces_client* dspaces_client_t;
#define dspaces_CLIENT_NULL ((dspaces_client_t)NULL)

/**
 * @brief register a PGAS view to server.
 *
 * This function will register the local view, which is described by the 
 * local bounding box {(lb[0],lb[1],..,lb[n-1]), (ub[0],ub[1],..,ub[n-1])},
 * to the server.
 * 
 *
 * @param[in] client dspaces client  
 * @param[in] var_name:     Name of the variable.
 * @param[in] ver:      Version of the variable.
 * @param[in] size:     Size (in bytes) for each element of the global
 *              array.
 * @param[in] ndim:     the number of dimensions for the local bounding
 *              box. 
 * @param[in] lb:       coordinates for the lower corner of the local
 *                  bounding box.
 * @param[in] ub:       coordinates for the upper corner of the local
 *                  bounding box. 
 * @return  0 indicates success.
 */
int dspaces_view_reg(dspaces_client_t client, 
        const char *var_name,
        unsigned int ver, int size,
        int ndim, uint64_t *lb, uint64_t *ub);

#if defined(__cplusplus)
}
#endif

#endif