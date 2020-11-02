/*
 * Copyright (c) 2020, Rutgers Discovery Informatics Institute, Rutgers University
 *
 * See COPYRIGHT in top-level directory.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>
#include <inttypes.h>
#include <pthread.h>
#include "ss_data.h"
#include "dspaces_pgas_client.h"
#include "pgas_client.h"
#include "gspace.h"

#define DEBUG_OUT(args...) \
    do { \
        if(client->f_debug) { \
           fprintf(stderr, "Rank %i: %s, line %i (%s): ", client->rank, __FILE__, __LINE__, __func__); \
           fprintf(stderr, args); \
        } \
    }while(0);

static hg_return_t get_server_address(dspaces_client_t client, hg_addr_t *server_addr)
{ 
    int peer_id = client->rank % client->size_sp;
 
    return(margo_addr_lookup(client->mid, client->server_address[peer_id], server_addr));
}

int dspaces_view_reg(dspaces_client_t client, 
        const char *var_name,
        unsigned int ver, int size,
        int ndim, uint64_t *lb, uint64_t *ub)
{
    hg_addr_t server_addr;
    hg_handle_t handle;
	hg_return_t hret;
    int ret = dspaces_SUCCESS;

    obj_descriptor odsc = {
            .version = ver, .owner = {0}, 
            .st = st,
            .size = elem_size,
            .bb = {.num_dims = ndim,}
    };

    memset(odsc.bb.lb.c, 0, sizeof(uint64_t)*BBOX_MAX_NDIM);
    memset(odsc.bb.ub.c, 0, sizeof(uint64_t)*BBOX_MAX_NDIM);

    memcpy(odsc.bb.lb.c, lb, sizeof(uint64_t)*ndim);
    memcpy(odsc.bb.ub.c, ub, sizeof(uint64_t)*ndim);

    strncpy(odsc.name, var_name, sizeof(odsc.name)-1);
    odsc.name[sizeof(odsc.name)-1] = '\0';

    odsc_gdim_t in;
    bulk_out_t out;
    struct global_dimension odsc_gdim;
    set_global_dimension(&(client->dcg->gdim_list), var_name, &(client->dcg->default_gdim),
                         &odsc_gdim);


    in.odsc.size = sizeof(odsc);
    in.odsc.raw_odsc = (char*)(&odsc);
    in.odsc.gdim_size = sizeof(struct global_dimension);
    in.odsc.raw_gdim = (char*)(&odsc_gdim);

    
    get_server_address(client, &server_addr);
    /* create handle */
    hret = margo_create( client->mid,
            server_addr,
            client->put_id,
            &handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,"ERROR: (%s): margo_create() failed\n", __func__);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_forward(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,"ERROR: (%s): margo_forward() failed\n", __func__);
        margo_destroy(handle);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_get_output(handle, &out);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,"ERROR: (%s): margo_get_output() failed\n", __func__);
        margo_destroy(handle);
        return dspaces_ERR_MERCURY;
    }

    ret = out.ret;
    margo_free_output(handle, &out);
    margo_destroy(handle);
    margo_addr_free(client->mid, server_addr);
	return ret;
}