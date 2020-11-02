/*
 * Copyright (c) 2020, Rutgers Discovery Informatics Institute, Rutgers University
 *
 * See COPYRIGHT in top-level directory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <abt.h>
#include "ss_data.h"
#include "dspaces-server.h"
#include "gspace.h"

#define DEBUG_OUT(args...) \
    do { \
        if(server->f_debug) { \
           fprintf(stderr, "Rank %i: %s, line %i (%s): ", server->rank, __FILE__, __LINE__, __func__); \
           fprintf(stderr, args); \
        } \
    }while(0);



DECLARE_MARGO_RPC_HANDLER(view_reg_rpc);

static void view_reg_rpc(hg_handle_t h);

static void view_reg_rpc(hg_handle_t handle)
{
    hg_return_t hret;
    odsc_gdim_t in;
    bulk_out_t out;

    margo_instance_id mid = margo_hg_handle_get_instance(handle);

    hret = margo_get_input(handle, &in);
    assert(hret == HG_SUCCESS);

    obj_descriptor in_odsc;
    memcpy(&in_odsc, in.odsc.raw_odsc, sizeof(in_odsc));
    //set the owner to be this server address
    hg_addr_t owner_addr;
    size_t owner_addr_size = 128;

    margo_addr_self(server->mid, &owner_addr);
    margo_addr_to_string(server->mid, in_odsc.owner, &owner_addr_size, owner_addr);
    margo_addr_free(server->mid, owner_addr);

    obj_descriptor **odsc_tab; 
    uint64_t ent_num = obj_desc_to1Dbbox(&in_odsc, odsc_tab);

    int i;
    for(i = 0; i < ent_num; i++) {
        struct obj_data *od;
        od = obj_data_alloc(odsc_tab[i]);
        memcpy(&od->gdim, in.odsc.raw_gdim, sizeof(struct global_dimension));

        if(!od)
            fprintf(stderr, "ERROR: (%s): object allocation failed!\n", __func__);

        ABT_mutex_lock(server->ls_mutex);
        ls_add_obj(server->dsg->ls, od);
        ABT_mutex_unlock(server->ls_mutex);

        obj_update_dht(server, od, DS_OBJ_NEW);
    }

    DEBUG_OUT("Finished obj_put_update from put_rpc\n");

    out.ret = dspaces_SUCCESS;

    margo_respond(handle, &out);
    margo_free_input(handle, &in);
    margo_destroy(handle);
  
}
DEFINE_MARGO_RPC_HANDLER(view_reg_rpc)