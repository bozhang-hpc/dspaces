/*
 * Copyright (c) 2020, Rutgers Discovery Informatics Institute, Rutgers University
 *
 * See COPYRIGHT in top-level directory.
 */

#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"
#include "dspaces.h"
#include "mpi.h"

int main(int argc, char** argv)
{
    int rank, nprocs; 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm gcomm = MPI_COMM_WORLD;

    dspaces_client_t ndcl = dspaces_CLIENT_NULL;
    dspaces_init(rank, &ndcl);

    char var_name[128];
    sprintf(var_name, "example1_data");

    int err = 0;


    int dim0 = 2;
    int dim1 = 8;

    double *data = (double*) malloc(dim0*dim1*sizeof(double));
    double *recv_data = (double*) malloc(dim0*dim1*sizeof(double));

    prinf("=================PUT================\n");
    for(int i = 0 ; i < dim0; i++) {
        for(int j = 0; j < dim1; j++) {
            data[i*dim1+j] = i*dim1+j;
            printf("%lf ", data[i*dim1+j]);
        }
        printf("\n");
    }


    int ndim = 2;

    uint64_t lb[2] = {0}, ub[2] = {0};

    lb[0] = 0;
    lb[1] = 0;
    ub[0] = 7;
    ub[1] = 1;

    
    err = dspaces_put(ndcl, var_name, 0, sizeof(double), ndim, lb, ub, data);
    
    MPI_Barrier(gcomm);

    err = dspaces_transpose(ndcl, var_name, 0, sizeof(double), ndim, lb, ub);

    err = dspaces_get_layout(ndcl, var_name, 0, sizeof(double), ndim, lb, ub, dspaces_LAYOUT_RIGHT, recv_data, -1);

    if(err != 0 )
        goto free;

    prinf("=================GET================\n");
    for(int i = 0 ; i < dim0; i++) {
        for(int j = 0; j < dim1; j++) {
            printf("%lf ", recv_data[i*dim1+j]);
        }
        printf("\n");
    }

    
free:

    MPI_Barrier(gcomm);
    free(data);
    free(recv_data);

    dspaces_fini(ndcl);

    MPI_Finalize();

    return err;

}