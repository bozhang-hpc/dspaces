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
    int dim2 = 4;

    int sub_dim0 = 2;
    int sub_dim1 = 4;
    int sub_dim2 = 2;

    double *data = (double*) malloc(dim0*dim1*dim2*sizeof(double));
    double *recv_data = (double*) malloc(sub_dim0*sub_dim1*sub_dim2*sizeof(double));

    printf("=================PUT================\n");
    for(int i = 0 ; i < dim0; i++) {
        for(int j = 0; j < dim1; j++) {
            for(int k = 0; k < dim2; k++) {
                data[i*dim1*dim2+j*dim2+k] = i*dim1*dim2+j*dim2+k;
                printf("%lf ", data[i*dim1*dim2+j*dim2+k]);
            }
            printf("\n");
        }
        printf("**************\n");
    }


    int ndim = 3;

    uint64_t lb[3] = {0}, ub[3] = {0};

    lb[0] = 0;
    lb[1] = 0;
    lb[2] = 0;
    ub[0] = 3;
    ub[1] = 7;
    ub[2] = 1;

    uint64_t recv_lb[3] = {0}, recv_ub[3] = {0};

    recv_lb[0] = 1;
    recv_lb[1] = 2;
    recv_lb[2] = 0;
    recv_ub[0] = 2;
    recv_ub[1] = 5;
    recv_ub[2] = 1;

    
    err = dspaces_put(ndcl, var_name, 0, sizeof(double), ndim, lb, ub, data);
    
    MPI_Barrier(gcomm);

    // if you only want a subset of a data object, just transpose the subset bbox 

    err = dspaces_transpose(ndcl, var_name, 0, sizeof(double), ndim, recv_lb, recv_ub);

    err = dspaces_get_layout(ndcl, var_name, 0, sizeof(double), ndim, recv_lb, recv_ub, dspaces_LAYOUT_RIGHT, recv_data, -1);

    if(err != 0 )
        goto free;

    printf("=================GET================\n");
    for(int i = 0 ; i < sub_dim0; i++) {
        for(int j = 0; j < sub_dim1; j++) {
            for(int k = 0; k < sub_dim2; k++) {
                printf("%lf ", recv_data[i+j*sub_dim0+k*sub_dim0*sub_dim1]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    
free:

    MPI_Barrier(gcomm);
    free(data);
    free(recv_data);

    dspaces_fini(ndcl);

    MPI_Finalize();

    return err;

}