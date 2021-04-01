/*
 * Copyright (c) 2020, Rutgers Discovery Informatics Institute, Rutgers University
 *
 * See COPYRIGHT in top-level directory.
 * 
 * This is the test example for single object transposition
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
    /*----------------------------------TestCase1-----------------------*/
    sprintf(var_name, "example2_data");

    int err = 0;


    int dim0 = 2;
    int dim1 = 8;
    int dim2 = 4;

    double *data = (double*) malloc(dim0*dim1*dim2*sizeof(double));
    double *recv_data = (double*) malloc(dim0*dim1*dim2*sizeof(double));

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

    
    err = dspaces_put(ndcl, var_name, 0, sizeof(double), ndim, lb, ub, data);
    
    MPI_Barrier(gcomm);

    err = dspaces_transpose(ndcl, var_name, 0, sizeof(double), ndim, lb, ub);

    err = dspaces_get_layout(ndcl, var_name, 0, sizeof(double), ndim, lb, ub, dspaces_LAYOUT_RIGHT, recv_data, -1);

    if(err != 0 )
        goto free;

    // opposite-major check
    printf("=================GET================\n");
    for(int i = 0 ; i < dim0; i++) {
        for(int j = 0; j < dim1; j++) {
            for(int k = 0; k < dim2; k++) {
                printf("%lf ", recv_data[i+j*dim0+k*dim1*dim0]);
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