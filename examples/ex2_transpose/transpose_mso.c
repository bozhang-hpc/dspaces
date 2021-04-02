/*
 * Copyright (c) 2020, Rutgers Discovery Informatics Institute, Rutgers University
 *
 * See COPYRIGHT in top-level directory.
 * 
 * This is the test example for multi-sub-object transposition
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
    sprintf(var_name, "example2_data");

    int err = 0;

    int sub_dim0 = 2;
    int sub_dim1 = 4;
    int sub_dim2 = 2;


    int loc_dim0 = 1;
    int loc_dim1 = 4;
    int loc_dim2 = 4;

    
    double *data = (double*) malloc(loc_dim0*loc_dim1*loc_dim2*sizeof(double));
    double *recv_data = (double*) malloc(sub_dim0*sub_dim1*sub_dim2*sizeof(double));

    int ndim = 3;
    uint64_t lb[3] = {0}, ub[3] = {0};

    printf("=================PUT 1st OBJ================\n");
    for(int i = 0 ; i < loc_dim0; i++) {
        for(int j = 0; j < loc_dim1; j++) {
            for(int k = 0; k < loc_dim2; k++) {
                data[i*loc_dim1*loc_dim2+j*loc_dim2+k] = i*loc_dim1*loc_dim2+j*loc_dim2+k;
                printf("%lf ", data[i*loc_dim1*loc_dim2+j*loc_dim2+k]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    lb[0] = 0;
    lb[1] = 0;
    lb[2] = 0;
    ub[0] = 3;
    ub[1] = 3;
    ub[2] = 0;

    err = dspaces_put(ndcl, var_name, 0, sizeof(double), ndim, lb, ub, data);


    printf("=================PUT 2nd OBJ================\n");
    for(int i = 0 ; i < loc_dim0; i++) {
        for(int j = 0; j < loc_dim1; j++) {
            for(int k = 0; k < loc_dim2; k++) {
                data[i*loc_dim1*loc_dim2+j*loc_dim2+k] = 16+i*loc_dim1*loc_dim2+j*loc_dim2+k;
                printf("%lf ", data[i*loc_dim1*loc_dim2+j*loc_dim2+k]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    lb[0] = 0;
    lb[1] = 4;
    lb[2] = 0;
    ub[0] = 3;
    ub[1] = 7;
    ub[2] = 0;

    err = dspaces_put(ndcl, var_name, 0, sizeof(double), ndim, lb, ub, data);


    printf("=================PUT 3rd OBJ================\n");
    for(int i = 0 ; i < loc_dim0; i++) {
        for(int j = 0; j < loc_dim1; j++) {
            for(int k = 0; k < loc_dim2; k++) {
                data[i*loc_dim1*loc_dim2+j*loc_dim2+k] = 32+i*loc_dim1*loc_dim2+j*loc_dim2+k;
                printf("%lf ", data[i*loc_dim1*loc_dim2+j*loc_dim2+k]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    lb[0] = 0;
    lb[1] = 0;
    lb[2] = 1;
    ub[0] = 3;
    ub[1] = 3;
    ub[2] = 1;

    err = dspaces_put(ndcl, var_name, 0, sizeof(double), ndim, lb, ub, data);


    printf("=================PUT 4th OBJ================\n");
    for(int i = 0 ; i < loc_dim0; i++) {
        for(int j = 0; j < loc_dim1; j++) {
            for(int k = 0; k < loc_dim2; k++) {
                data[i*loc_dim1*loc_dim2+j*loc_dim2+k] = 48+i*loc_dim1*loc_dim2+j*loc_dim2+k;
                printf("%lf ", data[i*loc_dim1*loc_dim2+j*loc_dim2+k]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    lb[0] = 0;
    lb[1] = 4;
    lb[2] = 1;
    ub[0] = 3;
    ub[1] = 7;
    ub[2] = 1;

    err = dspaces_put(ndcl, var_name, 0, sizeof(double), ndim, lb, ub, data);

    MPI_Barrier(gcomm);

    printf("=================GET ENTIRE TRANSPOSED OBJ================\n");

    uint64_t recv_lb[3] = {0}, recv_ub[3] = {0};

    recv_lb[0] = 1;
    recv_lb[1] = 2;
    recv_lb[2] = 0;
    recv_ub[0] = 2;
    recv_ub[1] = 5;
    recv_ub[2] = 1;

    err = dspaces_transpose(ndcl, var_name, 0, sizeof(double), ndim, recv_lb, recv_ub);

    err = dspaces_get_layout(ndcl, var_name, 0, sizeof(double), ndim, recv_lb, recv_ub, dspaces_LAYOUT_RIGHT, recv_data, -1);

    if(err != 0 )
        goto free;

    printf("=================OLD METHOD================\n");

    printf("=================Serial Mem check================\n");
    // serial mem check
    for(int i = 0 ; i < sub_dim0; i++) {
        for(int j = 0; j < sub_dim1; j++) {
            for(int k = 0; k < sub_dim2; k++) {
                printf("%lf ", recv_data[i*sub_dim1*sub_dim2+j*sub_dim2+k]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    printf("=================Opposite-major check================\n");
    // opposite-major check
    for(int i = 0 ; i < sub_dim0; i++) {
        for(int j = 0; j < sub_dim1; j++) {
            for(int k = 0; k < sub_dim2; k++) {
                printf("%lf ", recv_data[i+j*sub_dim0+k*sub_dim1*sub_dim0]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    err = dspaces_get_transposed(ndcl, var_name, 0, sizeof(double), ndim, recv_lb, recv_ub, dspaces_LAYOUT_RIGHT, recv_data, -1);

    printf("=================NEW METHOD================\n");

    printf("=================Serial Mem check================\n");
    // serial mem check
    for(int i = 0 ; i < sub_dim0; i++) {
        for(int j = 0; j < sub_dim1; j++) {
            for(int k = 0; k < sub_dim2; k++) {
                printf("%lf ", recv_data[i*sub_dim1*sub_dim2+j*sub_dim2+k]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    printf("=================Opposite-major check================\n");
    // opposite-major check
    for(int i = 0 ; i < dim0; i++) {
        for(int j = 0; j < dim1; j++) {
            for(int k = 0; k < dim2; k++) {
                printf("%lf ", recv_data[i+j*sub_dim0+k*sub_dim1*sub_dim0]);
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