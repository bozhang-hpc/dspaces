/*
 * Copyright (c) 2020, Rutgers Discovery Informatics Institute, Rutgers University
 *
 * See COPYRIGHT in top-level directory.
 * 
 * This is the test example for sub-region transposition in single object 
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


    int dim0 = 2;
    int dim1 = 8;
    int dim2 = 4;

    int sub_dim0 = 2;
    int sub_dim1 = 4;
    int sub_dim2 = 2;

    int ndim = 3;

    uint64_t put_lb[3] = {0}, put_ub[3] = {0}, get_lb[3] = {0}, get_ub[3] = {0};

    double *data = (double*) malloc(dim0*dim1*dim2*sizeof(double));
    double *recv_data = (double*) malloc(sub_dim0*sub_dim1*sub_dim2*sizeof(double));

    /*----------------------------------TestCase1-----------------------*/
    printf("================TESTCASE1: PUT ROW-MAJOR GET ROW-MAJOR\n");
    sprintf(var_name, "example2_test1_data");

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


    put_lb[0] = 0;
    put_lb[1] = 0;
    put_lb[2] = 0;
    put_ub[0] = 3;
    put_ub[1] = 7;
    put_ub[2] = 1;

    err = dspaces_put_layout(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub, dspaces_LAYOUT_RIGHT, data);
    
    MPI_Barrier(gcomm);

    // if you only want a subset of a data object, just transpose the subset bbox 

    get_lb[0] = 1;
    get_lb[1] = 2;
    get_lb[2] = 0;
    get_ub[0] = 2;
    get_ub[1] = 5;
    get_ub[2] = 1;

    err = dspaces_get_layout(ndcl, var_name, 0, sizeof(double), ndim, get_lb, get_ub, dspaces_LAYOUT_RIGHT, recv_data, -1);

    if(err != 0 )
        goto free;

    printf("=================GET================\n");
    for(int i = 0 ; i < sub_dim0; i++) {
        for(int j = 0; j < sub_dim1; j++) {
            for(int k = 0; k < sub_dim2; k++) {
                printf("%lf ", recv_data[i*sub_dim1*sub_dim2+j*sub_dim2+k]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    
    /*----------------------------------TestCase2-----------------------*/
    printf("================TESTCASE2: PUT COLUMN-MAJOR GET COLUMN-MAJOR\n");
    sprintf(var_name, "example2_test2_data");
    //column-major put
    printf("=================PUT================\n");
    for(int i = 0 ; i < dim0; i++) {
        for(int j = 0; j < dim1; j++) {
            for(int k = 0; k < dim2; k++) {
                data[i+j*dim0+k*dim1*dim0] = i+j*dim0+k*dim1*dim0;
                printf("%lf ", data[i+j*dim0+k*dim1*dim0]);
            }
            printf("\n");
        }
        printf("**************\n");
    }


    put_lb[0] = 0;
    put_lb[1] = 0;
    put_lb[2] = 0;
    put_ub[0] = 1;
    put_ub[1] = 7;
    put_ub[2] = 3;

    err = dspaces_put_layout(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub, dspaces_LAYOUT_LEFT, data);

    MPI_Barrier(gcomm);

    get_lb[0] = 0;
    get_lb[1] = 2;
    get_lb[2] = 1;
    get_ub[0] = 1;
    get_ub[1] = 5;
    get_ub[2] = 2;

    err = dspaces_get_layout(ndcl, var_name, 0, sizeof(double), ndim, get_lb, get_ub, dspaces_LAYOUT_LEFT, recv_data, -1);

    // column-major check
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
    
    /*----------------------------------TestCase3-----------------------*/
    printf("================TESTCASE3: PUT ROW-MAJOR GET COLUMN-MAJOR\n");
    sprintf(var_name, "example2_test3_data");


    //row-major put
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

    put_lb[0] = 0;
    put_lb[1] = 0;
    put_lb[2] = 0;
    put_ub[0] = 3;
    put_ub[1] = 7;
    put_ub[2] = 1;
    
    err = dspaces_put_layout(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub, dspaces_LAYOUT_RIGHT, data);
    
    MPI_Barrier(gcomm);

    get_lb[0] = 0;
    get_lb[1] = 2;
    get_lb[2] = 1;
    get_ub[0] = 1;
    get_ub[1] = 5;
    get_ub[2] = 2;

    err = dspaces_get_layout(ndcl, var_name, 0, sizeof(double), ndim, get_lb, get_ub, dspaces_LAYOUT_LEFT, recv_data, -1);

    if(err != 0 )
        goto free;

    // column-major check
    printf("=================GET================\n");
    for(int i = 0 ; i < sub_dim0; i++) {
        for(int j = 0; j < sub_dim1; j++) {
            for(int k = 0; k < sub_dim2; k++) {
                printf("%lf ", recv_data[i+j*sub_dim0+k*sub_dim1*sub_dim0]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    /*----------------------------------TestCase4-----------------------*/
    printf("================TESTCASE4: PUT COLUMN-MAJOR GET ROW-MAJOR\n");
    sprintf(var_name, "example2_test4_data");

    //column-major put
    printf("=================PUT================\n");
    for(int i = 0 ; i < dim0; i++) {
        for(int j = 0; j < dim1; j++) {
            for(int k = 0; k < dim2; k++) {
                data[i+j*dim0+k*dim1*dim0] = i+j*dim0+k*dim1*dim0;
                printf("%lf ", data[i+j*dim0+k*dim1*dim0]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    put_lb[0] = 0;
    put_lb[1] = 0;
    put_lb[2] = 0;
    put_ub[0] = 1;
    put_ub[1] = 7;
    put_ub[2] = 3;


    err = dspaces_put_layout(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub, dspaces_LAYOUT_LEFT, data);

    MPI_Barrier(gcomm);

    get_lb[0] = 1;
    get_lb[1] = 2;
    get_lb[2] = 0;
    get_ub[0] = 2;
    get_ub[1] = 5;
    get_ub[2] = 1;

    err = dspaces_get_layout(ndcl, var_name, 0, sizeof(double), ndim, get_lb, get_ub, dspaces_LAYOUT_RIGHT, recv_data, -1);

    if(err != 0 )
        goto free;

    // row-major check
    printf("=================GET================\n");
    for(int i = 0 ; i < sub_dim0; i++) {
        for(int j = 0; j < sub_dim1; j++) {
            for(int k = 0; k < sub_dim2; k++) {
                printf("%lf ", recv_data[i*sub_dim1*sub_dim2+j*sub_dim2+k]);
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