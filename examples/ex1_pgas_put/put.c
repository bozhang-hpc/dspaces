/*
 * Copyright (c) 2020, Rutgers Discovery Informatics Institute, Rutgers University
 *
 * See COPYRIGHT in top-level directory.
 */

#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"
#include "dspaces-client.h"
#include "mpi.h"

int main(int argc, char** argv)
{
    char *listen_addr = argv[1];

    int rank, nprocs; 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm gcomm = MPI_COMM_WORLD;

    dspaces_client_t ndcl = dspaces_CLIENT_NULL;
    client_init(listen_addr, rank, &ndcl);

    char var_name[128];
    sprintf(var_name, "example1_data");

/*
    int dim0 = 2;
    int dim1 = 8;

    double *data = (double*) malloc(dim0*dim1*sizeof(double));

    for(int i = 0 ; i < dim0; i++) {
        for(int j = 0; j < dim1; j++) {
            data[i*dim1+j] = i*dim1+j;
            printf("%lf ", data[i*dim1+j]);
        }
        printf("\n");
    }

    int ndim = 2;
    uint64_t view_layout[2];
    view_layout[0]=8;
    view_layout[1]=2;
    uint64_t lb[2] = {0}, ub[2] = {0};

    lb[0] = 1;
    lb[1] = 0;
    ub[0] = 5;
    ub[1] = 1;

*/

/*
    int dim0 = 2;
    int dim1 = 4;
    int dim2 = 4;

    double *data = (double*) malloc(dim0*dim1*dim2*sizeof(double));

    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < dim1; j++) {
            for(int k = 0; k < dim2; k++) {
                data[(i*dim1+j)*dim2+k] = (i*dim1+j)*dim2+k;
                printf("%lf ", data[(i*dim1+j)*dim2+k]);
            }
            printf("\n");
        }
        printf("\n\n");
    }

    int ndim = 3;
    uint64_t view_layout[3];
    view_layout[0]=dim2;
    view_layout[1]=dim1;
    view_layout[2]=dim0;
    uint64_t lb[3] = {0}, ub[3] = {0};


    ub[0] = 3;
    ub[1] = 1;
    ub[2] = 1;
*/

    int dim0 = 2;
    int dim1 = 4;
    int dim2 = 4;

    double *data = (double*) malloc(dim0*dim1*dim2*sizeof(double));

    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < dim1; j++) {
            for(int k = 0; k < dim2; k++) {
                data[(i*dim1+j)*dim2+k] = (i*dim1+j)*dim2+k;
                printf("%lf ", data[(i*dim1+j)*dim2+k]);
            }
            printf("\n");
        }
        printf("\n\n");
    }

    int ndim = 3;
    uint64_t view_layout[3];
    view_layout[0]=dim2;
    view_layout[1]=dim1;
    view_layout[2]=dim0;
    uint64_t lb[3] = {0}, ub[3] = {0};


    ub[0] = 3;
    ub[1] = 3;
    ub[2] = 1;


    int err = dspaces_view_put(ndcl, var_name, 0, sizeof(double), ndim, view_layout, lb, ub, data);

    MPI_Barrier(gcomm);

    sleep(15);

    client_finalize(ndcl);

    return err;

}