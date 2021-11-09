/*
 * Copyright (c) 2020, Rutgers Discovery Informatics Institute, Rutgers University
 *
 * See COPYRIGHT in top-level directory.
 * 
 * This is the test example for multi-object transposition
 */

#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"
#include "dspaces.h"
#include "mpi.h"

int main(int argc, char** argv)
{
    // int mode = atoi(argv[1]);
    int rank, nprocs; 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm gcomm = MPI_COMM_WORLD;

    dspaces_client_t ndcl = dspaces_CLIENT_NULL;
    dspaces_init_mpi(gcomm, &ndcl);

    char var_name[128];
    sprintf(var_name, "example2_data");

    int err = 0;

    int dim0 = 2;
    int dim1 = 8;
    int dim2 = 4;


    int loc_dim0 = 1;
    int loc_dim1 = 4;
    int loc_dim2 = 4;

    int ndim = 3;

    uint64_t put_lb[3] = {0}, put_ub[3] = {0}, get_lb[3] = {0}, get_ub[3] = {0};


    double *data = (double*) malloc(loc_dim0*loc_dim1*loc_dim2*sizeof(double));
    double *recv_data = (double*) malloc(dim0*dim1*dim2*sizeof(double));

    /*----------------------------------TestCase1-----------------------*/
    printf("================TESTCASE1: PUT ROW-MAJOR GET ROW-MAJOR\n");
    sprintf(var_name, "example2_test1_data");
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

    put_lb[0] = 0;
    put_lb[1] = 0;
    put_lb[2] = 0;
    put_ub[0] = 3;
    put_ub[1] = 3;
    put_ub[2] = 0;

    err = dspaces_put_layout_new(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub,
                                 dspaces_LAYOUT_RIGHT, data);


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

    put_lb[0] = 0;
    put_lb[1] = 4;
    put_lb[2] = 0;
    put_ub[0] = 3;
    put_ub[1] = 7;
    put_ub[2] = 0;

    err = dspaces_put_layout_new(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub,
                                 dspaces_LAYOUT_RIGHT, data);


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

    put_lb[0] = 0;
    put_lb[1] = 0;
    put_lb[2] = 1;
    put_ub[0] = 3;
    put_ub[1] = 3;
    put_ub[2] = 1;

    err = dspaces_put_layout_new(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub,
                                 dspaces_LAYOUT_RIGHT, data);


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

    put_lb[0] = 0;
    put_lb[1] = 4;
    put_lb[2] = 1;
    put_ub[0] = 3;
    put_ub[1] = 7;
    put_ub[2] = 1;

    err = dspaces_put_layout_new(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub,
                                 dspaces_LAYOUT_RIGHT, data);

    MPI_Barrier(gcomm);

    printf("=================GET ENTIRE OBJ================\n");


    get_lb[0] = 0;
    get_lb[1] = 0;
    get_lb[2] = 0;
    get_ub[0] = 3;
    get_ub[1] = 7;
    get_ub[2] = 1;

    err = dspaces_get_layout_new(ndcl, var_name, 0, sizeof(double), ndim, get_lb, get_ub,
                                 dspaces_LAYOUT_RIGHT, recv_data, -1);

    if(err != 0 )
        goto free;

    // row-major check
    for(int i = 0 ; i < dim0; i++) {
        for(int j = 0; j < dim1; j++) {
            for(int k = 0; k < dim2; k++) {
                printf("%lf ", recv_data[i*dim1*dim2+j*dim2+k]);
            }
            printf("\n");
        }
        printf("**************\n");
    }


    /*----------------------------------TestCase2-----------------------*/
    printf("================TESTCASE2: PUT COLUMN-MAJOR GET COLUMN-MAJOR\n");
    sprintf(var_name, "example2_test2_data");
    printf("=================PUT 1st OBJ================\n");
    for(int i = 0 ; i < loc_dim0; i++) {
        for(int j = 0; j < loc_dim1; j++) {
            for(int k = 0; k < loc_dim2; k++) {
                data[i+j*loc_dim0+k*loc_dim0*loc_dim1] = i+j*2+k*16;
                printf("%lf ", data[i+j*loc_dim0+k*loc_dim0*loc_dim1]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    put_lb[0] = 0;
    put_lb[1] = 0;
    put_lb[2] = 0;
    put_ub[0] = 0;
    put_ub[1] = 3;
    put_ub[2] = 3;

    err = dspaces_put_layout_new(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub,
                                 dspaces_LAYOUT_LEFT, data);


    printf("=================PUT 2nd OBJ================\n");
    for(int i = 0 ; i < loc_dim0; i++) {
        for(int j = 0; j < loc_dim1; j++) {
            for(int k = 0; k < loc_dim2; k++) {
                data[i+j*loc_dim0+k*loc_dim0*loc_dim1] = 8+i+j*2+k*16;
                printf("%lf ", data[i+j*loc_dim0+k*loc_dim0*loc_dim1]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    put_lb[0] = 0;
    put_lb[1] = 4;
    put_lb[2] = 0;
    put_ub[0] = 0;
    put_ub[1] = 7;
    put_ub[2] = 3;

    err = dspaces_put_layout_new(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub,
                                 dspaces_LAYOUT_LEFT, data);


    printf("=================PUT 3rd OBJ================\n");
    for(int i = 0 ; i < loc_dim0; i++) {
        for(int j = 0; j < loc_dim1; j++) {
            for(int k = 0; k < loc_dim2; k++) {
                data[i+j*loc_dim0+k*loc_dim0*loc_dim1] = 1+i+j*2+k*16;
                printf("%lf ", data[i+j*loc_dim0+k*loc_dim0*loc_dim1]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    put_lb[0] = 1;
    put_lb[1] = 0;
    put_lb[2] = 0;
    put_ub[0] = 1;
    put_ub[1] = 3;
    put_ub[2] = 3;

    err = dspaces_put_layout_new(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub,
                                 dspaces_LAYOUT_LEFT, data);


    printf("=================PUT 4th OBJ================\n");
    for(int i = 0 ; i < loc_dim0; i++) {
        for(int j = 0; j < loc_dim1; j++) {
            for(int k = 0; k < loc_dim2; k++) {
                data[i+j*loc_dim0+k*loc_dim0*loc_dim1] = 9+i+j*2+k*16;
                printf("%lf ", data[i+j*loc_dim0+k*loc_dim0*loc_dim1]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    put_lb[0] = 1;
    put_lb[1] = 4;
    put_lb[2] = 0;
    put_ub[0] = 1;
    put_ub[1] = 7;
    put_ub[2] = 3;

    err = dspaces_put_layout_new(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub,
                                 dspaces_LAYOUT_LEFT, data);

    MPI_Barrier(gcomm);


    printf("=================GET ENTIRE OBJ================\n");


    get_lb[0] = 0;
    get_lb[1] = 0;
    get_lb[2] = 0;
    get_ub[0] = 1;
    get_ub[1] = 7;
    get_ub[2] = 3;

    err = dspaces_get_layout_new(ndcl, var_name, 0, sizeof(double), ndim, get_lb, get_ub,
                                 dspaces_LAYOUT_LEFT, recv_data, -1);

    if(err != 0 )
        goto free;

    // column-major check
    for(int i = 0 ; i < dim0; i++) {
        for(int j = 0; j < dim1; j++) {
            for(int k = 0; k < dim2; k++) {
                printf("%lf ", recv_data[i+j*dim0+k*dim0*dim1]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    /*----------------------------------TestCase3-----------------------*/
    printf("================TESTCASE3: PUT ROW-MAJOR GET COLUMN-MAJOR\n");
    sprintf(var_name, "example2_test3_data");
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

    put_lb[0] = 0;
    put_lb[1] = 0;
    put_lb[2] = 0;
    put_ub[0] = 3;
    put_ub[1] = 3;
    put_ub[2] = 0;

    err = dspaces_put_layout_new(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub,
                                 dspaces_LAYOUT_RIGHT, data);


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

    put_lb[0] = 0;
    put_lb[1] = 4;
    put_lb[2] = 0;
    put_ub[0] = 3;
    put_ub[1] = 7;
    put_ub[2] = 0;

    err = dspaces_put_layout_new(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub,
                                 dspaces_LAYOUT_RIGHT, data);


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

    put_lb[0] = 0;
    put_lb[1] = 0;
    put_lb[2] = 1;
    put_ub[0] = 3;
    put_ub[1] = 3;
    put_ub[2] = 1;

    err = dspaces_put_layout_new(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub,
                                 dspaces_LAYOUT_RIGHT, data);


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

    put_lb[0] = 0;
    put_lb[1] = 4;
    put_lb[2] = 1;
    put_ub[0] = 3;
    put_ub[1] = 7;
    put_ub[2] = 1;

    err = dspaces_put_layout_new(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub,
                                 dspaces_LAYOUT_RIGHT, data);

    MPI_Barrier(gcomm);

    printf("=================GET ENTIRE OBJ================\n");


    get_lb[0] = 0;
    get_lb[1] = 0;
    get_lb[2] = 0;
    get_ub[0] = 1;
    get_ub[1] = 7;
    get_ub[2] = 3;

    err = dspaces_get_layout_new(ndcl, var_name, 0, sizeof(double), ndim, get_lb, get_ub,
                                 dspaces_LAYOUT_LEFT, recv_data, -1);

    if(err != 0 )
        goto free;

    // column-major check
    for(int i = 0 ; i < dim0; i++) {
        for(int j = 0; j < dim1; j++) {
            for(int k = 0; k < dim2; k++) {
                printf("%lf ", recv_data[i+j*dim0+k*dim0*dim1]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    /*----------------------------------TestCase4-----------------------*/
    printf("================TESTCASE4: PUT COLUMN-MAJOR GET ROW-MAJOR\n");
    sprintf(var_name, "example2_test4_data");
    printf("=================PUT 1st OBJ================\n");
    for(int i = 0 ; i < loc_dim0; i++) {
        for(int j = 0; j < loc_dim1; j++) {
            for(int k = 0; k < loc_dim2; k++) {
                data[i+j*loc_dim0+k*loc_dim0*loc_dim1] = i+j*2+k*16;
                printf("%lf ", data[i+j*loc_dim0+k*loc_dim0*loc_dim1]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    put_lb[0] = 0;
    put_lb[1] = 0;
    put_lb[2] = 0;
    put_ub[0] = 0;
    put_ub[1] = 3;
    put_ub[2] = 3;

    err = dspaces_put_layout_new(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub,
                                 dspaces_LAYOUT_LEFT, data);


    printf("=================PUT 2nd OBJ================\n");
    for(int i = 0 ; i < loc_dim0; i++) {
        for(int j = 0; j < loc_dim1; j++) {
            for(int k = 0; k < loc_dim2; k++) {
                data[i+j*loc_dim0+k*loc_dim0*loc_dim1] = 8+i+j*2+k*16;
                printf("%lf ", data[i+j*loc_dim0+k*loc_dim0*loc_dim1]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    put_lb[0] = 0;
    put_lb[1] = 4;
    put_lb[2] = 0;
    put_ub[0] = 0;
    put_ub[1] = 7;
    put_ub[2] = 3;

    err = dspaces_put_layout_new(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub,
                                 dspaces_LAYOUT_LEFT, data);


    printf("=================PUT 3rd OBJ================\n");
    for(int i = 0 ; i < loc_dim0; i++) {
        for(int j = 0; j < loc_dim1; j++) {
            for(int k = 0; k < loc_dim2; k++) {
                data[i+j*loc_dim0+k*loc_dim0*loc_dim1] = 1+i+j*2+k*16;
                printf("%lf ", data[i+j*loc_dim0+k*loc_dim0*loc_dim1]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    put_lb[0] = 1;
    put_lb[1] = 0;
    put_lb[2] = 0;
    put_ub[0] = 1;
    put_ub[1] = 3;
    put_ub[2] = 3;

    err = dspaces_put_layout_new(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub,
                                 dspaces_LAYOUT_LEFT, data);


    printf("=================PUT 4th OBJ================\n");
    for(int i = 0 ; i < loc_dim0; i++) {
        for(int j = 0; j < loc_dim1; j++) {
            for(int k = 0; k < loc_dim2; k++) {
                data[i+j*loc_dim0+k*loc_dim0*loc_dim1] = 9+i+j*2+k*16;
                printf("%lf ", data[i+j*loc_dim0+k*loc_dim0*loc_dim1]);
            }
            printf("\n");
        }
        printf("**************\n");
    }

    put_lb[0] = 1;
    put_lb[1] = 4;
    put_lb[2] = 0;
    put_ub[0] = 1;
    put_ub[1] = 7;
    put_ub[2] = 3;

    err = dspaces_put_layout_new(ndcl, var_name, 0, sizeof(double), ndim, put_lb, put_ub,
                                 dspaces_LAYOUT_LEFT, data);

    MPI_Barrier(gcomm);


    printf("=================GET ENTIRE OBJ================\n");


    get_lb[0] = 0;
    get_lb[1] = 0;
    get_lb[2] = 0;
    get_ub[0] = 3;
    get_ub[1] = 7;
    get_ub[2] = 1;

    err = dspaces_get_layout_new(ndcl, var_name, 0, sizeof(double), ndim, get_lb, get_ub,
                                 dspaces_LAYOUT_RIGHT, recv_data, -1);

    if(err != 0 )
        goto free;

    // row-major check
    for(int i = 0 ; i < dim0; i++) {
        for(int j = 0; j < dim1; j++) {
            for(int k = 0; k < dim2; k++) {
                printf("%lf ", recv_data[i*dim1*dim2+j*dim2+k]);
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