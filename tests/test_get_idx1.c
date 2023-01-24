#include "mpi.h"
#include "dspaces.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"

int main(int argc, char** argv) 
{
    int ret;
    int nprocs, rank;
    dspaces_client_t client;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);

    ret = dspaces_init_mpi(MPI_COMM_WORLD, &client);
    if(ret != dspaces_SUCCESS) {
        fprintf(stderr, "dspaces_init_mpi() failed with %d.\n", ret);
        return -1;
    }

    dspaces_idx1_params idxp = {.dir = "/home/zhangbo/Data/NASA/LLC4320_test",
                                .filename = "llc4320_x_y_depth.idx",
                                .varname = 'u',
                                .element_size = sizeof(float),
                                .resolution = -1,
                                .ndims = 3,
                                .lb = {0},
                                .ub = {17279, 12959, 89}
                                };

    size_t dsize = 17280*12960*90;

    void* data = (void*) malloc(dsize*sizeof(float));

    int nts = 1;
    for(int ts=0; ts<nts; ts++)
    {
        idxp.timestep = ts;
        dspaces_get_idx1(client, idxp, data);
        printf("ts = %d, get_idx1()\n", ts);
    }

    dspaces_fini(client);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}