#include "mpi.h"
#include "dspaces.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"
#include "sys/time.h"
#include "time.h"
#include "unistd.h"

int main(int argc, char** argv) 
{
    int ret;
    int nprocs, rank;
    dspaces_client_t client;
    struct timeval start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);

    ret = dspaces_init_mpi(MPI_COMM_WORLD, &client);
    if(ret != dspaces_SUCCESS) {
        fprintf(stderr, "dspaces_init_mpi() failed with %d.\n", ret);
        return -1;
    }

    dspaces_idx1_params idxp = {.dir = "/home/zhangbo/Data/NASA/LLC2160",
                                .filename = "u_face_0_depth_52_time_10.idx",
                                .varname = 'u',
                                .element_size = sizeof(float),
                                .resolution = -1,
                                .ndims = 3,
                                .lb = {0},
                                .ub = {1439, 1439, 51}
                                };

    uint64_t dimx = 1440;
    uint64_t dimy = 1440;
    uint64_t dimz = 52;
    uint64_t dsize = dimx * dimy * dimz;
    int nts = 10;

    void* data = (void*) malloc(dsize*sizeof(float));
    double* timer = (double*) malloc(nts*sizeof(double));
    double total_time = 0;

    FILE* fp = fopen("test_get_idx1.log", "w+");
    fprintf(fp, "Timestep,Millisecond\n");

    for(int ts=1; ts<nts+1; ts++)
    {
        idxp.timestep = ts;
        gettimeofday(&start, NULL);
        dspaces_get_idx1(client, idxp, data);
        gettimeofday(&end, NULL);
        timer[ts] = (end.tv_sec-start.tv_sec)*1e3 + (end.tv_usec-start.tv_usec)*1e-3;
        total_time += timer[ts];
        fprintf(fp, "%d,%lf\n", ts, timer[ts]);
        printf("ts = %d\n", ts);
    }

    fprintf(fp, "Total Time in seconds,%lf\n", total_time*1e-3);
    fclose(fp);

    free(timer);
    free(data);

    if(rank == 0) {
        dspaces_kill(client);
    }

    dspaces_fini(client);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}