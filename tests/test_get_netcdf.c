#include "mpi.h"
#include "dspaces.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"
#include "sys/time.h"
#include "time.h"

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

    int nts = 1980;
    size_t dsize = 96*144;

    void** data = (void**) malloc(nts*sizeof(void*));
    double* timer = (double*) malloc(nts*sizeof(double));
    double total_time = 0;

    FILE* fp = fopen("test.log", "w+");
    fprintf(fp, "Timestep,Millisecond\n");


    uint64_t lb[2] = {0};
    uint64_t ub[2];
    ub[0] = 96-1;
    ub[1] = 144-1;

    for(int ts=0; ts<nts; ts++)
    {
        data[ts] = (void*) malloc(dsize*sizeof(float));
        gettimeofday(&start, NULL);
        dspaces_get_netcdf(client, "/home/zhangbo/Codes/netcdf_read_test/data/tas_Amon_NorESM2-LM_historical_r1i1p1f1_gn_185001-201412.nc",
                        "tas", sizeof(float), 2, ts, lb, ub, data[ts]);
        gettimeofday(&end, NULL);
        timer[ts] = (end.tv_sec-start.tv_sec)*1e3 + (end.tv_usec-start.tv_usec)*1e-3;
        total_time += timer[ts];
        fprintf(fp, "%d,%lf\n", ts, timer[ts]);
        if(ts%100==0) printf("ts = %d\n", ts);
    }

    fprintf(fp, "Total Time in seconds,%lf\n", total_time*1e-3);
    fclose(fp);
    
    free(timer);
    for(int ts=0; ts<nts; ts++) {
        free(data[ts]);
    }
    free(data);

    

    dspaces_fini(client);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}