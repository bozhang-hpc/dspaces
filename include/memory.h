#ifndef __DS_MEMORY_H_
#define __DS_MEMORY_H_

#include <stdlib.h>
#include <mpi.h>

int get_memory_usage_kb(long* vmrss_kb, long* vmsize_kb);
int get_cluster_memory_usage_kb(long* vmrss_per_process, long* vmsize_per_process, int root, MPI_Comm comm);
int get_global_memory_usage_kb(long* global_vmrss, long* global_vmsize, MPI_Comm comm);

#endif