#include <stdio.h>
#include <string.h>
#include "memory.h"
 
/*
* Look for lines in the procfile contents like: 
* VmRSS:         5560 kB
* VmSize:         5560 kB
*
* Grab the number between the whitespace and the "kB"
* If 1 is returned in the end, there was a serious problem 
* (we could not find one of the memory usages)
*/
int get_memory_usage_kb(long* vmrss_kb, long* vmsize_kb)
{
    /* Get the the current process' status file from the proc filesystem */
    FILE* procfile = fopen("/proc/self/status", "r");

    //long to_read = 8192;
    char buffer[8192];
    int read = fread(buffer, sizeof(char), 8192, procfile);
    fclose(procfile);

    short found_vmrss = 0;
    short found_vmsize = 0;
    char* search_result;

    /* Look through proc status contents line by line */
    char delims[] = "\n";
    char* line = strtok(buffer, delims);

    while (line != NULL && (found_vmrss == 0 || found_vmsize == 0) )
    {
        search_result = strstr(line, "VmRSS:");
        if (search_result != NULL)
        {
            sscanf(line, "%*s %ld", vmrss_kb);
            found_vmrss = 1;
        }

        search_result = strstr(line, "VmSize:");
        if (search_result != NULL)
        {
            sscanf(line, "%*s %ld", vmsize_kb);
            found_vmsize = 1;
        }

        line = strtok(NULL, delims);
    }

    return (found_vmrss == 1 && found_vmsize == 1) ? 0 : 1;
}

int get_cluster_memory_usage_kb(long* vmrss_per_process, long* vmsize_per_process, int root, MPI_Comm comm)
{
    long vmrss_kb;
    long vmsize_kb;
    int ret_code = get_memory_usage_kb(&vmrss_kb, &vmsize_kb);

    if (ret_code != 0)
    {
        printf("Could not gather memory usage!\n");
        return ret_code;
    }

    MPI_Gather(&vmrss_kb, 1, MPI_UNSIGNED_LONG, 
        vmrss_per_process, 1, MPI_UNSIGNED_LONG, 
        root, comm);

    MPI_Gather(&vmsize_kb, 1, MPI_UNSIGNED_LONG, 
        vmsize_per_process, 1, MPI_UNSIGNED_LONG, 
        root, comm);

    return 0;
}

int get_global_memory_usage_kb(long* global_vmrss, long* global_vmsize, int np)
{
    long* vmrss_per_process  = (long*) malloc(np*sizeof(long));
    long* vmsize_per_process = (long*) malloc(np*sizeof(long));
    int ret_code = get_cluster_memory_usage_kb(vmrss_per_process, vmsize_per_process, 0, np);

    if (ret_code != 0)
    {
        return ret_code;
    }

    *global_vmrss = 0;
    *global_vmsize = 0;
    for (int i = 0; i < np; i++)
    {
        *global_vmrss += vmrss_per_process[i];
        *global_vmsize += vmsize_per_process[i];
    }

    free(vmrss_per_process);
    free(vmsize_per_process);

    return 0;
}