#include "stdlib.h"
#include "stdio.h"
#include "stdint.h"
#include "netcdf.h"
#include "ss_data.h"
#include "netcdf_wrapper.h"

int netcdf_open_rd(char* filepath, int* ncid)
{
    int ret = 0;
    /* Open the file. */
    ret = nc_open(filepath, NC_NOWRITE, &ncid);
    if(ret != NC_NOERR) {
        fprintf(stderr,"Rank %i: %s, line %i (%s): %s", server->rank,
                     __FILE__, __LINE__, __func__, nc_strerror(ret));   
    }
    return ret;
}

int netcdf_inq_path(int ncid, char* filepath)
{
    int ret = 0;
    ret = nv_inq_path(ncid, NULL, filepath);
    if(ret != NC_NOERR) {
        fprintf(stderr,"Rank %i: %s, line %i (%s): %s", server->rank,
                     __FILE__, __LINE__, __func__, nc_strerror(ret));   
    }
    return ret;
}

/* Get the varid of the data variable, based on its name. */
int netcdf_inq_varid(int ncid, char* varname, int* varid)
{
    int ret = 0;
    ret = nc_inq_varid(ncid, varname, &varid);
    if(ret != NC_NOERR) {
        fprintf(stderr,"Rank %i: %s, line %i (%s): %s", server->rank,
                     __FILE__, __LINE__, __func__, nc_strerror(ret));   
    }
    return ret;
}

int netcdf_inq_varname(int ncid, int varid, char* name)
{
    int ret = 0;
    ret = nc_inq_varname(ncid, varid, name);
    if(ret != NC_NOERR) {
        fprintf(stderr,"Rank %i: %s, line %i (%s): %s", server->rank,
                     __FILE__, __LINE__, __func__, nc_strerror(ret));   
    }
    return ret;
}

int netcdf_inq_var_ndim(int ncid, int varid, int* ndim)
{
    int ret = 0;
    ret = nc_inq_varndims(ncid, varid, ndim);
    if(ret != NC_NOERR) {
        fprintf(stderr,"%s, line %i (%s): %s",
                     __FILE__, __LINE__, __func__, nc_strerror(ret));   
    }
    return ret;
}

int netcdf_inq_var_dimlen(int ncid, int varid, size_t* len)
{
    int ret = 0;
    int idx = 0, ndim;
    int *dimid;
    ret = nc_inq_varndims(ncid, varid, &ndim);
    if(ret != NC_NOERR) {
        fprintf(stderr,"%s, line %i (%s): %s",
                     __FILE__, __LINE__, __func__, nc_strerror(ret));   
    }
    dimid = (int*) malloc(ndim*sizeof(int));
    ret = nc_inq_vardimid(ncid, varid, dimid);
    for(int i=0; i<ndim; i++) {
        ret = nc_inq_dimlen(ncid, dimids[i], &len[i]);
        if(ret != NC_NOERR) {
            fprintf(stderr,"%s, line %i (%s): %s",
                        __FILE__, __LINE__, __func__, nc_strerror(ret));   
        }
    }
    return ret;
}

/* Get the timestep index & number of total timestep*/
int netcdf_inq_var_timestep(int ncid, int varid, int* tidx, int* nts)
{
    int ret = 0;
    nc_type var_type;
    int var_ndims;
    int  var_dimids[10];
    int var_natts;
    char dimname[128];
    int found = 0;
    /* Get the attributes of the variable */
    ret = nc_inq_var(ncid, varid, NULL, &var_type, &var_ndims, var_dimids, &var_natts);
    if(ret != NC_NOERR) {
        fprintf(stderr,"%s, line %i (%s): %s",
                     __FILE__, __LINE__, __func__, nc_strerror(ret));   
    }

    for(int i=0; i<var_ndims; i++) {
        ret = nc_inq_dimname(ncid, var_dimids[i], dimname);
        if(ret != NC_NOERR) {
            fprintf(stderr,"%s, line %i (%s): %s",
                        __FILE__, __LINE__, __func__, nc_strerror(ret));   
        }
        if(strcmp(dimname, "time") == 0) { //equal
            *tidx = i;
            ret = nc_inq_dimlen(ncid, var_dimids[i], nts);
            if(ret != NC_NOERR) {
                fprintf(stderr,"%s, line %i (%s): %s",
                            __FILE__, __LINE__, __func__, nc_strerror(ret));
                return ret;
            }
            found = 1;
        }
    }
    if(!found) {
        *tidx = -1;
        *nts = -1;
    }
    return ret;
}

int netcdf_inq_var_elemsize(int ncid, int varid, size_t* elemsize)
{
    int ret = 0;
    /* Get elem size from nc_var_type */
    ret = nc_inq_type(ncid, var_type, NULL, elem_size);
    if(ret != NC_NOERR) {
        fprintf(stderr,"Rank %i: %s, line %i (%s): %s", server->rank,
                     __FILE__, __LINE__, __func__, nc_strerror(ret));   
    }
    return ret;
}

int netcdf_read_var_ts(int ncid, int varid, int nc_ndims, size_t* nc_dimlen,
                         int tidx, int ts, void* data)
{
    int ret = 0;
    size_t* nc_var_start = (size_t*) malloc(nc_ndims*sizeof(size_t));
    size_t* nc_var_count = (size_t*) malloc(nc_ndims*sizeof(size_t));
    for(int i=0; i<nc_ndims; i++) {
        nc_var_start[i] = 0;
        nc_var_count[i] = nc_dimlen[i] - 1;
    }
    /* time dim */
    nc_var_start[tidx] = ts;
    nc_var_count[tidx] = 1;
    ret = nc_get_vara(ncid, varid, nc_var_start, nc_var_count, data);
    if(ret != NC_NOERR) {
        fprintf(stderr,"Rank %i: %s, line %i (%s): %s", server->rank,
                     __FILE__, __LINE__, __func__, nc_strerror(ret));
    }
    free(nc_var_start);
    free(nc_var_count);
    return ret;

}
