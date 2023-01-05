#include "mpi.h"

/* Open a NetCDF file, return its ncid. */
int netcdf_open_rd(char* filepath, int* ncid);

/* close a NetCDF file */
int netcdf_close(int ncid);

/* Inquire the NetCDF file path based on its ncid. */
int netcdf_inq_path(int ncid, char* filepath);

/* Inquire the varid of a variable, based on its name. */
int netcdf_inq_varid(int ncid, char* varname, int* varid);

/* Inquire the name of a variable, based on its varid. */
int netcdf_inq_varname(int ncid, int varid, char* name);

/* Inquire the number of dimension of a variable, based on its varid. */
int netcdf_inq_var_ndim(int ncid, int varid, int* ndim);

/* Inquire the length of each dimension of a variable, based on its varid. */
int netcdf_inq_var_dimlen(int ncid, int varid, size_t* len);

/* Inquire the timestep index & number of total timestep of a variable. */
int netcdf_inq_var_timestep(int ncid, int varid, int* tidx, int* nts);

/* Inquire the element size of a variable. */
int netcdf_inq_var_elemsize(int ncid, int varid, size_t* elemsize);

/* Read the a timestep of a variable into data buffer. */
int netcdf_read_var_ts(int ncid, int varid, int nc_ndims, size_t* nc_dimlen,
                         int tidx, int ts, void* data);