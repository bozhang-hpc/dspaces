# - Try to find NetCDF
# Once done this will define
#  NETCDF_FOUND - System has NetCDF
#  NETCDF_INCLUDE_DIRS - The NetCDF include directories
#  NETCDF_LIBRARIES - The libraries needed to use NetCDF

find_package(PkgConfig)
pkg_check_modules(PC_NETCDF REQUIRED netcdf)

find_path(NetCDF_INCLUDE_DIR netcdf.h
  HINTS ${PC_NETCDF_INCLUDEDIR} ${PC_NETCDF_INCLUDE_DIRS})

find_library(NetCDF_LIBRARY NAMES netcdf
  HINTS ${PC_NETCDF_LIBDIR} ${PC_NETCDF_LIBRARY_DIRS})

set(NetCDF_INCLUDE_DIRS ${NetCDF_INCLUDE_DIR})
set(NetCDF_LIBRARIES ${NetCDF_LIBRARY})
include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set NETCDF_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(NetCDF DEFAULT_MSG
                                  NetCDF_INCLUDE_DIR NetCDF_LIBRARY)

mark_as_advanced(NetCDF_INCLUDE_DIR NetCDF_LIBRARY)