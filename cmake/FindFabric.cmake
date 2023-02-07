# Use pkg-config to find libfabric
#  only needs the include dir 
#  to use the fi_* functions
find_package(PkgConfig)
pkg_check_modules(Fabric libfabric)
if(NOT Fabric_FOUND)
    find_path(Fabric_INCLUDE_DIR rdma/fabric.h
    HINTS ${Fabric_INCLUDEDIR} ${Fabric_INCLUDE_DIRS})
    set(Fabric_INCLUDE_DIRS ${Fabric_INCLUDE_DIR})
endif()