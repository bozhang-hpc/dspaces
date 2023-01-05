
find_path(OpenVisus_DIR Names OpenVisusConfig.cmake NO_DEFAULT_PATH)
include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenVisus DEFAULT_MSG OpenVisus_DIR)

get_filename_component(OpenVisus_ROOT "${OpenVisus_DIR}/../../../" REALPATH)
MESSAGE(STATUS "OpenVisus_ROOT ${OpenVisus_ROOT} ")

# add_library(OpenVisus::Kernel SHARED IMPORTED GLOBAL)
# set_target_properties(OpenVisus::Kernel  PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${OpenVisus_ROOT}/include/Kernel")
# set_target_properties(OpenVisus::Kernel  PROPERTIES IMPORTED_IMPLIB "${OpenVisus_ROOT}/bin/libVisusKernel.so")

# add_library(OpenVisus::Db SHARED IMPORTED GLOBAL)
# set_target_properties(OpenVisus::Db  PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${OpenVisus_ROOT}/include/Db")
# set_target_properties(OpenVisus::Db  PROPERTIES IMPORTED_IMPLIB "${OpenVisus_ROOT}/bin/libVisusDb.so")
# set_target_properties(OpenVisus::Db  PROPERTIES INTERFACE_LINK_LIBRARIES "OpenVisus::Kernel") 

add_library(OpenVisus::Kernel SHARED IMPORTED GLOBAL)
set_target_properties(OpenVisus::Kernel  PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${OpenVisus_ROOT}/include/Kernel")
set_target_properties(OpenVisus::Kernel  PROPERTIES IMPORTED_LOCATION "${OpenVisus_ROOT}/bin/libVisusKernel.so")

add_library(OpenVisus::Db SHARED IMPORTED GLOBAL)
set_target_properties(OpenVisus::Db  PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${OpenVisus_ROOT}/include/Db")
set_target_properties(OpenVisus::Db  PROPERTIES IMPORTED_LOCATION "${OpenVisus_ROOT}/bin/libVisusDb.so")
set_target_properties(OpenVisus::Db  PROPERTIES INTERFACE_LINK_LIBRARIES "OpenVisus::Kernel") 
