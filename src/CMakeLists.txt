find_package(DRC)
if(HAVE_DRC)
    include_directories(${DRC_INCLUDE_DIRS})
    add_definitions( -DHAVE_DRC)
endif()

# need NetCDF lib
find_package(NetCDF REQUIRED)

include(OpenVisusConfig)

######## dspaces client start ########

# list of source files
set(dspaces-src util.c bbox.c ss_data.c dspaces-client.c)
add_library(dspaces ${dspaces-src})

target_link_libraries (dspaces margo m pthread)
target_include_directories (dspaces PUBLIC $<INSTALL_INTERFACE:include>)
if(HAVE_DRC)
    target_link_libraries (dspaces ${DRC_LIBRARIES})
    target_include_directories (dspaces PUBLIC ${DRC_INCLUDE_DIRS})
endif()

# local include's BEFORE, in case old incompatable .h files in prefix/include
target_include_directories (dspaces BEFORE PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

# for shared libs, establish the lib version
set_target_properties (dspaces
    PROPERTIES VERSION ${dspaces_VERSION}
    SOVERSION ${dspaces_VERSION_MAJOR})

######## dspaces server start ########

# list of source files
set(dspaces-server-src util.c bbox.c ss_data.c dspaces-server.c
    toml.c netcdf_wrapper.c idx1_wrapper.cpp) # add idx_wrapper.cpp later
add_library(dspaces-server ${dspaces-server-src})

target_link_libraries (dspaces-server margo m pthread)
target_link_libraries (dspaces-server ${NetCDF_LIBRARIES})
target_link_libraries (dspaces-server OpenVisus::Kernel OpenVisus::Db)
target_include_directories (dspaces-server BEFORE PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
target_include_directories (dspaces-server BEFORE PUBLIC ${NetCDF_INCLUDE_DIRS})
if(HAVE_DRC)
    target_link_libraries (dspaces-server ${DRC_LIBRARIES})
    target_include_directories (dspaces-server BEFORE PUBLIC ${DRC_INCLUDE_DIRS})
endif()
target_compile_features(dspaces-server PUBLIC cxx_std_17)

# for shared libs, establish the lib version
set_target_properties (dspaces-server
    PROPERTIES VERSION ${dspaces_VERSION}
    SOVERSION ${dspaces_VERSION_MAJOR})


# load package helper for generating cmake CONFIG packages
include (CMakePackageConfigHelpers)

# where to install files for "find_package"
set (dspaces-pkg "share/cmake/dspaces")

set (dspaces-vers "${dspaces_VERSION_MAJOR}.${dspaces_VERSION_MINOR}")

#
# installation stuff (packaging and install commands)
#
write_basic_package_version_file(
    "dspaces-config-version.cmake"
    VERSION ${dspaces_VERSION}
    COMPATIBILITY AnyNewerVersion)

# generate our config file for find_package()
configure_file (dspaces-config.cmake.in dspaces-config.cmake @ONLY)

# some bits for the pkg-config file
set (DEST_DIR "${CMAKE_INSTALL_PREFIX}")
set (PRIVATE_LIBS "-ldspaces -lpthread -lmargo -labt -lmercury -lm -lnetcdf")
configure_file ("dspaces.pc.in" "dspaces.pc" @ONLY)
configure_file ("dspaces-server.pc.in" "dspaces-server.pc" @ONLY)

#
# "make install" rules
#
install (TARGETS dspaces dspaces-server EXPORT dspaces-targets
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib)
install (EXPORT dspaces-targets NAMESPACE dspaces::
         DESTINATION ${dspaces-pkg}
         FILE "dspaces-targets.cmake")
install (FILES "${CMAKE_CURRENT_BINARY_DIR}/dspaces-config.cmake"
               "${CMAKE_CURRENT_BINARY_DIR}/dspaces-config-version.cmake"
               "../cmake/xpkg-import.cmake"
         DESTINATION ${dspaces-pkg} )
install (FILES "${CMAKE_CURRENT_BINARY_DIR}/dspaces.pc"
		DESTINATION "lib/pkgconfig/")
install (FILES "${CMAKE_CURRENT_BINARY_DIR}/dspaces-server.pc"
        DESTINATION "lib/pkgconfig/")   
