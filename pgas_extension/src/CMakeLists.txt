# list of source files
set(dspaces_pgas_src pgas_bbox.c pgas_ss_data.c dspaces_pgas_client.c dspaces_pgas_server.c)

# load package helper for generating cmake CONFIG packages
include (CMakePackageConfigHelpers)

target_sources(dspaces PRIVATE ${dspaces_pgas_src})
target_include_directories (dspaces PUBLIC $<INSTALL_INTERFACE:include>)

# local include's BEFORE, in case old incompatable .h files in prefix/include
target_include_directories (dspaces BEFORE PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)