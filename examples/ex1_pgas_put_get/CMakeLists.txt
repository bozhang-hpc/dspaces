add_executable(put put.c)
target_link_libraries(put dspaces)

add_executable(get get.c)
target_link_libraries(get dspaces)