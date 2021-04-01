add_executable(transpose_1o transpose_1o.c)
target_link_libraries(transpose_1o dspaces)

add_executable(transpose_so transpose_so.c)
target_link_libraries(transpose_so dspaces)

add_executable(transpose_mo transpose_mo.c)
target_link_libraries(transpose_mo dspaces)