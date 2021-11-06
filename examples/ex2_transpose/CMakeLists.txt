add_executable(transpose_1o transpose_1o.c)
target_link_libraries(transpose_1o dspaces)

add_executable(transpose_so transpose_so.c)
target_link_libraries(transpose_so dspaces)

add_executable(transpose_mo transpose_mo.c)
target_link_libraries(transpose_mo dspaces)

add_executable(transpose_mso transpose_mso.c)
target_link_libraries(transpose_mso dspaces)

add_executable(cache_hit cache_hit.c)
target_link_libraries(cache_hit dspaces)

add_executable(hybrid_cache_hit hybrid_cache_hit.c)
target_link_libraries(hybrid_cache_hit dspaces)