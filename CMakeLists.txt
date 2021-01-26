cmake_minimum_required(VERSION 3.17)
project(SpBench LANGUAGES CXX CUDA)

# Exposed options. Disable something, if do not want to build cuda or cl stuff.
option(BENCH_WITH_CUBOOL "Add cubool lib and related benchmark" ON)

add_library(sp_bench_base INTERFACE)
target_include_directories(sp_bench_base INTERFACE ${CMAKE_CURRENT_LIST_DIR}/src)

# Append here all benchmark targets
set(TARGETS)

# Cubool specific stuff
if (BENCH_WITH_CUBOOL)
    SET(CUBOOL_WITH_CUB ON CACHE BOOL "" FORCE)
    SET(CUBOOL_WITH_NSPARSE ON CACHE BOOL "" FORCE)
    SET(CUBOOL_WITH_NAIVE OFF CACHE BOOL "" FORCE)
    SET(CUBOOL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(thirdparty/cubool)

    add_executable(cubool_mult_add src/cubool_multiply_add.cpp)
    target_link_libraries(cubool_mult_add PUBLIC cubool)
    target_link_libraries(cubool_mult_add PUBLIC sp_bench_base)

    add_executable(cubool_tc src/cubool_transitive_closure.cpp)
    target_link_libraries(cubool_tc PUBLIC cubool)
    target_link_libraries(cubool_tc PUBLIC sp_bench_base)

    list(APPEND TARGETS cubool_mult_add)
    list(APPEND TARGETS cubool_tc)
endif()

foreach(TARGET ${TARGETS})
    message(STATUS "Build target benchmark ${TARGET}")
endforeach()