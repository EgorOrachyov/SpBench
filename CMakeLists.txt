cmake_minimum_required(VERSION 3.17)
project(SpBench LANGUAGES CXX CUDA)

# Exposed options. Disable something, if do not want to build cuda or cl stuff.
option(BENCH_WITH_CUBOOL "Add cubool lib and related benchmarks" ON)
option(BENCH_WITH_CUSP   "Add cusp lib and related benchmarks" ON)

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

# CUSP specifics
if (BENCH_WITH_CUSP)
    add_library(cusp INTERFACE)
    target_include_directories(cusp INTERFACE ${CMAKE_CURRENT_LIST_DIR}/thirdparty/cusp)

    add_executable(cusp_basic src/cusp_basic.cu)
    target_link_libraries(cusp_basic PUBLIC cusp)
    target_link_libraries(cusp_basic PUBLIC sp_bench_base)

    target_compile_features(cusp_basic PUBLIC cxx_std_14)

    set_target_properties(cusp_basic PROPERTIES CXX_STANDARD 17)
    set_target_properties(cusp_basic PROPERTIES CXX_STANDARD_REQUIRED ON)
    set_target_properties(cusp_basic PROPERTIES CUDA_STANDARD 14)
    set_target_properties(cusp_basic PROPERTIES CUDA_STANDARD_REQUIRED ON)
    set_target_properties(cusp_basic PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    list(APPEND TARGETS cusp_mult_add_float)
endif()

foreach(TARGET ${TARGETS})
    message(STATUS "Build target benchmark ${TARGET}")
endforeach()