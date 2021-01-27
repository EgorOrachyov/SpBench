cmake_minimum_required(VERSION 3.17)
project(SpBench LANGUAGES CXX CUDA)

# Exposed options. Disable something, if do not want to build cuda or cl stuff.
option(BENCH_WITH_CUBOOL   "Add cubool lib and related benchmarks" ON)
option(BENCH_WITH_CUSP     "Add cusp lib and related benchmarks" ON)
option(BENCH_WITH_CUSPARSE "Add cuSPARSE lib and related benchmarks" ON)
option(BENCH_WITH_CLSPARSE "Add clSPARSE lib and related benchmarks" ON)

add_library(sp_bench_base INTERFACE)
target_include_directories(sp_bench_base INTERFACE ${CMAKE_CURRENT_LIST_DIR}/src)

# Append here all benchmark targets
set(TARGETS)

# Cubool specific stuff
if (BENCH_WITH_CUBOOL)
    set(CUBOOL_WITH_CUB ON CACHE BOOL "" FORCE)
    set(CUBOOL_WITH_NSPARSE ON CACHE BOOL "" FORCE)
    set(CUBOOL_WITH_NAIVE OFF CACHE BOOL "" FORCE)
    set(CUBOOL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(thirdparty/cubool)

    set(CUBOOl_TARGETS)
    add_executable(cubool_mult src/cubool_multiply.cpp)
    add_executable(cubool_add src/cubool_add.cpp)
    add_executable(cubool_mult_add src/cubool_multiply_add.cpp)
    add_executable(cubool_tc src/cubool_transitive_closure.cpp)
    list(APPEND CUBOOl_TARGETS cubool_mult cubool_add cubool_mult_add cubool_tc)

    foreach(CUBOOL_TARGET ${CUBOOl_TARGETS})
        target_link_libraries(${CUBOOL_TARGET} PUBLIC cubool)
        target_link_libraries(${CUBOOL_TARGET} PUBLIC sp_bench_base)

        list(APPEND TARGETS ${CUBOOL_TARGET})
    endforeach()
endif()

# CUSP specifics
if (BENCH_WITH_CUSP)
    add_library(cusp INTERFACE)
    target_include_directories(cusp INTERFACE ${CMAKE_CURRENT_LIST_DIR}/thirdparty/cusp)

    set(CUSP_TARGETS)

    add_executable(cusp_basic src/cusp_basic.cu)
    add_executable(cusp_mult src/cusp_multiply.cu)
    add_executable(cusp_add src/cusp_add.cu)
    add_executable(cusp_mult_add src/cusp_multiply_add.cu)
    list(APPEND CUSP_TARGETS cusp_basic cusp_mult cusp_add cusp_mult_add)

    foreach(CUSP_TARGET ${CUSP_TARGETS})
        target_link_libraries(${CUSP_TARGET} PUBLIC cusp)
        target_link_libraries(${CUSP_TARGET} PUBLIC sp_bench_base)

        target_compile_features(${CUSP_TARGET} PUBLIC cxx_std_14)

        set_target_properties(${CUSP_TARGET} PROPERTIES CXX_STANDARD 17)
        set_target_properties(${CUSP_TARGET} PROPERTIES CXX_STANDARD_REQUIRED ON)
        set_target_properties(${CUSP_TARGET} PROPERTIES CUDA_STANDARD 14)
        set_target_properties(${CUSP_TARGET} PROPERTIES CUDA_STANDARD_REQUIRED ON)
        set_target_properties(${CUSP_TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        list(APPEND TARGETS ${CUSP_TARGET})
    endforeach()
endif()

if (BENCH_WITH_CUSPARSE)
    set(CUSPARSE_TARGETS)

    find_package(CUDA REQUIRED)

    add_executable(cusparse_mult src/cusparse_multiply.cu)
    add_executable(cusparse_add src/cusparse_add.cu)
    list(APPEND CUSPARSE_TARGETS cusparse_add cusparse_mult)

    foreach(CUSPARSE_TARGET ${CUSPARSE_TARGETS})
        target_link_libraries(${CUSPARSE_TARGET} PUBLIC sp_bench_base)
        target_link_libraries(${CUSPARSE_TARGET} PUBLIC cusparse)

        target_compile_features(${CUSPARSE_TARGET} PUBLIC cxx_std_14)
        set_target_properties(${CUSPARSE_TARGET} PROPERTIES CXX_STANDARD 17)
        set_target_properties(${CUSPARSE_TARGET} PROPERTIES CXX_STANDARD_REQUIRED ON)
        set_target_properties(${CUSPARSE_TARGET} PROPERTIES CUDA_STANDARD 14)
        set_target_properties(${CUSPARSE_TARGET} PROPERTIES CUDA_STANDARD_REQUIRED ON)
        set_target_properties(${CUSPARSE_TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        list(APPEND TARGETS ${CUSP_TARGET})
    endforeach()
endif()

if (BENCH_WITH_CLSPARSE)
    set(BUILD_clSPARSE ON CACHE BOOL "" FORCE)
    set(BUILD_Boost OFF CACHE BOOL "" FORCE)
    set(BUILD_gMock ON CACHE BOOL "" FORCE)
    set(BUILD_MTX OFF CACHE BOOL "" FORCE)
    set(BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
    set(USE_SYSTEM_CL2HPP ON CACHE BOOL "" FORCE)
    add_subdirectory(thirdparty/clSPARSE)

    set(CLSPARSE_TARGETS)

    add_executable(clsparse_basic src/clsparse_basic.cpp)
    add_executable(clsparse_mult src/clsparse_multiply.cpp)
    list(APPEND CLSPARSE_TARGETS clsparse_basic clsparse_mult)

    set(clSPARSER_PATH ${CMAKE_BINARY_DIR}/thirdparty/clSPARSE/clSPARSE-build)

    find_package(OpenCL REQUIRED)

    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(clSPARSE_LIB_NAME clSPARSEd)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(clSPARSE_LIB_NAME clSPARSEr)
    else()
        message(SEND_ERROR "Unknown build type")
    endif()

    foreach(CLSPARSE_TARGET ${CLSPARSE_TARGETS})
        target_link_libraries(${CLSPARSE_TARGET} PUBLIC sp_bench_base)
        target_link_libraries(${CLSPARSE_TARGET} PUBLIC OpenCL)
        target_link_libraries(${CLSPARSE_TARGET} PUBLIC ${clSPARSE_LIB_NAME})
        target_link_directories(${CLSPARSE_TARGET} PUBLIC ${clSPARSER_PATH}/library)
        target_include_directories(${CLSPARSE_TARGET} PUBLIC thirdparty/clSPARSE/src/include)
        target_include_directories(${CLSPARSE_TARGET} PUBLIC ${clSPARSER_PATH}/library)

        target_compile_features(${CLSPARSE_TARGET} PUBLIC cxx_std_14)

        set_target_properties(${CLSPARSE_TARGET} PROPERTIES CXX_STANDARD 17)
        set_target_properties(${CLSPARSE_TARGET} PROPERTIES CXX_STANDARD_REQUIRED ON)

        list(APPEND TARGETS ${CLSPARSE_TARGET})
    endforeach()
endif()

# Some fancy stuff here
foreach(TARGET ${TARGETS})
    message(STATUS "Build target benchmark ${TARGET}")
endforeach()

# Copy data into build directory
file(COPY ${CMAKE_CURRENT_LIST_DIR}/data/ DESTINATION ${CMAKE_BINARY_DIR}/data/)

# Unpack data for benchmarks into build directory
add_custom_target(unzip_dataset ALL)
add_custom_command(TARGET unzip_dataset PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_LIST_DIR}/dataset/dataset.zip
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/data
        DEPENDS ${CMAKE_SOURCE_DIR}/dataset/dataset.zip
        COMMENT "Unpacking benchmark dataset from dataset.zip"
        VERBATIM)