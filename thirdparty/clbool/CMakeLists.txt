cmake_minimum_required(VERSION 3.1)
project(sparse-boolean-matrix-operations)

# у меня не работает без явного указания путей
#set(OpenCL_LIBRARY "C:/Windows/System32/OpenCL.dll")
#set(OpenCL_INCLUDE_DIR_LIBRARY "C:/Windows/System32/")
#set(OpenCL_INCLUDE_DIR "libs/clew")

set(Library_classes "src/library_classes")

set(CLBOOL_SOURCES
        src/library_classes/matrix_coo.cpp

        src/coo/coo_matrix_addition.cpp
        src/coo/coo_initialization.cpp
        src/coo/coo_utils.cpp
        src/coo/coo_kronecker_product.cpp
        src/dcsr/dscr_matrix_multiplication.cpp
        src/common/matrices_conversions.cpp
        src/common/cl_operations.cpp
        src/common/utils.cpp

        src/cl/prepare_positions.cl
        src/cl/merge_path.cl
        src/cl/coo_kronecker.cl
        src/cl/prefix_sum.cl
        src/cl/clion_defines.cl
        src/cl/set_positions.cl
        src/cl/coo_bitonic_sort.cl
        src/cl/count_workload.cl
        src/cl/heap_merge.cl
        src/cl/bitonic_esc.cl
        src/cl/copy_one_value.cl
        src/cl/to_result_matrix_work_group.cl
        src/cl/to_result_matrix_single_thread.cl
        src/cl/merge_large_rows.cl
        src/cl/dscr_to_coo.cl

        src/cl/for_test/half_sized_scan.cl
        src/cl/for_test/new_merge.cl
        )

add_library(clbool ${CLBOOL_SOURCES})
target_link_libraries(clbool PUBLIC OpenCL)
target_include_directories(clbool PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src)
target_compile_definitions(clbool PUBLIC CL_HPP_ENABLE_EXCEPTIONS)
target_compile_definitions(clbool PUBLIC __CL_ENABLE_EXCEPTIONS)


# Handle OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})
#add_executable (sparse-boolean-matrix-operations src/example/cl/clion_defines.cl src/example/convolution.cpp)
#target_link_libraries (sparse-boolean-matrix-operations ${OpenCL_LIBRARY})

if (BUILD_EXECUTABLE)
add_executable(sparse-boolean-matrix-operations
        src/library_classes/matrix_coo.cpp

        src/coo/main_coo.cpp
        src/coo/coo_matrix_addition.cpp
        src/coo/coo_initialization.cpp
        src/coo/coo_utils.cpp
        src/coo/coo_kronecker_product.cpp
        src/dcsr/dscr_matrix_multiplication.cpp
        src/common/matrices_conversions.cpp
        src/common/cl_operations.cpp
        src/common/utils.cpp

        src/cl/prepare_positions.cl
        src/cl/merge_path.cl
        src/cl/coo_kronecker.cl
        src/cl/prefix_sum.cl
        src/cl/clion_defines.cl
        src/cl/set_positions.cl
        src/cl/coo_bitonic_sort.cl
        src/cl/count_workload.cl
        src/cl/heap_merge.cl
        src/cl/bitonic_esc.cl
        src/cl/copy_one_value.cl
        src/cl/to_result_matrix_work_group.cl
        src/cl/to_result_matrix_single_thread.cl
        src/cl/merge_large_rows.cl
        src/cl/dscr_to_coo.cl

        src/cl/for_test/half_sized_scan.cl
        src/cl/for_test/new_merge.cl


        src/test/coo_addition_test.cpp
        src/test/coo_bitonic_test.cpp
        src/test/coo_reduce_duplicates_test.cpp
        src/test/coo_kronecker_test.cpp
        src/test/coo_to_dcsr_test.cpp
        src/test/count_workload_test.cpp
        src/test/count_workload_and_allocation_test.cpp
        src/test/heap_kernels_test.cpp
        src/test/large_rows_test.cpp
        src/test/check_objects_copying.cpp
        src/test/test_half_sized_scan.cpp
        src/test/esc_kernels_test.cpp
        src/test/test_new_merge.cpp
        src/test/test_multiplication.cpp
        )

target_link_libraries(sparse-boolean-matrix-operations ${OpenCL_LIBRARY})

# у меня этот макрос через раз отрабатывает, вроде файл поменяла, а он не скопировался
# copy cl files to binary folder
FILE(GLOB CL_FILES src/cl/*.cl)
foreach (cl_file ${CL_FILES})
    add_custom_command(
            TARGET sparse-boolean-matrix-operations
            POST_BUILD
            COMMAND ${CMAKE_COMMAND}
            ARGS -E copy ${cl_file} ${PROJECT_BINARY_DIR}/.
    )
endforeach (cl_file)
endif()

foreach (cl_file ${CL_FILES})
    add_custom_command(TARGET clbool
                       PRE_BUILD
                       COMMAND ${CMAKE_COMMAND}
                       ARGS -E copy ${cl_file} ${PROJECT_BINARY_DIR}/.)
endforeach (cl_file)
