cmake_minimum_required (VERSION 3.1)
project(sparse-boolean-matrix-operations)

add_subdirectory(libs)

# у меня не работает без явного указания путей
#set(OpenCL_LIBRARY "C:/Windows/System32/OpenCL.dll")
#set(OpenCL_INCLUDE_DIR_LIBRARY "C:/Windows/System32/")
#set(OpenCL_INCLUDE_DIR "libs/clew")

set(Library_classes "src/library_classes")

# Handle OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})
#add_executable (sparse-boolean-matrix-operations src/example/cl/clion_defines.cl src/example/convolution.cpp)
#target_link_libraries (sparse-boolean-matrix-operations ${OpenCL_LIBRARY})


convertIntoHeader(src/cl/merge_path.cl src/cl/headers/merge_path.h merge_path_kernel)
convertIntoHeader(src/cl/prepare_positions.cl src/cl/headers/prepare_positions.h prepare_positions_kernel)
convertIntoHeader(src/cl/coo_bitonic_sort.cl src/cl/headers/coo_bitonic_sort.h coo_bitonic_sort_kernel)
convertIntoHeader(src/cl/set_positions.cl src/cl/headers/set_positions.h set_positions_kernel)
convertIntoHeader(src/cl/prefix_sum.cl src/cl/headers/prefix_sum.h prefix_sum_kernel)
convertIntoHeader(src/cl/coo_kronecker.cl src/cl/headers/coo_kronecker.h coo_kronecker_kernel)
convertIntoHeader(src/cl/dscr_to_coo.cl src/cl/headers/dscr_to_coo.h dscr_to_coo_kernel)
convertIntoHeader(src/cl/for_test/half_sized_scan.cl src/cl/headers/half_sized_scan.h half_sized_scan_kernel)
convertIntoHeader(src/cl/to_result_matrix_single_thread.cl src/cl/headers/to_result_matrix_single_thread.h to_result_matrix_single_thread_kernel)
convertIntoHeader(src/cl/to_result_matrix_work_group.cl src/cl/headers/to_result_matrix_work_group.h to_result_matrix_work_group_kernel)
convertIntoHeader(src/cl/heap_merge.cl src/cl/headers/heap_merge.h heap_merge_kernel)
convertIntoHeader(src/cl/copy_one_value.cl src/cl/headers/copy_one_value.h copy_one_value_kernel)
convertIntoHeader(src/cl/merge_large_rows.cl src/cl/headers/merge_large_rows.h merge_large_rows_kernel)
convertIntoHeader(src/cl/bitonic_esc.cl src/cl/headers/bitonic_esc.h bitonic_esc_kernel)
convertIntoHeader(src/cl/count_workload.cl src/cl/headers/count_workload.h count_workload_kernel)
convertIntoHeader(src/cl/for_test/new_merge.cl src/cl/headers/new_merge.h new_merge_kernel)

set(CLBOOL_SOURCES
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


        #  ------------------------------ cl files -----------------------------
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
        # -----------------------------------------------------------------------

        # ----------------------------- test cl ---------------------------------
        src/cl/for_test/half_sized_scan.cl
        src/cl/for_test/new_merge.cl
        # -----------------------------------------------------------------------


        # ----------------------------- headers ---------------------------------
        src/cl/headers/merge_path.h
        src/cl/headers/prepare_positions.h
        src/cl/headers/coo_bitonic_sort.h
        src/cl/headers/set_positions.h
        src/cl/headers/prefix_sum.h
        src/cl/headers/coo_kronecker.h
        src/cl/headers/dscr_to_coo.h
        src/cl/headers/half_sized_scan.h
        src/cl/headers/to_result_matrix_work_group.h
        src/cl/headers/to_result_matrix_single_thread.h
        src/cl/headers/heap_merge.h
        src/cl/headers/copy_one_value.h
        src/cl/headers/merge_large_rows.h
        src/cl/headers/bitonic_esc.h
        src/cl/headers/count_workload.h
        src/cl/headers/new_merge.h
        # -----------------------------------------------------------------------

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

add_library(clbool SHARED ${CLBOOL_SOURCES})
target_link_libraries(clbool PUBLIC OpenCL)
target_include_directories(clbool PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/)
target_compile_definitions(clbool PUBLIC CL_HPP_ENABLE_EXCEPTIONS)
target_compile_definitions(clbool PUBLIC __CL_ENABLE_EXCEPTIONS)

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


        #  ------------------------------ cl files -----------------------------
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
        # -----------------------------------------------------------------------

        # ----------------------------- test cl ---------------------------------
        src/cl/for_test/half_sized_scan.cl
        src/cl/for_test/new_merge.cl
        # -----------------------------------------------------------------------


        # ----------------------------- headers ---------------------------------
        src/cl/headers/merge_path.h
        src/cl/headers/prepare_positions.h
        src/cl/headers/coo_bitonic_sort.h
        src/cl/headers/set_positions.h
        src/cl/headers/prefix_sum.h
        src/cl/headers/coo_kronecker.h
        src/cl/headers/dscr_to_coo.h
        src/cl/headers/half_sized_scan.h
        src/cl/headers/to_result_matrix_work_group.h
        src/cl/headers/to_result_matrix_single_thread.h
        src/cl/headers/heap_merge.h
        src/cl/headers/copy_one_value.h
        src/cl/headers/merge_large_rows.h
        src/cl/headers/bitonic_esc.h
        src/cl/headers/count_workload.h
        src/cl/headers/new_merge.h
        # -----------------------------------------------------------------------

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

    target_link_libraries (sparse-boolean-matrix-operations ${OpenCL_LIBRARY})

endif()
