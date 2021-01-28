#include "coo_tests.hpp"

#include "../coo/coo_utils.hpp"
#include "../dcsr/dscr_matrix_multiplication.hpp"

using namespace coo_utils;
using namespace utils;
const uint32_t BINS_NUM = 38;

void testHeapAndCopyKernels() {
    Controls controls = utils::create_controls();

    uint32_t nnz_limit = 25;
    uint32_t max_size = 10;

    matrix_dcsr_cpu a_cpu = coo_to_dcsr_cpu(generate_random_matrix_coo_cpu(nnz_limit, max_size));
    matrix_dcsr_cpu b_cpu = coo_to_dcsr_cpu(generate_random_matrix_coo_cpu(nnz_limit + 1, max_size));

    coo_utils::print_matrix(a_cpu);
    coo_utils::print_matrix(b_cpu);

    matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
    matrix_dcsr b_gpu = matrix_dcsr_from_cpu(controls, b_cpu, max_size);

    cl::Buffer nnz_estimation;
    count_workload(controls, nnz_estimation, a_gpu, b_gpu);


    std::vector<cpu_buffer> cpu_workload_groups(BINS_NUM, cpu_buffer());
    cpu_buffer groups_pointers(BINS_NUM + 1);
    cpu_buffer groups_length(BINS_NUM);

    cl::Buffer aux_ptr;
    cl::Buffer aux_mem;

    matrix_dcsr pre;
    build_groups_and_allocate_new_matrix(controls, pre,
                                         cpu_workload_groups, nnz_estimation, a_gpu, b_gpu.nCols(),
                                         aux_ptr, aux_mem
                                         );


    cl::Buffer gpu_workload_groups(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a_gpu.nzr());
    write_bins_info(controls, gpu_workload_groups, cpu_workload_groups, groups_pointers, groups_length);

    std::cout << "pre_rows_pointers: \n"; utils::print_gpu_buffer(controls, pre.rows_pointers_gpu(), a_gpu.nzr() + 1);
    std::cout << "gpu_workload_groups: \n"; utils::print_gpu_buffer(controls, gpu_workload_groups, a_gpu.nzr());
    std::cout << "groups_pointers: \n"; utils::print_cpu_buffer(groups_pointers);
    std::cout << "groups_length: \n"; utils::print_cpu_buffer(groups_length);

    cl::Buffer a;
    cl::Buffer b;

    run_kernels(controls, cpu_workload_groups, groups_length, groups_pointers,
                gpu_workload_groups, nnz_estimation,
                pre, a_gpu, b_gpu,
                a, b
    );


    std::cout << std::endl;
    std::cout << "pre: \n"; print_matrix(controls, pre);

}


void testMultiplication() {
    Controls controls = utils::create_controls();

    uint32_t nnz_limit = 15;
    uint32_t max_size = 30;

    matrix_dcsr_cpu a_cpu = coo_to_dcsr_cpu(generate_random_matrix_coo_cpu(nnz_limit, max_size));
    matrix_dcsr_cpu b_cpu = coo_to_dcsr_cpu(generate_random_matrix_coo_cpu(nnz_limit + 1, max_size));
    matrix_dcsr_cpu c_cpu;

    coo_utils::print_matrix(a_cpu);
    coo_utils::print_matrix(b_cpu);

    matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
    matrix_dcsr b_gpu = matrix_dcsr_from_cpu(controls, b_cpu, max_size);
    matrix_dcsr c_gpu;

    matrix_multiplication_cpu(c_cpu, a_cpu, b_cpu);
    print_matrix(c_cpu);

    matrix_multiplication(controls, c_gpu, a_gpu, b_gpu);
    compare_matrices(controls, c_gpu, c_cpu);
    print_matrix(controls, c_gpu);
}


