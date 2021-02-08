#include <cmath>
#include "coo_tests.hpp"

#include "../coo/coo_utils.hpp"
#include "../dcsr/dscr_matrix_multiplication.hpp"

using namespace coo_utils;
using namespace utils;

const uint32_t BINS_NUM = 38;

void test_multiplication() {
    Controls controls = utils::create_controls();
    for (uint32_t k = 18; k < 20; ++k) {
        for (uint32_t i = 1000; i < 2000; i += 5) {
//    for (int i = 0; i < 200; ++i) {
            std::cout << "iter = " << i <<  ", i = " << i << ", k = " << k << std::endl;
            uint32_t max_size = i;
            uint32_t nnz_max = std::max(10u, max_size * k);

            matrix_dcsr_cpu a_cpu = coo_to_dcsr_cpu(generate_random_matrix_coo_cpu(nnz_max, max_size));
//        print_cpu_buffer()
//        matrix_dcsr_cpu b_cpu = coo_to_dcsr_cpu(generate_random_matrix_coo_cpu(nnz_max, max_size - 5));
            matrix_dcsr_cpu c_cpu;
//        print_matrix(a_cpu);
            matrix_multiplication_cpu(c_cpu, a_cpu, a_cpu);

            std::cout << "matrix_multiplication_cpu finished" << std::endl;

            matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
            matrix_dcsr c_gpu;

//        print_matrix(c_cpu, 69);
            std::cout << "s\n";
            matrix_multiplication(controls, c_gpu, a_gpu, a_gpu);
            std::cout << "e\n";
//        print_matrix(controls, c_gpu, 69);
            compare_matrices(controls, c_gpu, c_cpu);
//        }
        }
    }
}

