#include <cmath>
#include "coo_tests.hpp"

#include "../coo/coo_utils.hpp"
#include "../dcsr/dcsr_matrix_multiplication.hpp"
#include "../dcsr/dcsr_matrix_multiplication_hash.hpp"

using namespace coo_utils;
using namespace utils;

const uint32_t BINS_NUM = 38;

void test_multiplication() {
    Controls controls = utils::create_controls();
    for (uint32_t k = 10; k < 20; ++k) {
        for (uint32_t i = 10; i < 2000; i += 5) {
            std::cout << "iter = " << i <<  ", i = " << 1035 << ", k = " << k << std::endl;
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
        }
    }
}

void test_multiplication_hash() {
    Controls controls = utils::create_controls();
    timer t;
    for (uint32_t k = 50; k < 60; ++k) {
        for (uint32_t i = 2000; i < 3000; i += 100) {
//            uint32_t i = 2500, k = 36;
            if (DEBUG_ENABLE) *logger << "\n\nITER ------------------------ i = " << i <<
            ", k = " << k << "-----------------------------\n";
            uint32_t max_size = i;
            uint32_t nnz_max = std::max(10u, max_size * k);
            t.restart();
            matrix_dcsr_cpu a_cpu = coo_to_dcsr_cpu(generate_random_matrix_coo_cpu(nnz_max, max_size));
            t.elapsed();
            if (DEBUG_ENABLE) *logger << "a_cpu created in " << t.last_elapsed();


            matrix_dcsr_cpu c_cpu;

            t.restart();
            matrix_multiplication_cpu(c_cpu, a_cpu, a_cpu);
            t.elapsed();
            if (DEBUG_ENABLE) *logger << "matrix_multiplication_cpu in " << t.last_elapsed();

            t.restart();
            matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
            t.elapsed();
            if (DEBUG_ENABLE) *logger << "matrix_dcsr_from_cpu in " << t.last_elapsed();
            matrix_dcsr c_gpu;

            t.restart();
            matrix_multiplication_hash(controls, c_gpu, a_gpu, a_gpu);
            t.elapsed();
            if (DEBUG_ENABLE) *logger << "matrix_multiplication_hash in " << t.last_elapsed();
            compare_matrices(controls, c_gpu, c_cpu);
        }
    }
}