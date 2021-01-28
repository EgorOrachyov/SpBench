#pragma once

#include <vector>
#include "../library_classes/controls.hpp"
#include "../library_classes/matrix_coo.hpp"
#include "../library_classes/matrix_dcsr.hpp"
#include "../library_classes/cpu_matrices.hpp"
#include "../common/utils.hpp"

namespace coo_utils {

    void check_correctness(const cpu_buffer &rows, const cpu_buffer &cols);

    void fill_random_matrix(cpu_buffer &rows, cpu_buffer &cols, uint32_t max_size = 1024);

    void
    form_cpu_matrix(matrix_coo_cpu_pairs &matrix_out, const cpu_buffer &rows, const cpu_buffer &cols);

    void get_vectors_from_cpu_matrix(cpu_buffer &rows_out, cpu_buffer &cols_out,
                                     const matrix_coo_cpu_pairs &matrix);

    matrix_coo_cpu_pairs generate_random_matrix_coo_cpu(uint32_t pseudo_nnz, uint32_t max_size = 1024);

    matrix_coo matrix_coo_from_cpu(Controls &controls, const matrix_coo_cpu_pairs &m_cpu);

    void
    matrix_addition_cpu(matrix_coo_cpu_pairs &matrix_out, const matrix_coo_cpu_pairs &matrix_a, const matrix_coo_cpu_pairs &matrix_b);

    void
    kronecker_product_cpu(matrix_coo_cpu_pairs &matrix_out, const matrix_coo_cpu_pairs &matrix_a, const matrix_coo_cpu_pairs &matrix_b);

    void print_matrix(const matrix_coo_cpu_pairs &m_cpu);

    void get_rows_pointers_and_compressed(cpu_buffer &rows_pointers,
                                          cpu_buffer &rows_compressed,
                                          const matrix_coo_cpu_pairs &matrix_cpu);



    // cpu class for double compressed matrix


    void matrix_multiplication_cpu(matrix_dcsr_cpu &c,
                                   const matrix_dcsr_cpu &a,
                                   const matrix_dcsr_cpu &b);

    matrix_dcsr_cpu coo_to_dcsr_cpu(const matrix_coo_cpu_pairs &matrix_coo);

    void get_workload(cpu_buffer &workload,
                      const matrix_dcsr_cpu &a_cpu,
                      const matrix_dcsr_cpu &b_cpu);


    std::pair<matrix_dcsr_cpu, matrix_dcsr_cpu> generate_random_matrices_esc(uint32_t max_size, uint32_t seed);
    std::pair<matrix_dcsr_cpu, matrix_dcsr_cpu> generate_random_matrices_large(uint32_t max_size, uint32_t seed);


    void print_matrix(const matrix_dcsr_cpu &m_cpu, uint32_t index = -1);
    void print_matrix(Controls &controls, const matrix_dcsr& m_gpu, uint32_t index = -1);
}
