#include "matrices_conversions.hpp"

#include "../cl/headers/dscr_to_coo.h"
#include "../cl/headers/prepare_positions.h"
#include "../cl/headers/set_positions.h"

matrix_coo dcsr_to_coo(Controls &controls, matrix_dcsr &a) {
    cl::Buffer c_rows_indices(controls.context, CL_MEM_READ_WRITE, sizeof(matrix_dcsr::index_type) * a.nnz());

    auto dscr_to_coo = program<cl::Buffer, cl::Buffer, cl::Buffer>(dscr_to_coo_kernel, dscr_to_coo_kernel_length)
            .set_kernel_name("dscr_to_coo")
            .set_block_size(64)
            .set_needed_work_size(a.nzr() * 64);

    dscr_to_coo.run(controls, a.rows_pointers_gpu(), a.rows_compressed_gpu(), c_rows_indices);
    return matrix_coo(a.nRows(), a.nCols(), a.nnz(), c_rows_indices, a.cols_indices_gpu());
}

namespace {
    void create_rows_pointers(Controls &controls,
                              cl::Buffer &rows_pointers_out,
                              cl::Buffer &rows_compressed_out,
                              const cl::Buffer &rows,
                              uint32_t size,
                              uint32_t &nzr // non zero rows
    ) {

        cl::Buffer positions(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);

        auto prepare_positions = program<cl::Buffer, cl::Buffer, uint32_t>(prepare_positions_kernel, prepare_positions_kernel_length)
                .set_kernel_name("prepare_array_for_rows_positions")
                .set_needed_work_size(size);
        prepare_positions.run(controls, positions, rows, size);

        prefix_sum(controls, positions, nzr, size);

        cl::Buffer rows_pointers(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (nzr + 1));
        cl::Buffer rows_compressed(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * nzr);

        auto set_positions = program<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, uint32_t, uint32_t>(
                 set_positions_kernel, set_positions_kernel_length)
                .set_kernel_name("set_positions_rows")
                .set_needed_work_size(size);

        set_positions.run(controls, rows_pointers, rows_compressed, rows, positions, size, nzr);

        rows_pointers_out = std::move(rows_pointers);
        rows_compressed_out = std::move(rows_compressed);
    }
}

matrix_dcsr coo_to_dcsr_gpu(Controls &controls, const matrix_coo &a) {
    cl::Buffer rows_pointers;
    cl::Buffer rows_compressed;
    uint32_t nzr;
    create_rows_pointers(controls, rows_pointers, rows_compressed, a.rows_indices_gpu(), a.nnz(), nzr);

    return matrix_dcsr(rows_pointers, rows_compressed, a.cols_indices_gpu(),
                       a.nRows(), a.nCols(), a.nnz(), nzr
    );
}

matrix_dcsr matrix_dcsr_from_cpu(Controls &controls, matrix_dcsr_cpu &m, uint32_t size) {

    cl::Buffer rows_pointers(controls.context, m.rows_pointers().begin(), m.rows_pointers().end(), false);
    cl::Buffer rows_compressed(controls.context, m.rows_compressed().begin(), m.rows_compressed().end(), false);
    cl::Buffer cols_indices(controls.context, m.cols_indices().begin(), m.cols_indices().end(), false);

    return matrix_dcsr(rows_pointers, rows_compressed, cols_indices,
                       size, size, m.cols_indices().size(), m.rows_compressed().size());

}

matrix_coo matrix_coo_from_cpu(Controls &controls, matrix_coo_cpu &m, uint32_t size) {

    cl::Buffer rows_indices(controls.context, m.rows_indices().begin(), m.rows_indices().end(), false);
    cl::Buffer cols_indices(controls.context, m.cols_indices().begin(), m.cols_indices().end(), false);

    return matrix_coo(size, size, m.rows_indices().size(), rows_indices, cols_indices);
}

matrix_dcsr_cpu matrix_dcsr_from_gpu(Controls &controls, matrix_dcsr &m) {

    cpu_buffer rows_pointers(m.nzr() + 1);
    cpu_buffer rows_compressed(m.nzr());
    cpu_buffer cols_indices(m.nnz());

    controls.queue.enqueueReadBuffer(m.rows_pointers_gpu(), CL_TRUE, 0,
                                     sizeof(matrix_dcsr::index_type) * rows_pointers.size(), rows_pointers.data());
    controls.queue.enqueueReadBuffer(m.rows_compressed_gpu(), CL_TRUE, 0,
                                     sizeof(matrix_dcsr::index_type) * rows_compressed.size(), rows_compressed.data());
    controls.queue.enqueueReadBuffer(m.cols_indices_gpu(), CL_TRUE, 0,
                                     sizeof(matrix_dcsr::index_type) * cols_indices.size(), cols_indices.data());

    return matrix_dcsr_cpu(rows_pointers, rows_compressed, cols_indices);

}

matrix_coo_cpu matrix_coo_from_gpu(Controls &controls, matrix_coo &m) {

    cpu_buffer rows_indices(m.nnz());
    cpu_buffer cols_indices(m.nnz());

    controls.queue.enqueueReadBuffer(m.rows_indices_gpu(), CL_TRUE, 0,
                                     sizeof(matrix_dcsr::index_type) * rows_indices.size(), rows_indices.data());
    controls.queue.enqueueReadBuffer(m.cols_indices_gpu(), CL_TRUE, 0,
                                     sizeof(matrix_dcsr::index_type) * cols_indices.size(), cols_indices.data());

    return matrix_coo_cpu(rows_indices, cols_indices);
}