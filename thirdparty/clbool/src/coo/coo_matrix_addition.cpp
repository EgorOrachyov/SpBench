
#include "../library_classes/controls.hpp"
#include "../common/utils.hpp"
#include "coo_utils.hpp"
#include "coo_matrix_addition.hpp"


void matrix_addition(Controls &controls,
                     matrix_coo &matrix_out,
                     const matrix_coo &a,
                     const matrix_coo &b) {

    cl::Buffer merged_rows;
    cl::Buffer merged_cols;
    uint32_t new_size;

    merge(controls, merged_rows, merged_cols, a, b);

    reduce_duplicates(controls, merged_rows, merged_cols, new_size, a.nnz() + b.nnz());

    matrix_out = matrix_coo(controls, a.nRows(), a.nCols(), new_size, merged_rows, merged_cols);
}


void merge(Controls &controls,
           cl::Buffer &merged_rows_out,
           cl::Buffer &merged_cols_out,
           const matrix_coo &a,
           const matrix_coo &b) {

    cl::Program program;

    try {

        uint32_t merged_size = a.nnz() + b.nnz();

        program = controls.create_program_from_file("thirdparty/clbool/merge_path.cl");

        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D RUN " << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, merged_size);

        cl::Buffer merged_rows(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);
        cl::Buffer merged_cols(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);

        cl::Kernel coo_merge(program, "merge");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                uint32_t, uint32_t> coo_merge_kernel(coo_merge);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        coo_merge_kernel(eargs,
                         merged_rows, merged_cols,
                         a.rows_indices_gpu(), a.cols_indices_gpu(),
                         b.rows_indices_gpu(), b.cols_indices_gpu(),
                         a.nnz(), b.nnz());

        // TODO: maybe add wait
//        check_merge_correctness(controls, merged_rows, merged_cols, merged_size);

        merged_rows_out = std::move(merged_rows);
        merged_cols_out = std::move(merged_cols);
//        std::cout << "\nmerge finished\n";
    } catch (const cl::Error &e) {
        utils::program_handler(e, program, controls.device, "merge");
    }
}


void reduce_duplicates(Controls &controls,
                       cl::Buffer &merged_rows,
                       cl::Buffer &merged_cols,
                       uint32_t &new_size,
                       uint32_t merged_size
) {
    // ------------------------------------ prepare array to count positions ----------------------

    cl::Buffer positions(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);

    prepare_positions(controls, positions, merged_rows, merged_cols, merged_size);

    // ------------------------------------ calculate positions, get new_size -----------------------------------

    prefix_sum(controls, positions, new_size, merged_size);

    cl::Buffer new_rows(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * new_size);
    cl::Buffer new_cols(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * new_size);

    set_positions(controls, new_rows, new_cols, merged_rows, merged_cols, positions, merged_size);

    merged_rows = std::move(new_rows);
    merged_cols = std::move(new_cols);

}


void prepare_positions(Controls &controls,
                       cl::Buffer &positions,
                       cl::Buffer &merged_rows,
                       cl::Buffer &merged_cols,
                       uint32_t merged_size
) {
    cl::Program program;
    try {
        program = controls.create_program_from_file("thirdparty/clbool/prepare_positions.cl");
        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D RUN " << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, merged_size);

        cl::Kernel coo_prepare_positions_kernel(program, "prepare_array_for_positions");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t> coo_prepare_positions(
                coo_prepare_positions_kernel);
        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        coo_prepare_positions(eargs, positions, merged_rows, merged_cols, merged_size);

    } catch (const cl::Error &e) {
        utils::program_handler(e, program, controls.device, "prepare_positions");
    }

}




void set_positions(Controls &controls,
                   cl::Buffer &new_rows,
                   cl::Buffer &new_cols,
                   cl::Buffer &merged_rows,
                   cl::Buffer &merged_cols,
                   cl::Buffer &positions,
                   uint32_t merged_size) {

    cl::Program program;
    try {
        program = controls.create_program_from_file("thirdparty/clbool/set_positions.cl");
        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D RUN " << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        cl::Kernel set_positions_kernel(program, "set_positions");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, unsigned int> set_positions(
                set_positions_kernel);

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, merged_size);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        set_positions(eargs, new_rows, new_cols, merged_rows, merged_cols, positions, merged_size);

    } catch (const cl::Error &e) {
        utils::program_handler(e, program, controls.device, "set_positions");
    }
}


void check_pref_correctness(const std::vector<uint32_t> &result,
                            const std::vector<uint32_t> &before) {
    uint32_t n = before.size();
    uint32_t acc = 0;

    for (uint32_t i = 0; i < n; ++i) {
        acc = i == 0 ? 0 : before[i - 1] + acc;

        if (acc != result[i]) {
            throw std::runtime_error("incorrect result");
        }
    }
    std::cout << "correct pref sum, the last value is " << result[n - 1] << std::endl;
}


// check weak correctness
void check_merge_correctness(Controls &controls, cl::Buffer &rows, cl::Buffer &cols, uint32_t merged_size) {
    std::vector<uint32_t> rowsC(merged_size);
    std::vector<uint32_t> colsC(merged_size);

    controls.queue.enqueueReadBuffer(rows, CL_TRUE, 0, sizeof(uint32_t) * merged_size, rowsC.data());
    controls.queue.enqueueReadBuffer(cols, CL_TRUE, 0, sizeof(uint32_t) * merged_size, colsC.data());

    coo_utils::check_correctness(rowsC, colsC);
}
