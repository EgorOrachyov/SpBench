#include "coo_kronecker_product.hpp"

#include "../library_classes/controls.hpp"
#include "../library_classes/matrix_coo.hpp"

void kronecker_product(Controls &controls,
                       matrix_coo& matrix_out,
                       const matrix_coo& matrix_a,
                       const matrix_coo& matrix_b) {

    cl::Program program;

    try {

        uint32_t res_size = matrix_a.nnz() * matrix_b.nnz();

        program = controls.create_program_from_file("thirdparty/clbool/coo_kronecker.cl");
        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D RUN " << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, res_size);

        cl::Buffer res_rows(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * res_size);
        cl::Buffer res_cols(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * res_size);

        cl::Kernel coo_kronecker(program, "kronecker");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                uint32_t, uint32_t, uint32_t> coo_kronecker_kernel(coo_kronecker);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        coo_kronecker_kernel(eargs,
                         res_rows, res_cols,
                         matrix_a.rows_indices_gpu(), matrix_a.cols_indices_gpu(),
                         matrix_b.rows_indices_gpu(), matrix_b.cols_indices_gpu(),
                         matrix_b.nnz(), matrix_b.nRows(), matrix_b.nCols());

        matrix_out = matrix_coo(controls, matrix_a.nRows() * matrix_b.nRows(), matrix_a.nCols() * matrix_b.nCols(),
                                res_size);

    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(controls.device);
        }
        throw std::runtime_error(exception.str());
    }

}
