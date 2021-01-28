#include "coo_tests.hpp"
#include "../coo/coo_utils.hpp"
#include "../dcsr/dscr_matrix_multiplication.hpp"

using namespace coo_utils;
using namespace utils;
const uint32_t BINS_NUM = 38;

void testScan() {
    Controls controls = create_controls();
    cpu_buffer array(50, 1);
    cl::Buffer array_gpu(controls.context, CL_MEM_READ_WRITE,  array.size() * sizeof(cpu_buffer::value_type));
    controls.queue.enqueueWriteBuffer(array_gpu, CL_TRUE, 0, array.size()
    * sizeof(cpu_buffer::value_type), array.data());
    utils::print_gpu_buffer(controls, array_gpu, array.size());
    cl::Program program;
    try {
        program = controls.create_program_from_file("../src/coo/cl/half_sized_scan.cl");
        uint32_t block_size = 32;

        std::stringstream options;
        options << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, array.size());

        cl::Kernel half_sized_scan_kernel(program, "scan_blelloch_half");
        cl::KernelFunctor<cl::Buffer, uint32_t> half_sized_scan(half_sized_scan_kernel);
        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        half_sized_scan(eargs, array_gpu, array.size());
        utils::print_gpu_buffer(controls, array_gpu, array.size());

    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(controls.device);
        }
        throw std::runtime_error(exception.str());
    }
}




