#include <vector>
#include <cstddef>
#include <cstdint>


#include "../common/utils.hpp"
#include "libutils/fast_random.h"
#include "coo_initialization.hpp"

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cmath>

#include "../cl/headers/coo_bitonic_sort.h"

void sort_arrays(Controls &controls, cl::Buffer &rows_gpu, cl::Buffer &cols_gpu, uint32_t n) {
    cl::Program program;

    try {
        program = controls.create_program_from_source(coo_bitonic_sort_kernel, coo_bitonic_sort_kernel_length);

        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D RUN " << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        uint32_t work_group_size = block_size;
        // a bitonic sort needs 2 time less threads than values in array to sort
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, utils::round_to_power2(n));

        cl::Kernel coo_bitonic_begin_kernel(program, "local_bitonic_begin");
        cl::Kernel coo_bitonic_global_step_kernel(program, "bitonic_global_step");
        cl::Kernel coo_bitonic_end_kernel(program, "bitonic_local_endings");

        cl::KernelFunctor<cl::Buffer, cl::Buffer, uint32_t>
                coo_bitonic_begin(coo_bitonic_begin_kernel);
        cl::KernelFunctor<cl::Buffer, cl::Buffer, uint32_t, uint32_t, uint32_t>
                coo_bitonic_global_step(coo_bitonic_global_step_kernel);
        cl::KernelFunctor<cl::Buffer, cl::Buffer, uint32_t>
                coo_bitonic_end(coo_bitonic_end_kernel);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));
        // ----------------------------------------------- main cycle -----------------------------------------------
        coo_bitonic_begin(eargs, rows_gpu, cols_gpu, n);

        uint32_t segment_length = work_group_size * 2 * 2;

//        // TODO : power function for unsigned?
        uint32_t outer = utils::ceil_to_power2(ceil(n * 1.0 / (work_group_size * 2)));

        while (outer != 1) {
            coo_bitonic_global_step(eargs, rows_gpu, cols_gpu, segment_length, 1, n);
            for (unsigned int i = segment_length / 2; i > work_group_size * 2; i >>= 1) {
                coo_bitonic_global_step(eargs, rows_gpu, cols_gpu, i, 0, n);
            }
            coo_bitonic_end(eargs, rows_gpu, cols_gpu, n);
            outer >>= 1;
            segment_length <<= 1;
        }
        std::cout << "\nbitonic sort finished" << std::endl;
    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(controls.device);
        }
        throw std::runtime_error(exception.str());
    }
}
