#include "cl_operations.hpp"

#include "../cl/headers/prefix_sum.h"

//namespace {
//    auto prefix_sum = program<cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl::Buffer, unsigned int>
//            (prefix_sum_kernel, prefix_sum_kernel_length)
//            .set_kernel_name();
//
//}

void prefix_sum(Controls &controls,
                cl::Buffer &array,
                uint32_t &total_sum,
                uint32_t array_size) {
    cl::Program program;
    try {
        program = controls.create_program_from_source(prefix_sum_kernel, prefix_sum_kernel_length);
        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D RUN " << "-D GROUP_SIZE=" << block_size;

        program.build(options.str().c_str());

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, array_size);

        uint32_t a_size = (array_size + block_size - 1) / block_size; // max to save first roots
        uint32_t b_size = (a_size + block_size - 1) / block_size; // max to save second roots

        cl::Buffer a_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a_size);
        cl::Buffer b_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * b_size);
        cl::Buffer total_sum_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t));

        cl::LocalSpaceArg local_array = cl::Local(sizeof(uint32_t) * block_size);

        // prefix sum step kernel
        cl::Kernel scan_kernel(program, "scan_blelloch");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl::Buffer, unsigned int> scan(scan_kernel);

        cl::Kernel update_kernel(program, "update_pref_sum");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, unsigned int, unsigned int> update(update_kernel);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        uint32_t leaf_size = 1;
        scan(eargs, a_gpu, array, local_array, total_sum_gpu, array_size);

        uint32_t outer = (array_size + block_size - 1) / block_size;

        cl::Buffer *a_gpu_ptr = &a_gpu;
        cl::Buffer *b_gpu_ptr = &b_gpu;

        unsigned int *a_size_ptr = &a_size;
        unsigned int *b_size_ptr = &b_size;

        while (outer > 1) {
            leaf_size *= block_size;
            cl::EnqueueArgs eargs_in_recursion(controls.queue,
                                               cl::NDRange((outer + work_group_size - 1) / work_group_size *
                                                           work_group_size),
                                               cl::NDRange(work_group_size));
            scan(eargs_in_recursion, *b_gpu_ptr, *a_gpu_ptr, local_array, total_sum_gpu, outer);
            update(eargs, array, *a_gpu_ptr, array_size, leaf_size);
            outer = (outer + block_size - 1) / block_size;
            std::swap(a_gpu_ptr, b_gpu_ptr);
            std::swap(a_size_ptr, b_size_ptr);
        }
        controls.queue.enqueueReadBuffer(total_sum_gpu, CL_TRUE, 0, sizeof(uint32_t), &total_sum);


    } catch (const cl::Error &e) {
        utils::program_handler(e, program, controls.device, "prefix_sum");
    }
}