#include <algorithm>
#include "coo_tests.hpp"

#include "../coo/coo_utils.hpp"
#include "../dcsr/dscr_matrix_multiplication.hpp"

using namespace coo_utils;
using namespace utils;
const uint32_t BINS_NUM = 38;


auto get_merge_kernel(Controls &controls) {
    cl::Program program;
    try {

        program = controls.create_program_from_file("../src/coo/cl/for_test/new_merge.cl");
        uint32_t block_size = controls.block_size;
        std::stringstream options;
        options << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = block_size;

        cl::Kernel new_merge_kernel(program, "new_merge");

        using KernelType = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t, uint32_t>;

        KernelType new_merge(new_merge_kernel);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        return std::pair<KernelType, cl::EnqueueArgs>(new_merge, eargs);
    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(controls.device);
        }
        throw std::runtime_error(exception.str());
    }
}




void test_new_merge() {
    Controls controls = create_controls();
    for (int test_size = 256; test_size < 30000; test_size += 500) {
        cpu_buffer a_cpu(test_size);
        cpu_buffer b_cpu(controls.block_size);
        cpu_buffer c_cpu;

        fill_random_buffer(a_cpu);
        fill_random_buffer(b_cpu);

        std::sort(a_cpu.begin(), a_cpu.end());
        std::sort(b_cpu.begin(), b_cpu.end());

        std::merge(a_cpu.begin(), a_cpu.end(), b_cpu.begin(), b_cpu.end(),
                   std::back_inserter(c_cpu));

        cl::Buffer a_gpu = cl::Buffer(controls.queue, a_cpu.begin(), a_cpu.end(), false);
        cl::Buffer b_gpu = cl::Buffer(controls.queue, b_cpu.begin(), b_cpu.end(), false);
        cl::Buffer c_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * c_cpu.size());

        auto merge = get_merge_kernel(controls);
        merge.first(merge.second,
                    a_gpu, b_gpu, c_gpu, a_cpu.size(), b_cpu.size());

        compare_buffers(controls, c_gpu, c_cpu, c_cpu.size());
    }
}

auto get_merge_full_kernel(Controls &controls, uint32_t size) {
    cl::Program program;
    try {
        program = controls.create_program_from_file("../src/coo/cl/for_test/new_merge.cl");
        uint32_t block_size = controls.block_size;
        std::stringstream options;
        options << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = calculate_global_size(block_size, size);

        cl::Kernel new_merge_full_kernel(program, "new_merge_full");

        using KernelType = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t, uint32_t>;

        KernelType new_merge_full(new_merge_full_kernel);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        return std::pair<KernelType, cl::EnqueueArgs>(new_merge_full, eargs);
    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(controls.device);
        }
        throw std::runtime_error(exception.str());
    }
}

void test_new_merge_full() {
    Controls controls = create_controls();
    for (int test_size_a = 256; test_size_a < 30000; test_size_a += 500) {
        for (int test_size_b = 256; test_size_b < 30000; test_size_b += 500) {
//            uint32_t test_size_a = 256;
//            uint32_t test_size_b = 256 + 500;


            cpu_buffer a_cpu(test_size_a);
            cpu_buffer b_cpu(test_size_b);
            cpu_buffer c_cpu;

            fill_random_buffer(a_cpu);
            fill_random_buffer(b_cpu);

            std::sort(a_cpu.begin(), a_cpu.end());
            std::sort(b_cpu.begin(), b_cpu.end());

            std::merge(a_cpu.begin(), a_cpu.end(), b_cpu.begin(), b_cpu.end(),
                       std::back_inserter(c_cpu));

            cl::Buffer a_gpu = cl::Buffer(controls.queue, a_cpu.begin(), a_cpu.end(), false);
            cl::Buffer b_gpu = cl::Buffer(controls.queue, b_cpu.begin(), b_cpu.end(), false);
            cl::Buffer c_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * c_cpu.size());
//            print_cpu_buffer(c_cpu);

            auto merge = get_merge_full_kernel(controls, a_cpu.size() + b_cpu.size());
            merge.first(merge.second,
                        a_gpu, b_gpu, c_gpu, a_cpu.size(), b_cpu.size());

//            std::cout << "~~~~~~~~~~~~~~\n~~~~~~~~~~~~~~\n";
//            print_gpu_buffer(controls, c_gpu, c_cpu.size());

//            std::cout << c_cpu[349] << std::endl;
            compare_buffers(controls, c_gpu, c_cpu, c_cpu.size());
        }
    }
}