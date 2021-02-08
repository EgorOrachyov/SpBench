#pragma once
#include "../library_classes/controls.hpp"
#include "../library_classes/matrix_coo.hpp"
#include "../library_classes/matrix_dcsr.hpp"
#include "../library_classes/cpu_matrices.hpp"


namespace utils {
    void compare_matrices(Controls &controls, matrix_dcsr m_gpu, matrix_dcsr_cpu m_cpu);

    using cpu_buffer = std::vector<uint32_t>;

    void fill_random_buffer(cpu_buffer &buf);

// https://stackoverflow.com/a/466242
    unsigned int ceil_to_power2(uint32_t v);

// https://stackoverflow.com/a/2681094
    uint32_t round_to_power2(uint32_t x);

    uint32_t calculate_global_size(uint32_t work_group_size, uint32_t n);

    Controls create_controls();

    std::string error_name(cl_int error);

    void print_gpu_buffer(Controls &controls, const cl::Buffer &buffer, uint32_t size);

    void print_cpu_buffer(const cpu_buffer &buffer, uint32_t size = -1);

    void compare_buffers(Controls &controls, const cl::Buffer &buffer_g, const cpu_buffer& buffer_c, uint32_t size, std::string name = "");

    void program_handler(const cl::Error &e, const cl::Program &program,
                         const cl::Device &device, const std::string& name);
//    matrix_dcsr matrix_dcsr_from_cpu(Controls &controls, const coo_utils::matrix_dcsr_cpu &m, uint32_t size);
}