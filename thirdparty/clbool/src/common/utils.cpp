#include "utils.hpp"
#include "fast_random.h"

namespace utils {
    void compare_buffers(Controls &controls, const cl::Buffer &buffer_g, const cpu_buffer &buffer_c, uint32_t size, std::string name) {
        cpu_buffer cpu_copy(size);
        controls.queue.enqueueReadBuffer(buffer_g, CL_TRUE, 0, sizeof(uint32_t) * cpu_copy.size(), cpu_copy.data());
        for (uint32_t i = 0; i < size; ++i) {
            if (cpu_copy[i] != buffer_c[i]) {
                uint32_t start = std::max(0, (int) i - 10);
                uint32_t stop = std::min(size, i + 10);
                for (uint32_t j = start; j < stop; ++j) {
                    std::cout << j << ": (" << cpu_copy[j] << ", " << buffer_c[j] << "), ";
                }
                std::cout << std::endl;
                throw std::runtime_error("buffers for " + name + " are different");
            }
        }
        std::cout << "buffers are equal" << std::endl;
    }

    void compare_matrices(Controls &controls, matrix_dcsr m_gpu, matrix_dcsr_cpu m_cpu) {
        if (m_gpu.nnz() != m_cpu.cols_indices().size()) {
            std::cout << "diff nnz, gpu: " << m_gpu.nnz() << " vs cpu: " << m_cpu.cols_indices().size() << std::endl;
        }
        compare_buffers(controls, m_gpu.rows_pointers_gpu(), m_cpu.rows_pointers(), m_gpu.nzr() + 1, "rows_pointers");
        compare_buffers(controls, m_gpu.rows_compressed_gpu(), m_cpu.rows_compressed(), m_gpu.nzr(), "rows_compressed");
        compare_buffers(controls, m_gpu.cols_indices_gpu(), m_cpu.cols_indices(), m_gpu.nnz(), "cols_indices");
    }


// https://stackoverflow.com/a/466242
    unsigned int ceil_to_power2(uint32_t v) {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }

// https://stackoverflow.com/a/2681094
    uint32_t round_to_power2(uint32_t x) {
        x = x | (x >> 1);
        x = x | (x >> 2);
        x = x | (x >> 4);
        x = x | (x >> 8);
        x = x | (x >> 16);
        return x - (x >> 1);
    }

    uint32_t calculate_global_size(uint32_t work_group_size, uint32_t n) {
        return (n + work_group_size - 1) / work_group_size * work_group_size;
    }

    Controls create_controls() {
        std::vector<cl::Platform> platforms;
        std::vector<cl::Device> devices;
        std::vector<cl::Kernel> kernels;
        cl::Program program;
        cl::Device device;
        int platformId = -1;

        try {
            cl::Platform::get(&platforms);
            // Here we try to select cuda, even if AMD is available
            {
                std::string keyWords[] = { "cuda", "CUDA", "Cuda", "NVIDIA", "nvidia", "Nvidia"};

                bool foundCuda = false;
                for (const auto& p : platforms) {
                    platformId += 1;

                    std::cout << "Platform ID " << platformId << " : " << p.getInfo<CL_PLATFORM_NAME>() << std::endl;
                    auto info = p.getInfo<CL_PLATFORM_NAME>();

                    for (auto& k: keyWords) {
                        if (info.find(k) != std::basic_string<char, std::char_traits<char>, std::allocator<char>>::npos) {
                            foundCuda = true;
                            std::cout << "Select Platform ID " << platformId << " : " << p.getInfo<CL_PLATFORM_NAME>() << std::endl;
                            break;
                        }
                    }

                    if (foundCuda)
                        break;
                }
            }

            platforms[platformId].getDevices(CL_DEVICE_TYPE_GPU, &devices);
            std::cout << devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
            return Controls(devices[0]);

        } catch (const cl::Error &e) {
            std::stringstream exception;
            exception << "\n" << e.what() << " : " << e.err() << "\n";
            throw std::runtime_error(exception.str());
        }
    }

    std::string error_name(cl_int error) {
        switch (error) {
            case 0:
                return "CL_SUCCESS";
            case -1:
                return "CL_DEVICE_NOT_FOUND";
            case -2:
                return "CL_DEVICE_NOT_AVAILABLE";
            case -3:
                return "CL_COMPILER_NOT_AVAILABLE";
            case -4:
                return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
            case -5:
                return "CL_OUT_OF_RESOURCES";
            case -6:
                return "CL_OUT_OF_HOST_MEMORY";
            case -7:
                return "CL_PROFILING_INFO_NOT_AVAILABLE";
            case -8:
                return "CL_MEM_COPY_OVERLAP";
            case -9:
                return "CL_IMAGE_FORMAT_MISMATCH";
            case -10:
                return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
            case -11:
                return "CL_BUILD_PROGRAM_FAILURE";
            case -12:
                return "CL_MAP_FAILURE";
            case -13:
                return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
            case -14:
                return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
            case -15:
                return "CL_COMPILE_PROGRAM_FAILURE";
            case -16:
                return "CL_LINKER_NOT_AVAILABLE";
            case -17:
                return "CL_LINK_PROGRAM_FAILURE";
            case -18:
                return "CL_DEVICE_PARTITION_FAILED";
            case -19:
                return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
            case -30:
                return "CL_INVALID_VALUE";
            case -31:
                return "CL_INVALID_DEVICE_TYPE";
            case -32:
                return "CL_INVALID_PLATFORM";
            case -33:
                return "CL_INVALID_DEVICE";
            case -34:
                return "CL_INVALID_CONTEXT";
            case -35:
                return "CL_INVALID_QUEUE_PROPERTIES";
            case -36:
                return "CL_INVALID_COMMAND_QUEUE";
            case -37:
                return "CL_INVALID_HOST_PTR";
            case -38:
                return "CL_INVALID_MEM_OBJECT";
            case -39:
                return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
            case -40:
                return "CL_INVALID_IMAGE_SIZE";
            case -41:
                return "CL_INVALID_SAMPLER";
            case -42:
                return "CL_INVALID_BINARY";
            case -43:
                return "CL_INVALID_BUILD_OPTIONS";
            case -44:
                return "CL_INVALID_PROGRAM";
            case -45:
                return "CL_INVALID_PROGRAM_EXECUTABLE";
            case -46:
                return "CL_INVALID_KERNEL_NAME";
            case -47:
                return "CL_INVALID_KERNEL_DEFINITION";
            case -48:
                return "CL_INVALID_KERNEL";
            case -49:
                return "CL_INVALID_ARG_INDEX";
            case -50:
                return "CL_INVALID_ARG_VALUE";
            case -51:
                return "CL_INVALID_ARG_SIZE";
            case -52:
                return "CL_INVALID_KERNEL_ARGS";
            case -53:
                return "CL_INVALID_WORK_DIMENSION";
            case -54:
                return "CL_INVALID_WORK_GROUP_SIZE";
            case -55:
                return "CL_INVALID_WORK_ITEM_SIZE";
            case -56:
                return "CL_INVALID_GLOBAL_OFFSET";
            case -57:
                return "CL_INVALID_EVENT_WAIT_LIST";
            case -58:
                return "CL_INVALID_EVENT";
            case -59:
                return "CL_INVALID_OPERATION";
            case -60:
                return "CL_INVALID_GL_OBJECT";
            case -61:
                return "CL_INVALID_BUFFER_SIZE";
            case -62:
                return "CL_INVALID_MIP_LEVEL";
            case -63:
                return "CL_INVALID_GLOBAL_WORK_SIZE";
            case -64:
                return "CL_INVALID_PROPERTY";
            default:
                return "unknown error code: " + std::to_string(error);
        }
    }

    void print_gpu_buffer(Controls &controls, const cl::Buffer &buffer, uint32_t size) {
        cpu_buffer cpu_copy(size);
        controls.queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(uint32_t) * cpu_copy.size(), cpu_copy.data());
        for (auto const &item: cpu_copy) {
            std::cout << item << ", ";
        }
        std::cout << std::endl;
    }

    void print_cpu_buffer(const cpu_buffer &buffer, uint32_t size) {
        uint32_t end = size;
        if (size == -1) end = buffer.size();
        for (uint32_t i = 0; i < end; ++i) {
            std::cout << buffer[i] << ", ";
        }
        std::cout << std::endl;
    }


    void fill_random_buffer(cpu_buffer &buf) {
        uint32_t n = buf.size();
        FastRandom r(n);
        for (uint32_t i = 0; i < n; ++i) {
            buf[i] = r.next();
        }
    }

    void
    program_handler(const cl::Error &e, const cl::Program &program, const cl::Device &device, const std::string &name) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << " in " << name << " \n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        }
        throw std::runtime_error(exception.str());
    }
}