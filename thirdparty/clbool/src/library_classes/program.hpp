#pragma once

#include "../common/utils.hpp"

template <typename ... Args>
class program {
public:
    using kernel_type = cl::KernelFunctor<Args...>;

private:
    std::string _file_name = "";
    std::string _kernel_name = "";
    uint32_t _block_size = 0;
    uint32_t _needed_work_size = 0;
    cl::Program cl_program;
    bool built = false;

    std::string options_str;

    void check_completeness() {
        if (_file_name == "") throw std::runtime_error("empty file_name");
        if (_kernel_name == "") throw std::runtime_error("empty kernel_name");
        if (_needed_work_size == 0) throw std::runtime_error("zero global_work_size");
    }

public:
    program() = default;
    explicit program(std::string file_name)
    : _file_name(std::move(file_name))
    {}

    program& set_file_name(std::string file_name) {
        _file_name = std::move(file_name);
        return *this;
    }

    program& set_kernel_name(std::string kernel_name) {
        _kernel_name = std::move(kernel_name);
        return *this;
    }

    program& set_block_size(uint32_t block_size) {
        _block_size = block_size;
        return *this;
    }

    program& set_needed_work_size(uint32_t needed_work_size) {
        _needed_work_size = needed_work_size;
        return *this;
    }

    program& add_option(std::string name, std::string value = "") {
        options_str += (" -D " + name + "=" + value);
        return *this;
    }

    template<typename OptionType>
    program& add_option(std::string name, const OptionType &value) {
        options_str += (" -D " + name + "=" + std::to_string(value));
        return *this;
    }

    cl::Event run(Controls &controls, Args ... args) {
        check_completeness();
        if (_block_size == 0) _block_size = controls.block_size;
        try {
            cl_program = controls.create_program_from_file(_file_name);
            std::stringstream options;
            options <<  options_str << " -D RUN " << " -D GROUP_SIZE=" << _block_size;
            cl_program.build(options.str().c_str());

            cl::Kernel kernel(cl_program, _kernel_name.c_str());

            kernel_type functor(kernel);

            cl::EnqueueArgs eargs(controls.queue, cl::NDRange(utils::calculate_global_size(_block_size, _needed_work_size)),
                                  cl::NDRange(_block_size));

            return functor(eargs, args...);
        } catch (const cl::Error &e) {
            utils::program_handler(e, cl_program, controls.device, _kernel_name);
        }
    }
};