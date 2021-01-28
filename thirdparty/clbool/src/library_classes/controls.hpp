#pragma once

#include "../common/cl_includes.hpp"
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

// TODO: in opencl 2.0 we have DeviceCommandQueue class so what about opencl 2.0?

struct Controls {
    const cl::Device device;
    const cl::Context context;
    cl::CommandQueue queue;
    const uint32_t block_size = uint32_t(256);

    Controls(cl::Device device) :
    device(device)
    , context(cl::Context(device))
    , queue(cl::CommandQueue(context))
    {}

    // TODO: do we really need methods?
    cl::Program create_program_from_file(std::string filename) const {
        // TODO: ADD FILE NAME IN BINARY WITH CMAKE TASK
        std::ifstream cl_file(filename);
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, {cl_string.c_str(), cl_string.length()});
        return cl::Program(context, source);
    }

};


