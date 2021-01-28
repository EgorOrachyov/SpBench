#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110

#define CL_ENABLE_EXCEPTIONS

#include <CL//cl.hpp>
#define KernelFunctor make_kernel
//#include "CL/opencl.hpp"