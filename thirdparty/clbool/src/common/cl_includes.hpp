#pragma once
#include <libutils/logger.h>
#include <libutils/timer.h>

//#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110

//#define FPGA
#define DEBUG_ENABLE 0


#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__))
#define WIN
#endif

#if 1 //DEBUG_ENABLE
    #ifdef WIN
    inline const Logger logger;/*("../log/log_GPU_merge_path.txt")*/;
    #else
    inline const Logger logger;/*("../log/log_FPGA_merge_path.txt")*/;
    #endif
#endif



#include "CL/cl2.hpp"
#include "CL/opencl.hpp"