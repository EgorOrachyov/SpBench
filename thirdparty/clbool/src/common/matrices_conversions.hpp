#pragma once

#include "../library_classes/controls.hpp"
#include "../library_classes/matrix_coo.hpp"
#include "../library_classes/matrix_dcsr.hpp"
#include "../library_classes/program.hpp"
#include "cl_operations.hpp"
#include "utils.hpp"


matrix_coo dcsr_to_coo(Controls &controls, matrix_dcsr &a);
matrix_dcsr coo_to_dcsr_gpu(Controls &controls, const matrix_coo &a);
matrix_dcsr matrix_dcsr_from_cpu(Controls &controls, matrix_dcsr_cpu &m, uint32_t size);
matrix_dcsr_cpu matrix_dcsr_from_gpu(Controls &controls, matrix_dcsr &m);
matrix_coo_cpu matrix_coo_from_gpu(Controls &controls, matrix_coo &m);