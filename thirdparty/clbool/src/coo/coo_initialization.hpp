#pragma once

#include "../common/cl_includes.hpp"
#include "../library_classes/controls.hpp"

void check_pref_correctness(const std::vector<uint32_t>& result,
                            const std::vector<uint32_t>& before);

void sort_arrays(Controls& controls, cl::Buffer &rows_gpu, cl::Buffer &cols_gpu, uint32_t n);

void fill_random_matrix(std::vector<uint32_t>& rows, std::vector<uint32_t>& cols);



