//
// Created by Egor.Orachev on 09.02.2021.
//

#ifndef SPBENCH_MATRIX_HPP
#define SPBENCH_MATRIX_HPP

#include <vector>

struct Matrix {
    size_t nrows = 0;
    size_t ncols = 0;
    size_t nvals = 0;
    std::vector<unsigned int> rows;
    std::vector<unsigned int> cols;
};

#endif //SPBENCH_MATRIX_HPP
