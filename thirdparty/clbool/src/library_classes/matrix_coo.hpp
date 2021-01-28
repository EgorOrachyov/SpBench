#pragma once


#include "matrix_base.hpp"
#include "controls.hpp"
#include "../coo/coo_initialization.hpp"
#include "../common/utils.hpp"
#include <vector>

class matrix_coo : public details::matrix_base<COO> {
private:
    // buffers for uint32only;
    cl::Buffer _rows_indices_gpu;
    cl::Buffer _cols_indices_gpu;

public:

    // -------------------------------------- constructors -----------------------------

    matrix_coo() = default;

    matrix_coo(Controls &controls,
               index_type nRows,
               index_type nCols,
               index_type nEntities);

    matrix_coo(index_type nRows,
               index_type nCols,
               index_type nEntities,
               cl::Buffer &rows_indices_gpu,
               cl::Buffer &cols_indices_gpu
               );

    matrix_coo(Controls &controls,
               index_type nRows,
               index_type nCols,
               index_type nEntities,
               std::vector<index_type> &rows_indices,
               std::vector<index_type> &cols_indices,
               bool sorted = false);

    /* we assume, that all input data are sorted */
    matrix_coo(Controls &controls,
               index_type nRows,
               index_type nCols,
               index_type nEntities,
               cl::Buffer &rows,
               cl::Buffer &cols,
               bool sorted = false);

    matrix_coo(matrix_coo const &other) = default;

    matrix_coo(matrix_coo &&other) noexcept = default;

    matrix_coo &operator=(matrix_coo other);

    const auto &rows_indices_gpu() const {
        return _rows_indices_gpu;
    }

    const auto &cols_indices_gpu() const {
        return _cols_indices_gpu;
    }

    const auto &rows_indices_gpu() {
        return _rows_indices_gpu;
    }

    const auto &cols_indices_gpu() {
        return _cols_indices_gpu;
    }

};

