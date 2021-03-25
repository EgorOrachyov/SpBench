#pragma once

#include "matrix_base.hpp"
#include "controls.hpp"

#include <vector>

/* very raw sketch to hold arrays */

class matrix_dcsr : public details::matrix_base<DCSR> {

private:
    // buffers for uint32only;
    cl::Buffer _rows_pointers_gpu;
    cl::Buffer _rows_compressed_gpu;
    cl::Buffer _cols_indices_gpu;

    uint32_t _nzr;

public:

    // -------------------------------------- constructors -----------------------------

    matrix_dcsr() = default;


    matrix_dcsr(cl::Buffer rows_pointers_gpu,
                cl::Buffer rows_compressed_gpu,
                cl::Buffer cols_indices_gpu,

                uint32_t n_rows,
                uint32_t n_cols,
                uint32_t nnz,
                uint32_t nzr
                )
    : details::matrix_base<DCSR>(n_rows, n_cols, nnz)
    , _rows_pointers_gpu(std::move(rows_pointers_gpu))
    ,  _rows_compressed_gpu(std::move(rows_compressed_gpu))
    ,  _cols_indices_gpu(std::move(cols_indices_gpu))
    , _nzr(nzr)
    {};


    const auto &rows_pointers_gpu() const {
        return _rows_pointers_gpu;
    }

    const auto &rows_compressed_gpu() const {
        return _rows_compressed_gpu;
    }

    const auto &cols_indices_gpu() const {
        return _cols_indices_gpu;
    }

    const uint32_t &nzr() const {
        return _nzr;
    }

    auto &rows_pointers_gpu()  {
        return _rows_pointers_gpu;
    }

    auto &rows_compressed_gpu()  {
        return _rows_compressed_gpu;
    }

    auto &cols_indices_gpu() {
        return _cols_indices_gpu;
    }

    uint32_t &nzr()  {
        return _nzr;
    }

};

