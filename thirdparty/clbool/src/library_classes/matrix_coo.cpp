#include "matrix_coo.hpp"

matrix_coo::matrix_coo(Controls &controls,
                       index_type nRows,
                       index_type nCols,
                       index_type nEntities)
    : matrix_base(nRows, nCols, nEntities)
    , _rows_indices_gpu(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(index_type) * _nnz))
    , _cols_indices_gpu(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(index_type) * _nnz))
    {}

matrix_coo::matrix_coo(index_type nRows,
                       index_type nCols,
                       index_type nEntities,
                       cl::Buffer &rows_indices_gpu,
                       cl::Buffer &cols_indices_gpu)
    : matrix_base(nRows, nCols, nEntities)
    , _rows_indices_gpu(rows_indices_gpu)
    , _cols_indices_gpu(cols_indices_gpu)
    {}


matrix_coo::matrix_coo(Controls &controls,
                       index_type nRows,
                       index_type nCols,
                       index_type nEntities,
                       std::vector<index_type> &rows_indices,
                       std::vector<index_type> &cols_indices,
                       bool sorted)
    : matrix_base(nRows, nCols, nEntities)
    , _rows_indices_gpu(cl::Buffer(controls.queue, rows_indices.begin(), rows_indices.end(), false))
    , _cols_indices_gpu(cl::Buffer(controls.queue, cols_indices.begin(), cols_indices.end(), false))
 {
    try {

        if (!sorted) {
            sort_arrays(controls, _rows_indices_gpu, _cols_indices_gpu, _nnz);
        }

    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        throw std::runtime_error(exception.str());
    }
}


matrix_coo::matrix_coo(Controls &controls,
                       index_type nRows,
                       index_type nCols,
                       index_type nEntities,
                       cl::Buffer &rows,
                       cl::Buffer &cols,
                       bool sorted)
    : matrix_base(nRows, nCols, nEntities)
    , _rows_indices_gpu(rows)
    , _cols_indices_gpu(cols)
 {
    try {
        if (!sorted) {
            sort_arrays(controls, _rows_indices_gpu, _cols_indices_gpu, _nnz);
        }

    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        throw std::runtime_error(exception.str());
    }
}

matrix_coo &matrix_coo::operator=(matrix_coo other) {
    n_cols = other.n_cols;
    n_rows = other.n_rows;
    _nnz = other._nnz;
    _rows_indices_gpu = other._rows_indices_gpu;
    _cols_indices_gpu = other._cols_indices_gpu;
    return *this;
}

