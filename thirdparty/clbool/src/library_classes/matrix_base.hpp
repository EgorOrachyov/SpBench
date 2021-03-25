#pragma once

#include <cstdint>

// format could be some global variable,
enum Format {
    COO,
    CSR,
    DCSR
};


namespace details {

    template<Format format>
    class matrix_base {
    public:
        using index_type = uint32_t;
    protected:

        Format sparse_format = format;

        index_type n_rows;
        index_type n_cols;
        index_type _nnz;

    public:

        matrix_base()
        : n_rows(0), n_cols(0), _nnz(0)
        {}

        matrix_base(index_type n_rows, index_type n_cols, index_type n_entities)
        : n_rows(n_rows), n_cols(n_cols), _nnz(n_entities)
        {}

        Format get_sparse_format() const {
            return sparse_format;
        };

        index_type nRows() const {
            return n_rows;
        };

        index_type nCols() const {
            return n_cols;
        };

        index_type nnz() const {
            return _nnz;
        };

    };
}