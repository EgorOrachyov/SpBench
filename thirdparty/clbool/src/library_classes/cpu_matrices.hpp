#pragma once

using coordinates = std::pair<uint32_t, uint32_t>;
using matrix_coo_cpu_pairs = std::vector<coordinates>;
using cpu_buffer = std::vector<uint32_t>;

class matrix_dcsr_cpu {
    cpu_buffer _rows_pointers;
    cpu_buffer _rows_compressed;
    cpu_buffer _cols_indices;

public:
    matrix_dcsr_cpu(cpu_buffer rows_pointers, cpu_buffer rows_compressed, cpu_buffer cols_indices)
            : _rows_pointers(std::move(rows_pointers)), _rows_compressed(std::move(rows_compressed)),
              _cols_indices(std::move(cols_indices)) {}

    matrix_dcsr_cpu() = default;

    matrix_dcsr_cpu &operator=(matrix_dcsr_cpu other) {
        _rows_pointers = std::move(other._rows_pointers);
        _rows_compressed = std::move(other._rows_compressed);
        _cols_indices = std::move(other._cols_indices);
        return *this;
    }

    cpu_buffer &rows_pointers() {
        return _rows_pointers;
    }

    cpu_buffer &rows_compressed() {
        return _rows_compressed;
    }

    cpu_buffer &cols_indices() {
        return _cols_indices;
    }

    const cpu_buffer &rows_pointers() const {
        return _rows_pointers;
    }

    const cpu_buffer &rows_compressed() const {
        return _rows_compressed;
    }

    const cpu_buffer &cols_indices() const {
        return _cols_indices;
    }

};


class matrix_coo_cpu {
    cpu_buffer _rows_indices;
    cpu_buffer _cols_indices;

public:
    matrix_coo_cpu(cpu_buffer rows_indices, cpu_buffer cols_indices)
            : _rows_indices(std::move(rows_indices))
            , _cols_indices(std::move(cols_indices))
            {}

    matrix_coo_cpu() = default;

    matrix_coo_cpu &operator=(matrix_coo_cpu other) {
        _rows_indices = std::move(other._rows_indices);
        _cols_indices = std::move(other._cols_indices);
        return *this;
    }

    cpu_buffer &rows_indices() {
        return _rows_indices;
    }

    cpu_buffer &cols_indices() {
        return _cols_indices;
    }

    const cpu_buffer &rows_indices() const {
        return _rows_indices;
    }

    const cpu_buffer &cols_indices() const {
        return _cols_indices;
    }

};