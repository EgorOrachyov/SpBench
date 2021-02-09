//
// Created by Egor.Orachev on 09.02.2021.
//

#ifndef SPBENCH_MATRIX_WRITER_HPP
#define SPBENCH_MATRIX_WRITER_HPP

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <unordered_set>
#include <matrix.hpp>

namespace benchmark {

    class MatrixWriter {
    public:

        /**
         * Write matrix data into Matrix Market file format.
         * @param path Path to the file
         */
        void save(const std::string& path, const Matrix& m) {
            assert(!loaded);

            std::ofstream file;
            file.open(path, std::ios_base::out);

            if (!file.is_open()) {
                error = "Failed to open file";
                return;
            }

            file << m.nrows << " " << m.ncols << " " << m.nvals << std::endl;

            for (auto i = 0; i < m.nvals; i++) {
                file << m.rows[i] + 1 << " " << m.cols[i] + 1 << std::endl;
            }

            file.close();
        }

    public:

        std::string error;
    };

}

#endif //SPBENCH_MATRIX_WRITER_HPP
