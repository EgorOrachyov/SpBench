//
// Created by Egor.Orachev on 26.01.2021.
//

#ifndef SPBENCH_MATRIX_LOADER_HPP
#define SPBENCH_MATRIX_LOADER_HPP

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cassert>


namespace benchmark {

    struct Matrix {
        size_t nrows = 0;
        size_t ncols = 0;
        size_t nvals = 0;
        std::vector<unsigned int> rows;
        std::vector<unsigned int> cols;
    };

    class MatrixLoader {
    public:

        /**
         * Load matrix data from Matrix Market file format.
         * @param path Path to the file
         * @param isUndirected True if graph in the matrix is undirected, and edges must be duplicated
         */
        explicit MatrixLoader(std::string path, bool isUndirected = false)
                : path(std::move(path)), isUndirected(isUndirected) {

        }

        /** Attempts to load data */
        void loadData() {
            assert(!loaded);

            std::ifstream file;
            file.open(path, std::ios_base::in);

            if (!file.is_open()) {
                error = "Failed to open file";
                return;
            }

            std::string line;

            do {
                std::getline(file, line);
            } while (line[0] == '%');

            std::stringstream lineStream(line);

            lineStream >> nrows;
            lineStream >> ncols;
            lineStream >> nvalsInFile;


            nvals = isUndirected? nvalsInFile * 2: nvalsInFile;
            pairs.reserve(nvals);

            for (auto i = 0; i < nvalsInFile; i++) {
                std::getline(file, line);

                unsigned int rowid, colid;

                lineStream = std::stringstream(line);
                lineStream >> rowid;
                lineStream >> colid;

                assert(rowid > 0);
                assert(colid > 0);

                rowid -= 1;
                colid -= 1;

                pairs.emplace_back(rowid, colid);

                if (isUndirected) {
                    pairs.emplace_back(colid, rowid);
                }
            }

            std::sort(pairs.begin(), pairs.end(), [](const pair& a, const pair& b) {
                return a.first < b.first || (a.first == b.first && a.second < b.second);
            });

            loaded = true;
        }

        bool isLoaded() const {
            return loaded;
        }

        /** @return Converted read data to basic coo matrix */
        Matrix getMatrix() const {
            Matrix matrix;
            matrix.nrows = nrows;
            matrix.ncols = ncols;
            matrix.nvals = nvals;
            matrix.rows.reserve(nvals);
            matrix.cols.reserve(nvals);

            for (const auto& p: pairs) {
                matrix.rows.push_back(p.first);
                matrix.cols.push_back(p.second);
            }

            return std::move(matrix);
        }

    private:

        using pair = std::pair<unsigned int, unsigned int>;

        bool loaded = false;
        bool isUndirected;
        std::string path;
        std::string error;
        size_t nrows = 0;
        size_t ncols = 0;
        size_t nvals = 0;
        size_t nvalsInFile = 0;
        std::vector<pair> pairs;
    };

}

#endif //SPBENCH_MATRIX_LOADER_HPP
