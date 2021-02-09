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
#include <unordered_set>
#include <matrix.hpp>

namespace benchmark {

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

            struct Hash {
                size_t operator()(const pair& p) const {
                    return std::hash<unsigned int>()(p.first) + std::hash<unsigned int>()(p.second);
                }
            };

            struct Eq {
                size_t operator()(const pair& a, const pair& b) const {
                    return a.first == b.first && a.second == b.second;
                }
            };

            std::unordered_set<pair, Hash, Eq> pairsSet;

            nvals = isUndirected? nvalsInFile * 2: nvalsInFile;
            pairsSet.reserve(nvals);

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

                assert(rowid < nrows);
                assert(colid < ncols);

                if (pairsSet.find({rowid, colid}) != pairsSet.end()) {
                    std::cerr << rowid << " " << colid << std::endl;
                }

                pairsSet.emplace(rowid, colid);

                if (isUndirected) {
                    pairsSet.emplace(colid, rowid);
                }
            }

            nvals = pairsSet.size();

            pairs.reserve(nvals);
            for (auto& p: pairsSet) {
                pairs.push_back(p);
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

                assert(p.first < nrows);
                assert(p.second < ncols);
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

    class MatrixLoader2 {
    public:

        explicit MatrixLoader2(const std::string& path)
        : path(path + "2"), loader(this->path, false) {

        }

        /** Attempts to load data */
        void loadData() {
            loader.loadData();
        }

        bool isLoaded() const {
            return loader.isLoaded();
        }

        /** @return Converted read data to basic coo matrix */
        Matrix getMatrix() const {
            return loader.getMatrix();
        }

    private:
        std::string path;
        MatrixLoader loader;
    };

}

#endif //SPBENCH_MATRIX_LOADER_HPP
