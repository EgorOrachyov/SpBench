////////////////////////////////////////////////////////////////////////////////////
// MIT License                                                                    //
//                                                                                //
// Copyright (c) 2021 Egor Orachyov                                               //
//                                                                                //
// Permission is hereby granted, free of charge, to any person obtaining a copy   //
// of this software and associated documentation files (the "Software"), to deal  //
// in the Software without restriction, including without limitation the rights   //
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      //
// copies of the Software, and to permit persons to whom the Software is          //
// furnished to do so, subject to the following conditions:                       //
//                                                                                //
// The above copyright notice and this permission notice shall be included in all //
// copies or substantial portions of the Software.                                //
//                                                                                //
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     //
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       //
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    //
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         //
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  //
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  //
// SOFTWARE.                                                                      //
////////////////////////////////////////////////////////////////////////////////////

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
            std::ofstream file;
            file.open(path, std::ios_base::out);

            if (!file.is_open()) {
                error = "Failed to open file";
                return;
            }

            using pair = std::pair<unsigned int, unsigned int>;

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

            for (auto i = 0; i < m.nvals; i++) {
                pairsSet.emplace(m.rows[i], m.cols[i]);
            }

            file << m.nrows << " " << m.ncols << " " << pairsSet.size() << std::endl;

            for (auto& e: pairsSet) {
                file << e.first + 1 << " " << e.second + 1 << std::endl;
            }

            file.close();
        }

    public:

        std::string error;
    };

}

#endif //SPBENCH_MATRIX_WRITER_HPP
