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

#include <benchmark_base.hpp>
#include <matrix_loader.hpp>

#include <cubool/cubool.h>

#define BENCH_DEBUG

namespace benchmark {

    class TransitiveClosure: public BenchmarkBase {
    public:

        TransitiveClosure() {
            benchmarkName = "Transitive Closure";
            experimentsCount = 3;
        }

    protected:

        static void messageCallback(CuBoolStatus status, const char* message, void* userData) {
            std::cout << "= Cubool: Status: " << status << " Message: \"" << message << "\"" << std::endl;
        }

        void setupBenchmark() override {
            CuBoolInstanceDesc desc{};
            desc.errorCallback = {nullptr, messageCallback };
            desc.memoryType = CUBOOL_GPU_MEMORY_TYPE_GENERIC;

            status = CuBool_Instance_New(&desc, &instance);
            assert(status == CUBOOL_STATUS_SUCCESS);
        }

        void tearDownBenchmark() override {
            status = CuBool_Instance_Free(instance);
            assert(status == CUBOOL_STATUS_SUCCESS);
        }

        void setupExperiment(size_t experimentIdx, size_t &iterationsCount, std::string& name) override {
            iterationsCount = ITERATIONS_COUNT[experimentIdx];
            name = FILES_NAMES[experimentIdx];

            const auto& file = FILES_NAMES[experimentIdx];
            const auto& type = IS_UNDIRECTED[experimentIdx];

            MatrixLoader loader(file, type);
            loader.loadData();
            input = std::move(loader.getMatrix());

#ifdef BENCH_DEBUG
            log       << ">   Load matrix: \"" << file << "\" isUndirected: " << type << std::endl
                      << "                 size: " << input.nrows << " x " << input.ncols << " nvals: " << input.nvals << std::endl;
#endif // BENCH_DEBUG

            CuBoolIndex_t n = input.nrows;
            assert(input.nrows == input.ncols);

            status = CuBool_Matrix_New(instance, &matrix, n, n);
            assert(status == CUBOOL_STATUS_SUCCESS);

            status = CuBool_Matrix_Build(instance, matrix, input.rows.data(), input.cols.data(), input.nvals);
            assert(status == CUBOOL_STATUS_SUCCESS);
        }

        void tearDownExperiment(size_t experimentIdx) override {
            input = Matrix{};

            status = CuBool_Matrix_Free(instance, matrix);
            assert(status == CUBOOL_STATUS_SUCCESS);
        }

        void setupIteration(size_t experimentIdx, size_t iterationIdx) override {

        }

        void execIteration(size_t experimentIdx, size_t iterationIdx) override {
            status = CuBool_Matrix_Duplicate(instance, matrix, &matrixTC);         /** Create result matrix and copy initial values */
            assert(status == CUBOOL_STATUS_SUCCESS);

            CuBoolSize_t total = 0;
            CuBoolSize_t current;

            status = CuBool_Matrix_Nvals(instance, matrixTC, &current);            /** Query current number on non-zero elements */
            assert(status == CUBOOL_STATUS_SUCCESS);

            while (current != total) {                                             /** Loop while values are added */
                total = current;

                status = CuBool_MxM(instance, matrixTC, matrixTC, matrixTC);       /** M_tc += M_tc * M_tc */
                assert(status == CUBOOL_STATUS_SUCCESS);

                status = CuBool_Matrix_Nvals(instance, matrixTC, &current);
                assert(status == CUBOOL_STATUS_SUCCESS);

#ifdef BENCH_DEBUG
                std::cout << "    Nvals= " << current << std::endl;
#endif
            }
        }

        void tearDownIteration(size_t experimentIdx, size_t iterationIdx) override {
            if (matrixTC) {
                CuBoolSize_t nvals;
                CuBoolIndex_t nrows;
                CuBoolIndex_t ncols;

                status = CuBool_Matrix_Nrows(instance, matrixTC, &nrows);
                assert(status == CUBOOL_STATUS_SUCCESS);

                status = CuBool_Matrix_Ncols(instance, matrixTC, &ncols);
                assert(status == CUBOOL_STATUS_SUCCESS);

                status = CuBool_Matrix_Nvals(instance, matrixTC, &nvals);
                assert(status == CUBOOL_STATUS_SUCCESS);

#ifdef BENCH_DEBUG
                std::cout << "   Result matrix_tc: size: " << nrows << " x " << ncols << " nvals: " << nvals << std::endl;
#endif // BENCH_DEBUG

                status = CuBool_Matrix_Free(instance, matrixTC);
                assert(status == CUBOOL_STATUS_SUCCESS);

                matrixTC = nullptr;
            }
        }


    protected:

        CuBoolInstance instance = nullptr;
        CuBoolMatrix matrix = nullptr;
        CuBoolMatrix matrixTC = nullptr;
        CuBoolStatus status = CUBOOL_STATUS_SUCCESS;

        Matrix input;

        static const size_t TOTAL_EXPERIMENTS = 3;

        const std::string FILES_NAMES[TOTAL_EXPERIMENTS] = {
                "wing.mtx",
                "coAuthorsCiteseer.mtx",
                "roadNet-CA.mtx"
        };

        const bool IS_UNDIRECTED[TOTAL_EXPERIMENTS] = {
                true,
                true,
                true
        };

        const size_t ITERATIONS_COUNT[TOTAL_EXPERIMENTS] = {
                1,
                1,
                1
        };

    };

}

int main() {
    benchmark::TransitiveClosure tc;
    tc.runBenchmark();
    return 0;
}