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
#include <args_processor.hpp>

#include <cubool/cubool.h>

#define BENCH_DEBUG
#define CUBOOL_CHECK(func) do { auto s = func; assert(s == CUBOOL_STATUS_SUCCESS); } while(0)

namespace benchmark {

    class Add: public BenchmarkBase {
    public:

        Add(int argc, const char** argv) {
            argsProcessor.parse(argc, argv);
            assert(argsProcessor.isParsed());

            benchmarkName = "Cubool-Add";
            experimentsCount = argsProcessor.getExperimentsCount();
        }

    protected:

        void setupBenchmark() override {
            CUBOOL_CHECK(cuBool_Initialize(CUBOOL_HINT_NO));
        }

        void tearDownBenchmark() override {
            CUBOOL_CHECK(cuBool_Finalize());
        }

        void setupExperiment(size_t experimentIdx, size_t &iterationsCount, std::string& name) override {
            auto& entry = argsProcessor.getEntries()[experimentIdx];

            iterationsCount = entry.iterations;
            name = entry.name;

            const auto& file = entry.name;
            const auto& type = entry.isUndirected;

            MatrixLoader loader(file, type);
            loader.loadData();
            input = std::move(loader.getMatrix());

#ifdef BENCH_DEBUG
            log       << ">   Load matrix: \"" << file << "\" isUndirected: " << type << std::endl
                      << "                 size: " << input.nrows << " x " << input.ncols << " nvals: " << input.nvals << std::endl;
#endif // BENCH_DEBUG

            cuBool_Index n = input.nrows;
            assert(input.nrows == input.ncols);

            CUBOOL_CHECK(cuBool_Matrix_New(&A, n, n));
            CUBOOL_CHECK(cuBool_Matrix_Build(A, input.rows.data(), input.cols.data(), input.nvals, CUBOOL_HINT_NO));

            MatrixLoader2 loader2(file);
            loader2.loadData();
            input = std::move(loader2.getMatrix());

            CUBOOL_CHECK(cuBool_Matrix_New(&A2, n, n));
            CUBOOL_CHECK(cuBool_Matrix_Build(A2, input.rows.data(), input.cols.data(), input.nvals, CUBOOL_HINT_NO));
        }

        void tearDownExperiment(size_t experimentIdx) override {
            input = Matrix{};

            CUBOOL_CHECK(cuBool_Matrix_Free(A));
            CUBOOL_CHECK(cuBool_Matrix_Free(A2));

            A = nullptr;
            A2 = nullptr;
        }

        void setupIteration(size_t experimentIdx, size_t iterationIdx) override {
            cuBool_Index n;
            CUBOOL_CHECK(cuBool_Matrix_Nrows(A, &n));
            CUBOOL_CHECK(cuBool_Matrix_New(&R, n, n));
        }


        void execIteration(size_t experimentIdx, size_t iterationIdx) override {
            CUBOOL_CHECK(cuBool_Matrix_EWiseAdd(R, A, A2, CUBOOL_HINT_NO));
        }

        void tearDownIteration(size_t experimentIdx, size_t iterationIdx) override {
            if (A) {
                cuBool_Index nvals;
                cuBool_Index nrows;
                cuBool_Index ncols;

                CUBOOL_CHECK(cuBool_Matrix_Nrows(R, &nrows));
                CUBOOL_CHECK(cuBool_Matrix_Ncols(R, &ncols));
                CUBOOL_CHECK(cuBool_Matrix_Nvals(R, &nvals));

#ifdef BENCH_DEBUG
                log << "   Result matrix: size: " << nrows << " x " << ncols << " nvals: " << nvals << std::endl;
#endif // BENCH_DEBUG

                CUBOOL_CHECK(cuBool_Matrix_Free(R));

                R = nullptr;
            }
        }


    protected:

        ArgsProcessor argsProcessor;

        cuBool_Matrix A = nullptr;
        cuBool_Matrix A2 = nullptr;
        cuBool_Matrix R = nullptr;

        Matrix input;

    };

}

int main(int argc, const char** argv) {
    benchmark::Add add(argc, argv);
    add.runBenchmark();
    return 0;
}