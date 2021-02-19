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

#include <CL/cl.hpp>

// clBool goes here
#include <library_classes/controls.hpp>
#include <library_classes/cpu_matrices.hpp>
#include <coo/coo_utils.hpp>
#include <common/utils.hpp>
#include <common/matrices_conversions.hpp>
#include <dcsr/dscr_matrix_multiplication.hpp>
#include <coo/coo_matrix_addition.hpp>

#define BENCH_DEBUG

namespace benchmark {
    class Add: public BenchmarkBase {
    public:

        Add(int argc, const char** argv) {
            argsProcessor.parse(argc, argv);
            assert(argsProcessor.isParsed());

            benchmarkName = "Clbool-Add";
            experimentsCount = argsProcessor.getExperimentsCount();
        }

    protected:

        void setupBenchmark() override {
            controls = new Controls(utils::create_controls());
        }

        void tearDownBenchmark() override {
            delete controls;
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
            log       << ">   Load A: \"" << file << "\" isUndirected: " << type << std::endl
                      << "                 size: " << input.nrows << " x " << input.ncols << " nvals: " << input.nvals << std::endl;
#endif // BENCH_DEBUG

            {
                size_t n = input.nrows;
                assert(input.nrows == input.ncols);

                A = std::move(matrix_coo(*controls, n, n, input.nvals, input.rows, input.cols, true));
            }

            MatrixLoader2 loader2(file);
            loader2.loadData();
            input = std::move(loader2.getMatrix());

#ifdef BENCH_DEBUG
            log       << ">   Load A2: \"" << file << "\" isUndirected: " << type << std::endl
                      << "                 size: " << input.nrows << " x " << input.ncols << " nvals: " << input.nvals << std::endl;
#endif // BENCH_DEBUG

            {
                size_t n = input.nrows;
                assert(input.nrows == input.ncols);

                A2 = std::move(matrix_coo(*controls, n, n, input.nvals, input.rows, input.cols, true));
            }
        }

        void tearDownExperiment(size_t experimentIdx) override {
            input = Matrix{};
            A = matrix_coo{};
            A2 = matrix_coo{};
        }

        void setupIteration(size_t experimentIdx, size_t iterationIdx) override {

        }

        void execIteration(size_t experimentIdx, size_t iterationIdx) override {
            matrix_addition(*controls, R, A, A2);
        }

        void tearDownIteration(size_t experimentIdx, size_t iterationIdx) override {
#ifdef BENCH_DEBUG
            log << "   Result matrix: size " << R.nRows() << " x " << R.nCols()
                << " nvals " << R.nnz() << std::endl;
#endif

            R = matrix_coo{};
        }

    protected:

        Controls* controls;
        matrix_coo A;
        matrix_coo A2;
        matrix_coo R;

        ArgsProcessor argsProcessor;
        Matrix input;
    };

}

int main(int argc, const char** argv) {
    benchmark::Add add(argc, argv);
    add.runBenchmark();
    return 0;
}