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

#include <cusp_compiler_fence.hpp>
#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/multiply.h>
#include <cusp/elementwise.h>

#include <thrust/functional.h>

#define BENCH_DEBUG

namespace benchmark {

    template<typename T>
    struct logic_or
    {
        typedef T first_argument_type;
        typedef T second_argument_type;
        typedef T result_type;
        __thrust_exec_check_disable__
        __host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs | rhs;}
    };

    template<typename T>
    struct logic_and
    {
        typedef T first_argument_type;
        typedef T second_argument_type;
        typedef T result_type;
        __thrust_exec_check_disable__
        __host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs & rhs;}
    };

    typedef bool value_type;
    static const value_type t = true;

    class Add: public BenchmarkBase {
    public:

        Add(int argc, const char** argv) {
            argsProcessor.parse(argc, argv);
            assert(argsProcessor.isParsed());

            benchmarkName = "Cusp-Add";
            experimentsCount = argsProcessor.getExperimentsCount();
        }

    protected:

        void setupBenchmark() override {

        }

        void tearDownBenchmark() override {

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

                host_matrix_t hostData(n, n, input.nvals);

                for (auto i = 0; i < input.nvals; i++) {
                    hostData.row_indices[i] = (int) input.rows[i];
                    hostData.column_indices[i] = (int) input.cols[i];
                    hostData.values[i] = t;
                }

                A = std::move(device_matrix_t(hostData));
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

                host_matrix_t hostData(n, n, input.nvals);

                for (size_t i = 0; i < input.nvals; i++) {
                    hostData.row_indices[i] = (int) input.rows[i];
                    hostData.column_indices[i] = (int) input.cols[i];
                    hostData.values[i] = t;
                }

                A2 = std::move(device_matrix_t(hostData));
            }
        }

        void tearDownExperiment(size_t experimentIdx) override {
            input = Matrix{};
            A = device_matrix_t{};
            A2 = device_matrix_t{};
        }

        void setupIteration(size_t experimentIdx, size_t iterationIdx) override {

        }

        void execIteration(size_t experimentIdx, size_t iterationIdx) override {
            // define multiply functors
            logic_or<value_type> reduce;

            // compute R = A + A2
            cusp::elementwise(A, A2, R, reduce);
        }

        void tearDownIteration(size_t experimentIdx, size_t iterationIdx) override {
#ifdef BENCH_DEBUG
            log << "   Result matrix: size " << R.num_rows << " x " << R.num_cols
                << " nvals " << R.num_entries << std::endl;
#endif

            R = device_matrix_t{};
        }

    protected:
        typedef cusp::coo_matrix<int, value_type, cusp::host_memory> host_matrix_t;
        typedef cusp::csr_matrix<int, value_type, cusp::device_memory> device_matrix_t;

        device_matrix_t A;
        device_matrix_t A2;
        device_matrix_t R;

        ArgsProcessor argsProcessor;
        Matrix input;

    };

}

int main(int argc, const char** argv) {
    benchmark::Add add(argc, argv);
    add.runBenchmark();
    return 0;
}