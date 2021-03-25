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
#include <profile_mem.hpp>

extern "C"
{
#include <GraphBLAS.h>
};

#define BENCH_DEBUG
#define GrB_CHECK(func) do { auto s = func; assert(s == GrB_SUCCESS); } while(0);

namespace benchmark {
    class Multiply: public BenchmarkBase {
    public:

        Multiply(int argc, const char** argv) {
            argsProcessor.parse(argc, argv);
            assert(argsProcessor.isParsed());

            benchmarkName = "SuiteSparse-Multiply";
            experimentsCount = argsProcessor.getExperimentsCount();
        }

        ~Multiply() {
            output_mem_profile(benchmarkName + "-Mem.txt", argsProcessor.getInputString());
        }

    protected:

        void setupBenchmark() override {
            GrB_CHECK(GrB_init(GrB_BLOCKING));
        }

        void tearDownBenchmark() override {
            GrB_CHECK(GrB_finalize());
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

            size_t n = input.nrows;
            assert(input.nrows == input.ncols);

            GrB_CHECK(GrB_Matrix_new(&A, GrB_BOOL, n, n));

            std::vector<GrB_Index> I(input.nvals);
            std::vector<GrB_Index> J(input.nvals);

            bool* X = (bool*) std::malloc(sizeof(bool) * input.nvals);

            for (auto i = 0; i < input.nvals; i++) {
                I[i] = input.rows[i];
                J[i] = input.cols[i];
                X[i] = true;
            }

            GrB_CHECK(GrB_Matrix_build_BOOL(A, I.data(), J.data(), X, input.nvals, GrB_FIRST_BOOL));

            std::free(X);
        }

        void tearDownExperiment(size_t experimentIdx) override {
            input = Matrix{};

            GrB_CHECK(GrB_Matrix_free(&A));
            A = nullptr;
        }

        void setupIteration(size_t experimentIdx, size_t iterationIdx) override {
            GrB_CHECK(GrB_Matrix_new(&R, GrB_BOOL, input.nrows, input.ncols));
        }

        void execIteration(size_t experimentIdx, size_t iterationIdx) override {
            GrB_CHECK(GrB_mxm(R, nullptr, nullptr, GrB_LOR_LAND_SEMIRING_BOOL, A, A, nullptr));
        }

        void tearDownIteration(size_t experimentIdx, size_t iterationIdx) override {
            GrB_Index nrows;
            GrB_Index ncols;
            GrB_Index nvals;

            GrB_CHECK(GrB_Matrix_nrows(&nrows, R));
            GrB_CHECK(GrB_Matrix_ncols(&ncols, R));
            GrB_CHECK(GrB_Matrix_nvals(&nvals, R));

#ifdef BENCH_DEBUG
            log << "   Result matrix: size " << nrows << " x " << ncols
                << " nvals " << nvals << std::endl;
#endif

            GrB_CHECK(GrB_Matrix_free(&R));
            R = nullptr;
        }

    protected:

        GrB_Matrix A;
        GrB_Matrix R;

        ArgsProcessor argsProcessor;
        Matrix input;
    };

}

int main(int argc, const char** argv) {
    benchmark::Multiply multiply(argc, argv);
    multiply.runBenchmark();
    return 0;
}