//
// Created by Egor.Orachev on 29.01.2021.
//

#include <benchmark_base.hpp>
#include <matrix_loader.hpp>
#include <args_processor.hpp>

extern "C"
{
#include <GraphBLAS.h>
};

#define BENCH_DEBUG
#define GrB_CHECK(func) do { auto s = func; assert(s == GrB_SUCCESS); } while(0);

namespace benchmark {
    class Add: public BenchmarkBase {
    public:

        Add(int argc, const char** argv) {
            argsProcessor.parse(argc, argv);
            assert(argsProcessor.isParsed());

            benchmarkName = "SuiteSparse-Add";
            experimentsCount = argsProcessor.getExperimentsCount();
        }

    protected:

        void setupBenchmark() override {
            GrB_CHECK(GrB_init(GrB_BLOCKING));
        }

        void tearDownBenchmark() override {
            GrB_CHECK(GrB_finalize());
        }

        void setupExperiment(size_t experimentIdx, size_t &iterationsCount) override {
            auto& entry = argsProcessor.getEntries()[experimentIdx];

            iterationsCount = entry.iterations;

            const auto& file = entry.name;
            const auto& type = entry.isUndirected;

            MatrixLoader loader(file, type);
            loader.loadData();
            input = std::move(loader.getMatrix());

#ifdef BENCH_DEBUG
            std::cout << ">   Load matrix: \"" << file << "\" isUndirected: " << type << std::endl
                      << "                 size: " << input.nrows << " x " << input.ncols << " nvals: " << input.nvals << std::endl;
#endif // BENCH_DEBUG

            size_t n = input.nrows;
            assert(input.nrows == input.ncols);

            GrB_CHECK(GrB_Matrix_new(&A, GrB_BOOL, n, n));
            GrB_CHECK(GrB_Matrix_new(&A2, GrB_BOOL, n, n));

            std::vector<GrB_Index> I(input.nvals);
            std::vector<GrB_Index> J(input.nvals);

            bool* X = (bool*) std::malloc(sizeof(bool) * input.nvals);

            for (auto i = 0; i < input.nvals; i++) {
                I[i] = input.rows[i];
                J[i] = input.cols[i];
                X[i] = true;
            }

            GrB_CHECK(GrB_Matrix_build_BOOL(A, I.data(), J.data(), X, input.nvals, GrB_FIRST_BOOL));
            GrB_CHECK(GrB_mxm(A2, nullptr, nullptr, GrB_LOR_LAND_SEMIRING_BOOL, A, A, nullptr));

            std::free(X);
        }

        void tearDownExperiment(size_t experimentIdx) override {
            input = Matrix{};

            GrB_CHECK(GrB_Matrix_free(&A));
            GrB_CHECK(GrB_Matrix_free(&A2));
            A = nullptr;
            A2 = nullptr;
        }

        void setupIteration(size_t experimentIdx, size_t iterationIdx) override {
            GrB_CHECK(GrB_Matrix_new(&R, GrB_BOOL, input.nrows, input.ncols));
        }

        void execIteration(size_t experimentIdx, size_t iterationIdx) override {
            GrB_CHECK(GrB_Matrix_eWiseAdd_BinaryOp(R, nullptr, nullptr, GrB_LOR, A, A2, nullptr));
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
        GrB_Matrix A2;
        GrB_Matrix R;

        ArgsProcessor argsProcessor;
        Matrix input;
    };

}

int main(int argc, const char** argv) {
    benchmark::Add add(argc, argv);
    add.runBenchmark();
    return 0;
}