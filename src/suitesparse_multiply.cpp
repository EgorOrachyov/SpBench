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

namespace benchmark {
    class Multiply: public BenchmarkBase {
    public:

        Multiply(int argc, const char** argv) {
            argsProcessor.parse(argc, argv);
            assert(argsProcessor.isParsed());

            benchmarkName = "SuiteSparse-Multiply";
            experimentsCount = argsProcessor.getExperimentsCount();
        }

    protected:

        void setupBenchmark() override {
            GrB_init(GrB_BLOCKING);
        }

        void tearDownBenchmark() override {
            GrB_finalize();
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

        }

        void tearDownExperiment(size_t experimentIdx) override {
            input = Matrix{};
        }

        void setupIteration(size_t experimentIdx, size_t iterationIdx) override {

        }

        void execIteration(size_t experimentIdx, size_t iterationIdx) override {

#ifdef BENCH_DEBUGsad
            log << "   Result matrix: size " << R.nRows() << " x " << R.nCols()
                << " nvals " << R.nnz() << std::endl;
#endif
        }

        void tearDownIteration(size_t experimentIdx, size_t iterationIdx) override {
        }

    protected:

        ArgsProcessor argsProcessor;
        Matrix input;
    };

}

int main(int argc, const char** argv) {
    benchmark::Multiply multiply(argc, argv);
    multiply.runBenchmark();
    return 0;
}