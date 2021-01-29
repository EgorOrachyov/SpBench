//
// Created by Egor.Orachev on 28.01.2021.
//

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

#define BENCH_DEBUG

namespace benchmark {
    class Multiply: public BenchmarkBase {
    public:

        Multiply(int argc, const char** argv) {
            argsProcessor.parse(argc, argv);
            assert(argsProcessor.isParsed());

            benchmarkName = "Clbool-Multiply";
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
            std::cout << ">   Load matrix: \"" << file << "\" isUndirected: " << type << std::endl
                      << "                 size: " << input.nrows << " x " << input.ncols << " nvals: " << input.nvals << std::endl;
#endif // BENCH_DEBUG

            size_t n = input.nrows;
            assert(input.nrows == input.ncols);

            matrix_coo_cpu_pairs matrixA;
            matrixA.reserve(input.nvals);

            for (auto i = 0; i < input.nvals; i++) {
                matrixA.push_back({ input.rows[i], input.cols[i] });
            }

            matrix_dcsr_cpu matrixDcsrA = coo_utils::coo_to_dcsr_cpu(matrixA);

            A = std::move(matrix_dcsr_from_cpu(*controls, matrixDcsrA, n));
        }

        void tearDownExperiment(size_t experimentIdx) override {
            input = Matrix{};
            A = matrix_dcsr{};
        }

        void setupIteration(size_t experimentIdx, size_t iterationIdx) override {

        }

        void execIteration(size_t experimentIdx, size_t iterationIdx) override {
            matrix_multiplication(*controls, R, A, A);
        }

        void tearDownIteration(size_t experimentIdx, size_t iterationIdx) override {
#ifdef BENCH_DEBUG
            log << "   Result matrix: size " << R.nRows() << " x " << R.nCols()
                << " nvals " << R.nnz() << std::endl;
#endif

            R = matrix_dcsr{};
        }

    protected:

        Controls* controls;
        matrix_dcsr A;
        matrix_dcsr R;

        ArgsProcessor argsProcessor;
        Matrix input;
    };

}

int main(int argc, const char** argv) {
    benchmark::Multiply multiply(argc, argv);
    multiply.runBenchmark();
    return 0;
}