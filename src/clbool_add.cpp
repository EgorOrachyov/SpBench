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

            matrix_coo_cpu_pairs matrixA;
            matrixA.reserve(input.nvals);

            for (auto i = 0; i < input.nvals; i++) {
                matrixA.push_back({ input.rows[i], input.cols[i] });
            }

            matrix_dcsr_cpu matrixDcsrA = coo_utils::coo_to_dcsr_cpu(matrixA);

            matrix_dcsr dcsrA = matrix_dcsr_from_cpu(*controls, matrixDcsrA, n);
            matrix_dcsr dcsrA2;

            // A2 = A * A
            matrix_multiplication(*controls, dcsrA2, dcsrA, dcsrA);

            A = std::move(dcsr_to_coo(*controls, dcsrA));
            A2 = std::move(dcsr_to_coo(*controls, dcsrA2));
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

#ifdef BENCH_DEBUG
            log << "   Result matrix: size " << R.nRows() << " x " << R.nCols()
                << " nvals " << R.nnz() << std::endl;
#endif
        }

        void tearDownIteration(size_t experimentIdx, size_t iterationIdx) override {
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