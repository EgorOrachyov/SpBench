//
// Created by Egor.Orachev on 27.01.2021.
//

#include <benchmark_base.hpp>
#include <matrix_loader.hpp>
#include <args_processor.hpp>

#include <cubool/cubool.h>

#define BENCH_DEBUG

namespace benchmark {

    class Multiply: public BenchmarkBase {
    public:

        Multiply(int argc, const char** argv) {
            argsProcessor.parse(argc, argv);
            assert(argsProcessor.isParsed());

            benchmarkName = "Cubool-Multiply";
            experimentsCount = argsProcessor.getExperimentsCount();
        }

    protected:

        static void messageCallback(CuBoolStatus status, const char* message, void* userData) {
            std::cout << "= Cubool: Status: " << status << " Message: \"" << message << "\"" << std::endl;
        }

        void setupBenchmark() override {
            CuBoolInstanceDesc desc{};
            desc.errorCallback = {nullptr, messageCallback };
            desc.memoryType = CUBOOL_GPU_MEMORY_TYPE_MANAGED;

            status = CuBool_Instance_New(&desc, &instance);
            assert(status == CUBOOL_STATUS_SUCCESS);
        }

        void tearDownBenchmark() override {
            status = CuBool_Instance_Free(instance);
            assert(status == CUBOOL_STATUS_SUCCESS);
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

            matrix = nullptr;
        }

        void setupIteration(size_t experimentIdx, size_t iterationIdx) override {
            CuBoolIndex_t n = input.nrows;
            assert(input.nrows == input.ncols);

            status = CuBool_Matrix_New(instance, &result, n, n);
            assert(status == CUBOOL_STATUS_SUCCESS);
        }

        void execIteration(size_t experimentIdx, size_t iterationIdx) override {
            status = CuBool_MxM(instance, result, matrix, matrix);
            assert(status == CUBOOL_STATUS_SUCCESS);
        }

        void tearDownIteration(size_t experimentIdx, size_t iterationIdx) override {
            if (matrix) {
                CuBoolSize_t nvals;
                CuBoolIndex_t nrows;
                CuBoolIndex_t ncols;

                status = CuBool_Matrix_Nrows(instance, result, &nrows);
                assert(status == CUBOOL_STATUS_SUCCESS);

                status = CuBool_Matrix_Ncols(instance, result, &ncols);
                assert(status == CUBOOL_STATUS_SUCCESS);

                status = CuBool_Matrix_Nvals(instance, result, &nvals);
                assert(status == CUBOOL_STATUS_SUCCESS);

#ifdef BENCH_DEBUG
                log << "   Result matrix: size: " << nrows << " x " << ncols << " nvals: " << nvals << std::endl;
#endif // BENCH_DEBUG

                status = CuBool_Matrix_Free(instance, result);
                assert(status == CUBOOL_STATUS_SUCCESS);

                result = nullptr;
            }
        }


    protected:

        ArgsProcessor argsProcessor;

        CuBoolInstance instance = nullptr;
        CuBoolMatrix matrix = nullptr;
        CuBoolMatrix result = nullptr;
        CuBoolStatus status = CUBOOL_STATUS_SUCCESS;

        Matrix input;

    };

}

int main(int argc, const char** argv) {
    benchmark::Multiply multiply(argc, argv);
    multiply.runBenchmark();
    return 0;
}