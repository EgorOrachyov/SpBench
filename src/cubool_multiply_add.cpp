//
// Created by Egor.Orachev on 26.01.2021.
//

#include <benchmark_base.hpp>
#include <matrix_loader.hpp>
#include <args_processor.hpp>

#include <cubool/cubool.h>

#define BENCH_DEBUG

namespace benchmark {

    class MultiplyAdd: public BenchmarkBase {
    public:

        MultiplyAdd(int argc, const char** argv) {
            argsProcessor.parse(argc, argv);
            assert(argsProcessor.isParsed());

            benchmarkName = "Cubool-Multiply-Add";
            experimentsCount = argsProcessor.getExperimentsCount();
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
        }

        void tearDownExperiment(size_t experimentIdx) override {
            input = Matrix{};
            assert(matrix == nullptr);
        }

        void setupIteration(size_t experimentIdx, size_t iterationIdx) override {
            CuBoolIndex_t n = input.nrows;
            assert(input.nrows == input.ncols);

            status = CuBool_Matrix_New(instance, &matrix, n, n);
            assert(status == CUBOOL_STATUS_SUCCESS);

            status = CuBool_Matrix_Build(instance, matrix, input.rows.data(), input.cols.data(), input.nvals);
            assert(status == CUBOOL_STATUS_SUCCESS);
        }

        void execIteration(size_t experimentIdx, size_t iterationIdx) override {
            status = CuBool_MxM(instance, matrix, matrix, matrix);
            assert(status == CUBOOL_STATUS_SUCCESS);
        }

        void tearDownIteration(size_t experimentIdx, size_t iterationIdx) override {
            if (matrix) {
                CuBoolSize_t nvals;
                CuBoolIndex_t nrows;
                CuBoolIndex_t ncols;

                status = CuBool_Matrix_Nrows(instance, matrix, &nrows);
                assert(status == CUBOOL_STATUS_SUCCESS);

                status = CuBool_Matrix_Ncols(instance, matrix, &ncols);
                assert(status == CUBOOL_STATUS_SUCCESS);

                status = CuBool_Matrix_Nvals(instance, matrix, &nvals);
                assert(status == CUBOOL_STATUS_SUCCESS);

#ifdef BENCH_DEBUG
                log << "   Result matrix: size: " << nrows << " x " << ncols << " nvals: " << nvals << std::endl;
#endif // BENCH_DEBUG

                status = CuBool_Matrix_Free(instance, matrix);
                assert(status == CUBOOL_STATUS_SUCCESS);

                matrix = nullptr;
            }
        }


    protected:

        ArgsProcessor argsProcessor;

        CuBoolInstance instance = nullptr;
        CuBoolMatrix matrix = nullptr;
        CuBoolStatus status = CUBOOL_STATUS_SUCCESS;

        Matrix input;

    };

}

int main(int argc, const char** argv) {
    benchmark::MultiplyAdd multiplyAdd(argc, argv);
    multiplyAdd.runBenchmark();
    return 0;
}