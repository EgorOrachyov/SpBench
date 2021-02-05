//
// Created by Egor.Orachev on 27.01.2021.
//

#include <benchmark_base.hpp>
#include <matrix_loader.hpp>
#include <args_processor.hpp>

#include <cubool/cubool.h>

#define BENCH_DEBUG

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

            CuBoolIndex_t n = input.nrows;
            assert(input.nrows == input.ncols);

            status = CuBool_Matrix_New(instance, &A, n, n);
            assert(status == CUBOOL_STATUS_SUCCESS);

            status = CuBool_Matrix_Build(instance, A, input.rows.data(), input.cols.data(), input.nvals);
            assert(status == CUBOOL_STATUS_SUCCESS);

            status = CuBool_Matrix_New(instance, &A2, n, n);
            assert(status == CUBOOL_STATUS_SUCCESS);

            status = CuBool_MxM(instance, A2, A, A);
            assert(status == CUBOOL_STATUS_SUCCESS);
        }

        void tearDownExperiment(size_t experimentIdx) override {
            input = Matrix{};

            status = CuBool_Matrix_Free(instance, A);
            assert(status == CUBOOL_STATUS_SUCCESS);

            status = CuBool_Matrix_Free(instance, A2);
            assert(status == CUBOOL_STATUS_SUCCESS);

            A = nullptr;
            A2 = nullptr;
        }

        void setupIteration(size_t experimentIdx, size_t iterationIdx) override {
            status = CuBool_Matrix_Duplicate(instance, A, &R);
            assert(status == CUBOOL_STATUS_SUCCESS);
        }


        void execIteration(size_t experimentIdx, size_t iterationIdx) override {
            status = CuBool_Matrix_Add(instance, R, A2);
            assert(status == CUBOOL_STATUS_SUCCESS);
        }

        void tearDownIteration(size_t experimentIdx, size_t iterationIdx) override {
            if (A) {
                CuBoolSize_t nvals;
                CuBoolIndex_t nrows;
                CuBoolIndex_t ncols;

                status = CuBool_Matrix_Nrows(instance, R, &nrows);
                assert(status == CUBOOL_STATUS_SUCCESS);

                status = CuBool_Matrix_Ncols(instance, R, &ncols);
                assert(status == CUBOOL_STATUS_SUCCESS);

                status = CuBool_Matrix_Nvals(instance, R, &nvals);
                assert(status == CUBOOL_STATUS_SUCCESS);

#ifdef BENCH_DEBUG
                log << "   Result matrix: size: " << nrows << " x " << ncols << " nvals: " << nvals << std::endl;
#endif // BENCH_DEBUG

                status = CuBool_Matrix_Free(instance, R);
                assert(status == CUBOOL_STATUS_SUCCESS);

                R = nullptr;
            }
        }


    protected:

        ArgsProcessor argsProcessor;

        CuBoolInstance instance = nullptr;
        CuBoolMatrix A = nullptr;
        CuBoolMatrix A2 = nullptr;
        CuBoolMatrix R = nullptr;
        CuBoolStatus status = CUBOOL_STATUS_SUCCESS;

        Matrix input;

    };

}

int main(int argc, const char** argv) {
    benchmark::Add add(argc, argv);
    add.runBenchmark();
    return 0;
}