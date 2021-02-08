//
// Created by Egor.Orachev on 27.01.2021.
//

#include <benchmark_base.hpp>
#include <matrix_loader.hpp>
#include <args_processor.hpp>

#include <cubool/cubool.h>

#define BENCH_DEBUG
#define CUBOOL_CHECK(func) do { auto s = func; assert(s == CUBOOL_STATUS_SUCCESS); } while(0)

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

            CUBOOL_CHECK(CuBool_Instance_New(&desc, &instance));
        }

        void tearDownBenchmark() override {
            CUBOOL_CHECK(CuBool_Instance_Free(instance));
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

            CUBOOL_CHECK(CuBool_Matrix_New(instance, &matrix, n, n));
            CUBOOL_CHECK(CuBool_Matrix_Build(instance, matrix, input.rows.data(), input.cols.data(), input.nvals));
        }

        void tearDownExperiment(size_t experimentIdx) override {
            input = Matrix{};

            CUBOOL_CHECK(CuBool_Matrix_Free(instance, matrix));

            matrix = nullptr;
        }

        void setupIteration(size_t experimentIdx, size_t iterationIdx) override {
            CuBoolIndex_t n = input.nrows;
            assert(input.nrows == input.ncols);

            CUBOOL_CHECK(CuBool_Matrix_New(instance, &result, n, n));
        }

        void execIteration(size_t experimentIdx, size_t iterationIdx) override {
            CUBOOL_CHECK(CuBool_MxM(instance, result, matrix, matrix));
        }

        void tearDownIteration(size_t experimentIdx, size_t iterationIdx) override {
            if (matrix) {
                CuBoolSize_t nvals;
                CuBoolIndex_t nrows;
                CuBoolIndex_t ncols;

                CUBOOL_CHECK(CuBool_Matrix_Nrows(instance, result, &nrows));
                CUBOOL_CHECK(CuBool_Matrix_Ncols(instance, result, &ncols));
                CUBOOL_CHECK(CuBool_Matrix_Nvals(instance, result, &nvals));

#ifdef BENCH_DEBUG
                log << "   Result matrix: size: " << nrows << " x " << ncols << " nvals: " << nvals << std::endl;
#endif // BENCH_DEBUG

                CUBOOL_CHECK(CuBool_Matrix_Free(instance, result));

                result = nullptr;
            }
        }


    protected:

        ArgsProcessor argsProcessor;

        CuBoolInstance instance = nullptr;
        CuBoolMatrix matrix = nullptr;
        CuBoolMatrix result = nullptr;

        Matrix input;

    };

}

int main(int argc, const char** argv) {
    benchmark::Multiply multiply(argc, argv);
    multiply.runBenchmark();
    return 0;
}