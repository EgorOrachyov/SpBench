//
// Created by Egor.Orachev on 26.01.2021.
//

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

    typedef char value_type;
    static const value_type t = true;

    class MultiplyAdd: public BenchmarkBase {
    public:

        MultiplyAdd(int argc, const char** argv) {
            argsProcessor.parse(argc, argv);
            assert(argsProcessor.isParsed());

            benchmarkName = "Cusp-Multiply-Add";
            experimentsCount = argsProcessor.getExperimentsCount();
        }

    protected:

        void setupBenchmark() override {

        }

        void tearDownBenchmark() override {

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

            hostData = host_matrix_t(n, n, input.nvals);

            for (auto i = 0; i < input.nvals; i++) {
                hostData.row_indices[i] = input.rows[i];
                hostData.column_indices[i] = input.cols[i];
                hostData.values[i] = t;
            }

            matrix = std::move(device_matrix_t(hostData));
        }

        void tearDownExperiment(size_t experimentIdx) override {
            input = Matrix{};
            matrix = device_matrix_t{};
        }

        void setupIteration(size_t experimentIdx, size_t iterationIdx) override {

        }

        void execIteration(size_t experimentIdx, size_t iterationIdx) override {
            // define multiply functors
            thrust::identity<value_type> identity;
            logic_and<value_type> combine;
            logic_or<value_type> reduce;

            device_matrix_t result;

            // compute R = M * M
            cusp::multiply(matrix, matrix, result, identity, combine, reduce);

            // compute R = R + M
            cusp::elementwise(matrix, result, result, reduce);

#ifdef BENCH_DEBUG
            std::cout << "   Result matrix: size " << result.num_rows << " x " << result.num_cols
                      << " nvals " << result.num_entries << std::endl;
#endif
        }

        void tearDownIteration(size_t experimentIdx, size_t iterationIdx) override {

        }

    protected:
        typedef cusp::coo_matrix<int, value_type, cusp::host_memory> host_matrix_t;
        typedef cusp::csr_matrix<int, value_type, cusp::device_memory> device_matrix_t;

        host_matrix_t hostData;
        device_matrix_t matrix;

        ArgsProcessor argsProcessor;
        Matrix input;

    };

}

int main(int argc, const char** argv) {
    benchmark::MultiplyAdd multiplyAdd(argc, argv);
    multiplyAdd.runBenchmark();
    return 0;
}