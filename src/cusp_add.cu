//
// Created by Egor.Orachev on 27.01.2021.
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

    class Add: public BenchmarkBase {
    public:

        Add(int argc, const char** argv) {
            argsProcessor.parse(argc, argv);
            assert(argsProcessor.isParsed());

            benchmarkName = "Cusp-Add";
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
            std::cout << ">   Load A: \"" << file << "\" isUndirected: " << type << std::endl
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

            A = std::move(device_matrix_t(hostData));

            thrust::identity<value_type> identity;
            logic_and<value_type> combine;
            logic_or<value_type> reduce;

            // compute M2 = M * M
            cusp::multiply(A, A, A2, identity, combine, reduce);
        }

        void tearDownExperiment(size_t experimentIdx) override {
            input = Matrix{};
            A = device_matrix_t{};
            A2 = device_matrix_t{};
        }

        void setupIteration(size_t experimentIdx, size_t iterationIdx) override {

        }

        void execIteration(size_t experimentIdx, size_t iterationIdx) override {
            // define multiply functors
            logic_or<value_type> reduce;

            // compute R = A + A2
            cusp::elementwise(A, A2, R, reduce);

#ifdef BENCH_DEBUG
            log << "   Result matrix: size " << R.num_rows << " x " << R.num_cols
                << " nvals " << R.num_entries << std::endl;
#endif
        }

        void tearDownIteration(size_t experimentIdx, size_t iterationIdx) override {
            R = device_matrix_t{};
        }

    protected:
        typedef cusp::coo_matrix<int, value_type, cusp::host_memory> host_matrix_t;
        typedef cusp::csr_matrix<int, value_type, cusp::device_memory> device_matrix_t;

        host_matrix_t hostData;
        device_matrix_t A;
        device_matrix_t A2;
        device_matrix_t R;

        ArgsProcessor argsProcessor;
        Matrix input;

    };

}

int main(int argc, const char** argv) {
    benchmark::Add add(argc, argv);
    add.runBenchmark();
    return 0;
}