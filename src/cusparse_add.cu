//
// Created by Egor.Orachev on 27.01.2021.
//

#include <benchmark_base.hpp>
#include <matrix_loader.hpp>
#include <args_processor.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cusparse_v2.h>

#define CUSPARSE_CHECH(function) { auto statusCall = function; assert(statusCall == CUSPARSE_STATUS_SUCCESS); }

#define BENCH_DEBUG

namespace benchmark {

    struct CsrMatrix {
        CsrMatrix() = default;

        CsrMatrix(const CsrMatrix &other) = default;

        CsrMatrix(CsrMatrix &&other) = default;

        CsrMatrix &operator=(const CsrMatrix &other) = default;

        CsrMatrix &operator=(CsrMatrix &&other) = default;

        void release() {
            rows.clear();
            cols.clear();
            nvals = 0;
            n = 0;
        }

        cusparseMatDescr_t desc{};
        thrust::device_vector<int> rows;
        thrust::device_vector<int> cols;
        int nvals = 0;
        int n = 0;
    };

    class Add : public BenchmarkBase {
    public:

        Add(int argc, const char **argv) {
            argsProcessor.parse(argc, argv);
            assert(argsProcessor.isParsed());

            benchmarkName = "cuSPARSE-Add";
            experimentsCount = argsProcessor.getExperimentsCount();
        }

    protected:

        void setupBenchmark() override {
            CUSPARSE_CHECH(cusparseCreate(&handle));
        }

        void tearDownBenchmark() override {
            cusparseDestroy(handle);
            handle = nullptr;
        }

        void setupExperiment(size_t experimentIdx, size_t &iterationsCount) override {
            auto &entry = argsProcessor.getEntries()[experimentIdx];

            iterationsCount = entry.iterations;

            const auto &file = entry.name;
            const auto &type = entry.isUndirected;

            MatrixLoader loader(file, type);
            loader.loadData();
            input = std::move(loader.getMatrix());

#ifdef BENCH_DEBUG
            std::cout << ">   Load matrix: \"" << file << "\" isUndirected: " << type << std::endl
                      << "                 size: " << input.nrows << " x " << input.ncols << " nvals: " << input.nvals
                      << std::endl;
#endif // BENCH_DEBUG

            size_t n = input.nrows;
            assert(input.nrows == input.ncols);

            thrust::host_vector<int> rowsPtr(n + 1, 0);
            thrust::host_vector<int> colsInd(input.nvals);

            for (auto i = 0; i < input.nvals; i++) {
                rowsPtr[input.rows[i]] += 1;
                colsInd[i] = input.cols[i];
            }

            int sum = 0;
            for (auto &r: rowsPtr) {
                int prev = sum;
                sum += r;
                r = prev;
            }

            CUSPARSE_CHECH(cusparseCreateMatDescr(&A.desc));
            CUSPARSE_CHECH(cusparseSetMatType(A.desc, CUSPARSE_MATRIX_TYPE_GENERAL));
            CUSPARSE_CHECH(cusparseSetMatIndexBase(A.desc, CUSPARSE_INDEX_BASE_ZERO));

            A.nvals = input.nvals;
            A.n = n;
            A.rows.resize(rowsPtr.size());
            A.cols.resize(colsInd.size());
            values.resize(A.nvals, 1.0f);

            thrust::copy(rowsPtr.begin(), rowsPtr.end(), A.rows.begin());
            thrust::copy(colsInd.begin(), colsInd.end(), A.cols.begin());

            // A2 = A * A

            CUSPARSE_CHECH(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

            CUSPARSE_CHECH(cusparseCreateMatDescr(&A2.desc));
            CUSPARSE_CHECH(cusparseSetMatType(A2.desc, CUSPARSE_MATRIX_TYPE_GENERAL));
            CUSPARSE_CHECH(cusparseSetMatIndexBase(A2.desc, CUSPARSE_INDEX_BASE_ZERO));

            A2.rows.resize(A.n + 1);

            int nnzA2 = 0;
            int *nnzTotalDevHostPtr = &nnzA2;

            CUSPARSE_CHECH(cusparseXcsrgemmNnz(
                    handle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    A.n, A.n, A.n,
                    A.desc, A.nvals, A.rows.data().get(), A.cols.data().get(),
                    A.desc, A.nvals, A.rows.data().get(), A.cols.data().get(),
                    A2.desc, A2.rows.data().get(), nnzTotalDevHostPtr
            ));

            assert(nnzTotalDevHostPtr != nullptr);
            nnzA2 = *nnzTotalDevHostPtr;

            A2.n = A.n;
            A2.nvals = nnzA2;
            A2.cols.resize(nnzA2);

            valuesA2.resize(nnzA2, 1.0f);

            CUSPARSE_CHECH(cusparseScsrgemm(
                    handle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    A.n, A.n, A.n,
                    A.desc, A.nvals, values.data().get(), A.rows.data().get(), A.cols.data().get(),
                    A.desc, A.nvals, values.data().get(), A.rows.data().get(), A.cols.data().get(),
                    A2.desc, valuesA2.data().get(), A2.rows.data().get(), A2.cols.data().get()
            ));
        }

        void tearDownExperiment(size_t experimentIdx) override {
            input = Matrix{};

            CUSPARSE_CHECH(cusparseDestroyMatDescr(A.desc));
            A.release();

            CUSPARSE_CHECH(cusparseDestroyMatDescr(A2.desc));
            A2.release();
        }

        void setupIteration(size_t experimentIdx, size_t iterationIdx) override {
            CUSPARSE_CHECH(cusparseCreateMatDescr(&R.desc));
            CUSPARSE_CHECH(cusparseSetMatType(R.desc, CUSPARSE_MATRIX_TYPE_GENERAL));
            CUSPARSE_CHECH(cusparseSetMatIndexBase(R.desc, CUSPARSE_INDEX_BASE_ZERO));

            R.rows.resize(A.n + 1);
        }

        void execIteration(size_t experimentIdx, size_t iterationIdx) override {
            CUSPARSE_CHECH(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

            int nnzC = 0;
            int *nnzTotalDevHostPtr = &nnzC;

            CUSPARSE_CHECH(cusparseXcsrgeamNnz(
                    handle,
                    A.n, A.n,
                    A.desc, A.nvals, A.rows.data().get(), A.cols.data().get(),
                    A2.desc, A2.nvals, A2.rows.data().get(), A2.cols.data().get(),
                    R.desc, R.rows.data().get(), nnzTotalDevHostPtr
            ));

            assert(nnzTotalDevHostPtr != nullptr);
            nnzC = *nnzTotalDevHostPtr;

            R.n = A.n;
            R.nvals = nnzC;
            R.cols.resize(nnzC);

            float alpha = 1.0f;
            float beta = 1.0f;

            thrust::device_vector<float> tmp(nnzC);

            CUSPARSE_CHECH(cusparseScsrgeam(
                    handle,
                    A.n, A.n,
                    &alpha,
                    A.desc, A.nvals, values.data().get(), A.rows.data().get(), A.cols.data().get(),
                    &beta,
                    A2.desc, A2.nvals, valuesA2.data().get(), A2.rows.data().get(), A2.cols.data().get(),
                    R.desc, tmp.data().get(), R.rows.data().get(), R.cols.data().get()
            ));
        }

        void tearDownIteration(size_t experimentIdx, size_t iterationIdx) override {
#ifdef BENCH_DEBUG
            log << "   Result matrix: size " << R.n << " x " << R.n
                << " nvals " << R.nvals << std::endl;
#endif

            CUSPARSE_CHECH(cusparseDestroyMatDescr(R.desc));
            R.release();
        }

    protected:

        cusparseHandle_t handle;

        CsrMatrix A;
        CsrMatrix A2;
        CsrMatrix R;
        thrust::device_vector<float> values;
        thrust::device_vector<float> valuesA2;

        ArgsProcessor argsProcessor;
        Matrix input;

    };

}

int main(int argc, const char **argv) {
    benchmark::Add add(argc, argv);
    add.runBenchmark();
    return 0;
}