//
// Created by Egor.Orachev on 27.01.2021.
//

#include <benchmark_base.hpp>
#include <matrix_loader.hpp>
#include <args_processor.hpp>

#include <clSPARSE.h>
#include <clSPARSE-error.h>

#include <CL/cl.hpp>

#define BENCH_DEBUG

namespace benchmark {
    class Multiply: public BenchmarkBase {
    public:

        Multiply(int argc, const char** argv) {
            argsProcessor.parse(argc, argv);
            assert(argsProcessor.isParsed());

            benchmarkName = "clSPARSE-Multiply";
            experimentsCount = argsProcessor.getExperimentsCount();
        }

    protected:

        void setupBenchmark() override {
            std::vector<cl::Platform> platforms;
            cl_int clStatus = cl::Platform::get(&platforms);

            if (clStatus != CL_SUCCESS) {
                std::cerr << "Failed to get OpenCL platforms " << clStatus << std::endl;
                return;
            }

            std::string keyWords[] = { "cuda", "CUDA", "Cuda", "NVIDIA", "nvidia", "Nvidia"};

            int selectedPlatformId = -1;
            int platformId = -1;
            bool foundCuda = false;
            for (const auto& p : platforms) {
                platformId += 1;

                std::cout << "Platform ID " << platformId << " : " << p.getInfo<CL_PLATFORM_NAME>() << std::endl;
                auto info = p.getInfo<CL_PLATFORM_NAME>();

                for (auto& k: keyWords) {
                    if (info.find(k) != std::basic_string<char, std::char_traits<char>, std::allocator<char>>::npos) {
                        foundCuda = true;
                        selectedPlatformId = platformId;
                        std::cout << "Select Platform ID " << platformId << " : " << p.getInfo<CL_PLATFORM_NAME>() << std::endl;
                        break;
                    }
                }

                if (foundCuda)
                    break;
            }

            clPlatform = platforms[selectedPlatformId];


            std::vector<cl::Device> devices;
            clStatus = clPlatform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

            if (clStatus != CL_SUCCESS) {
                std::cerr << "Problem with getting devices from platform"
                          << " [" << selectedPlatformId << "] " << clPlatform.getInfo<CL_PLATFORM_NAME>()
                          << " error: [" << clStatus << "]" << std::endl;

                return;
            }

            cl_int deviceId = 0;
            for (const auto& device : devices) {
                std::cout << "Device ID " << deviceId++ << " : " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            }

            int selectedDeviceId = 0;
            clDevice = devices[selectedDeviceId];
            clContext = cl::Context(clDevice);
            clCommandQueue = cl::CommandQueue(clContext, clDevice);

            clsparseStatus status = clsparseSetup();
            if (status != clsparseSuccess) {
                std::cerr << "Problem with executing clsparseSetup()" << std::endl;
                return;
            }

            clsparseCreateResult createResult = clsparseCreateControl( clCommandQueue( ) );
            CLSPARSE_V( createResult.status, "Failed to create clsparse control" );

            control = createResult.control;
        }

        void tearDownBenchmark() override {
            Status = clsparseReleaseControl(control);
            assert(Status == clsparseStatus::clsparseSuccess);

            clsparseStatus status = clsparseTeardown();
            if (status != clsparseSuccess) {
                std::cerr << "Problem with executing clsparseSetup()" << std::endl;
                return;
            }
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

            size_t n = input.nrows;
            assert(input.nrows == input.ncols);

            Status = clsparseInitCsrMatrix(&M);
            assert(Status == clsparseStatus::clsparseSuccess);

            std::vector<clsparseIdx_t> rowsPtr(n + 1, 0);
            std::vector<clsparseIdx_t> colsInd(input.nvals);
            std::vector<float> values(input.nvals, 1.0f);

            for (auto i = 0; i < input.nvals; i++) {
                rowsPtr[input.rows[i]] += 1;
                colsInd[i] = input.cols[i];
            }

            clsparseIdx_t sum = 0;
            for (auto& r: rowsPtr) {
                clsparseIdx_t prev = sum;
                sum += r;
                r = prev;
            }

            M.num_rows = n;
            M.num_cols = n;
            M.num_nonzeros = input.nvals;

            M.row_pointer = ::clCreateBuffer(clContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(clsparseIdx_t) * (rowsPtr.size()), rowsPtr.data(),&clStatus);
            assert(M.row_pointer != nullptr);

            M.col_indices = ::clCreateBuffer(clContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(clsparseIdx_t) * (colsInd.size()), colsInd.data() ,&clStatus);
            assert(M.col_indices != nullptr);

            M.values = ::clCreateBuffer(clContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * (values.size()), values.data() ,&clStatus);
            assert(M.values != nullptr);
        }

        void tearDownExperiment(size_t experimentIdx) override {
            input = Matrix{};

            clReleaseMemObject(M.values);
            clReleaseMemObject(M.col_indices);
            clReleaseMemObject(M.row_pointer);
        }

        void setupIteration(size_t experimentIdx, size_t iterationIdx) override {
            clsparseInitCsrMatrix(&R);
        }

        void execIteration(size_t experimentIdx, size_t iterationIdx) override {
            Status = clsparseScsrSpGemm(&M, &M, &R, control);
            assert(Status == clsparseStatus::clsparseSuccess);
        }

        void tearDownIteration(size_t experimentIdx, size_t iterationIdx) override {
#ifdef BENCH_DEBUG
            log << "   Result matrix: size " << R.num_rows << " x " << R.num_cols
                << " nvals " << R.num_nonzeros << std::endl;
#endif

            clReleaseMemObject(R.values);
            clReleaseMemObject(R.col_indices);
            clReleaseMemObject(R.row_pointer);
        }

    protected:

        cl::Platform clPlatform;
        cl::Device clDevice;
        cl::Context clContext;
        cl::CommandQueue clCommandQueue;

        clsparseCsrMatrix M;
        clsparseCsrMatrix R;

        clsparseControl control;
        clsparseStatus Status;
        cl_int clStatus;


        ArgsProcessor argsProcessor;
        Matrix input;
    };

}

int main(int argc, const char** argv) {
    benchmark::Multiply multiply(argc, argv);
    multiply.runBenchmark();
    return 0;
}