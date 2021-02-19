////////////////////////////////////////////////////////////////////////////////////
// MIT License                                                                    //
//                                                                                //
// Copyright (c) 2021 Egor Orachyov                                               //
//                                                                                //
// Permission is hereby granted, free of charge, to any person obtaining a copy   //
// of this software and associated documentation files (the "Software"), to deal  //
// in the Software without restriction, including without limitation the rights   //
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      //
// copies of the Software, and to permit persons to whom the Software is          //
// furnished to do so, subject to the following conditions:                       //
//                                                                                //
// The above copyright notice and this permission notice shall be included in all //
// copies or substantial portions of the Software.                                //
//                                                                                //
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     //
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       //
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    //
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         //
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  //
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  //
// SOFTWARE.                                                                      //
////////////////////////////////////////////////////////////////////////////////////

#include <args_processor.hpp>
#include <matrix_loader.hpp>
#include <matrix_writer.hpp>
#include <cubool/cubool.h>

using namespace benchmark;

#define CUBOOL_CHECK(func) do { auto s = func; assert(s == CUBOOL_STATUS_SUCCESS); } while(0)

static void messageCallback(CuBoolStatus status, const char* message, void* userData) {
    std::cout << "= Cubool: Status: " << status << " Message: \"" << message << "\"" << std::endl;
}

int main(int argc, const char** argv) {
    ArgsProcessor argsProcessor;
    Matrix input;

    argsProcessor.parse(argc, argv);

    CuBoolInstance instance = nullptr;
    CuBoolMatrix matrix = nullptr;
    CuBoolMatrix result = nullptr;

    CuBoolInstanceDesc desc{};
    desc.errorCallback = {nullptr, messageCallback };
    desc.memoryType = CUBOOL_GPU_MEMORY_TYPE_MANAGED;

    CUBOOL_CHECK(CuBool_Instance_New(&desc, &instance));

    for (auto& entry: argsProcessor.getEntries()) {
        const auto& file = entry.name;
        const auto& type = entry.isUndirected;

        MatrixLoader loader(file, type);
        loader.loadData();
        input = std::move(loader.getMatrix());

        CuBoolIndex_t n = input.nrows;
        assert(input.nrows == input.ncols);

        CUBOOL_CHECK(CuBool_Matrix_New(instance, &matrix, n, n));
        CUBOOL_CHECK(CuBool_Matrix_Build(instance, matrix, input.rows.data(), input.cols.data(), input.nvals));

        CUBOOL_CHECK(CuBool_Matrix_New(instance, &result, input.nrows, input.nrows));
        CUBOOL_CHECK(CuBool_MxM(instance, result, matrix, matrix));

        CuBoolSize_t nvals;
        CuBoolIndex_t nrows;
        CuBoolIndex_t ncols;

        CUBOOL_CHECK(CuBool_Matrix_Nrows(instance, result, &nrows));
        CUBOOL_CHECK(CuBool_Matrix_Ncols(instance, result, &ncols));
        CUBOOL_CHECK(CuBool_Matrix_Nvals(instance, result, &nvals));

        std::cout << "Result matrix " << file << "2 : size: " << nrows << " x " << ncols << " nvals: " << nvals << std::endl;

        Matrix m2;
        m2.nrows = nrows;
        m2.ncols = ncols;
        m2.nvals = nvals;
        m2.rows.resize(nvals);
        m2.cols.resize(nvals);

        CUBOOL_CHECK(CuBool_Matrix_ExtractPairs(instance, result, m2.rows.data(), m2.cols.data(), &m2.nvals));

        MatrixWriter writer;
        writer.save(file + "2", m2);

        CUBOOL_CHECK(CuBool_Matrix_Free(instance, result));
        CUBOOL_CHECK(CuBool_Matrix_Free(instance, matrix));

        result = nullptr;
        matrix = nullptr;
    }

    CUBOOL_CHECK(CuBool_Instance_Free(instance));

    return 0;
}