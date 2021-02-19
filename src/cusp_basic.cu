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

// The original example is https://cusplibrary.github.io/md_quickstart.html

#include <cuda.h>
#include <cusp_compiler_fence.hpp>

#include <cusp/hyb_matrix.h>
#include <cusp/monitor.h>
#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>

// where to perform the computation
typedef cusp::device_memory MemorySpace;

// which floating point type to use
typedef float ValueType;

int main()
{
    // Version query
    int cuda_major =  CUDA_VERSION / 1000;
    int cuda_minor = (CUDA_VERSION % 1000) / 10;
    int thrust_major = THRUST_MAJOR_VERSION;
    int thrust_minor = THRUST_MINOR_VERSION;
    int cusp_major = CUSP_MAJOR_VERSION;
    int cusp_minor = CUSP_MINOR_VERSION;

    std::cout << "CUDA   v" << cuda_major   << "." << cuda_minor   << std::endl;
    std::cout << "Thrust v" << thrust_major << "." << thrust_minor << std::endl;
    std::cout << "Cusp   v" << cusp_major   << "." << cusp_minor   << std::endl;

    // create an empty sparse matrix structure (HYB format)
    cusp::hyb_matrix<int, ValueType, MemorySpace> A;

    // create a 2d Poisson problem on a 10x10 mesh
    cusp::gallery::poisson5pt(A, 10, 10);

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<ValueType, MemorySpace> x(A.num_rows, 0);
    cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);

    // set stopping criteria:
    //  iteration_limit    = 100
    //  relative_tolerance = 1e-3
    //  absolute_tolerance = 0
    //  verbose            = true
    cusp::monitor<ValueType> monitor(b, 100, 1e-3, 0, true);

    // set preconditioner (identity)
    cusp::identity_operator<ValueType, MemorySpace> M(A.num_rows, A.num_rows);

    // solve the linear system A * x = b with the Conjugate Gradient method
    cusp::krylov::cg(A, x, b, monitor, M);

    return 0;
}
