# SpBench

Performance evaluation for sparse linear algebra libraries for CPU and GPU targets.
This evaluation attempts to measure execution time of common and most
popular sparse matrix operations: `x` and `+`. 

## Tested libraries and frameworks

| Library name          | Target platform        | Technology              | Operation `x` | Operation `+` |
| :---                  | :---                   | :---                    | :---          | :---          | 
| **cuBool**            | GPU                    | Nvidia Cuda             | yes           | yes           |
| **CUSP**              | GPU                    | Nvidia Cuda             | yes           | yes           |
| **cuSPARSE**          | GPU                    | Nvidia Cuda             | yes           | yes           |
| **clSPRASE**          | GPU                    | OpenCL                  | yes           | no            |
| **clBool**            | GPU                    | OpenCL                  | yes           | yes           | 
| **SuiteSprase**       | CPU                    | CPU Optimizations       | yes           | yes           |

Links:

- [cuBool](https://github.com/JetBrains-Research/cuBool)
- [CUSP](https://github.com/cusplibrary/cusplibrary)
- [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html)
- [clSPRASE](https://github.com/clMathLibraries/clSPARSE)
- [clBool](https://github.com/mkarpenkospb/sparse_boolean_matrix_operations)
- [SuiteSprase](https://github.com/DrTimothyAldenDavis/SuiteSparse)

## How to get and build

Get the source code and go into project directory.

```shell script
$ git clone https://github.com/EgorOrachyov/SpBench
$ cd SpBench
```

Create build directory and go into it.
The example demonstrates, how to build benchmarks in `release` mode. 

```shell script
$ mkdir build-release
$ cd build-release
```

Configure build and run actual compilation process.

```shell script
$ cmake . -DCMAKE_BUILD_TYPE=Release
$ cmake --build . --target benchmark -j `nproc`
```

After compilation process benchmark executabels 
will be stored in the build directory. The naming follows the next
pattern: `{tool-name}_{operation name}`.

## Dataset

The matrix data is selected from The SuiteSparse Matrix Collection 
(formerly the University of Florida Sparse Matrix Collection) 
[link](https://sparse.tamu.edu).

| Matrix name       | Size        | Nnz M       | Nnz M * M   | Nnz M + M * M |
| :---              | :---        | :---        | :---        | :---          |
