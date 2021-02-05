# SpBench

Performance evaluation for sparse linear algebra libraries for CPU and GPU targets.
This evaluation attempts to measure execution time of common and most
popular sparse matrix operations `x` and `+` over boolean semiring. 

## Tested libraries and frameworks

| Library name                                                                    | Target platform | Technology   | Operation `x` | Operation `+` |
| :---                                                                            | :---            | :---         | :---          | :---          | 
| [cuBool     ](https://github.com/JetBrains-Research/cuBool)                     | GPU             | Nvidia Cuda  | yes           | yes           |
| [CUSP       ](https://github.com/cusplibrary/cusplibrary)                       | GPU             | Nvidia Cuda  | yes           | yes           |
| [cuSPARSE   ](https://docs.nvidia.com/cuda/cusparse/index.html)                 | GPU             | Nvidia Cuda  | yes           | yes           |
| [clSPRASE   ](https://github.com/clMathLibraries/clSPARSE)                      | GPU             | OpenCL       | yes           | no            |
| [clBool     ](https://github.com/mkarpenkospb/sparse_boolean_matrix_operations) | GPU             | OpenCL       | yes           | yes           | 
| [SuiteSprase](https://github.com/DrTimothyAldenDavis/SuiteSparse)               | CPU             | CPU          | yes           | yes           |

## How to get and build

Get the source code and go into project directory.

```shell script
$ git clone https://github.com/EgorOrachyov/SpBench
$ cd SpBench
$ git submodule update --init --recursive
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

After compilation process benchmark executables 
will be stored in the build directory. The naming follows the next
pattern: `{tool-name}_{operation name}`.

## Dataset

The matrix data is selected from The SuiteSparse Matrix Collection 
(formerly the University of Florida Sparse Matrix Collection) 
[link](https://sparse.tamu.edu).

| Matrix name             | # Rows      | Nnz M     | Nnz M * M   | Nnz M + M * M |
| :---                    | :---        | :---      | :---        | :---          |
| DIMACS10/wing		      | 62,032      | 243088    | 714200      | 917178        |
| DIMACS10/luxembourg_osm | 114,599     | 239332    | 393261      | 632185        |
| SNAP/amazon0312         | 400,727     | 3200440   | 14390544    | 14968909      |
| LAW/amazon-2008         | 735,323     | 5158388   | 25366745    | 26402678      |
| SNAP/web-Google         | 916,428     | 5105039   | 29710164    | 30811855      |
| SNAP/roadNet-PA         | 1,090,920   | 3083796   | 7238920     | 9931528       |
| SNAP/roadNet-TX	      | 1,393,383   | 3843320   | 8903897     | 12264987      |
| DIMACS10/belgium_osm    | 1,441,295   | 3099940   | 5323073     | 8408599       |
| SNAP/roadNet-CA	      | 1,971,281   | 5533214   | 12908450    | 17743342      |
| SNAP/wiki-Talk	      | 2,394,385   | 5021410   | 1155524188  | 1158853825    |