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

| Matrix name             | # Rows      | Nnz M       | Nnz M * M   | Nnz M + M * M |
| :---                    | :---        | :---        | :---        | :---          |
| DIMACS10/wing		      | 62,032      | 243,088     | 714,200     | 917,178       |
| DIMACS10/luxembourg_osm | 114,599     | 239,332     | 393,261     | 632,185       |
| SNAP/amazon0312         | 400,727     | 3,200,440   | 14,390,544  | 14,968,909    |
| LAW/amazon-2008         | 735,323     | 5,158,388   | 25,366,745  | 26,402,678    |
| SNAP/web-Google         | 916,428     | 5,105,039   | 29,710,164  | 30,811,855    |
| SNAP/roadNet-PA         | 1,090,920   | 3,083,796   | 7,238,920   | 9,931,528     |
| SNAP/roadNet-TX	      | 1,393,383   | 3,843,320   | 8,903,897   | 12,264,987    |
| DIMACS10/belgium_osm    | 1,441,295   | 3,099,940   | 5,323,073   | 8,408,599     |
| SNAP/roadNet-CA	      | 1,971,281   | 5,533,214   | 12,908,450  | 17,743,342    |
| DIMACS10/netherlands_osm| 2,216,688   | 4,882,476   | 8,755,758   | 136,261,32    |