# SpBench - Sparse Boolean Linear Algebra Libraries Benchmark 

Performance evaluation for sparse linear algebra libraries for CPU and GPU targets.
This evaluation attempts to measure execution time of common and most
popular sparse matrix operations `x` and `+` over boolean semiring. 

**Note**: some of these tested libraries does not provide extra optimizations
for the boolean-values operations. Thus, these operations are imitated by 
the provided primitives, where non-zero (floating point type) values are interpreted as true.

## Tested libraries and frameworks

| Library name                                                                    | Target platform | Technology   | Operation `x` | Operation `+` |
|---                                                                              |---              |---           |---            |---            | 
| [cuBool     ](https://github.com/JetBrains-Research/cuBool)                     | GPU             | Nvidia Cuda  | yes           | yes           |
| [CUSP       ](https://github.com/cusplibrary/cusplibrary)                       | GPU             | Nvidia Cuda  | yes           | yes           |
| [cuSPARSE   ](https://docs.nvidia.com/cuda/cusparse/index.html)                 | GPU             | Nvidia Cuda  | yes           | yes           |
| [clSPARSE   ](https://github.com/clMathLibraries/clSPARSE)                      | GPU             | OpenCL       | yes           | no            |
| [clBool     ](https://github.com/mkarpenkospb/sparse_boolean_matrix_operations) | GPU             | OpenCL       | yes           | yes           | 
| [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse)               | CPU             | CPU          | yes           | yes           |

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
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ cmake --build . --target all -j `nproc`
```

After compilation process benchmark executables 
will be stored in the build directory. The naming follows the next
pattern: `{tool-name}_{operation name}`.

## Dataset

The matrix data is selected from The SuiteSparse Matrix Collection 
(formerly the University of Florida Sparse Matrix Collection) 
[link](https://sparse.tamu.edu).

| №  | Matrix name              | # Rows      | Nnz M       | Nnz M^2     | Nnz M + M^2   |
|--- |---                       |---          |---          |---          |---            |
| 0  | DIMACS10/wing		    | 62,032      | 243,088     | 714,200     | 917,178       |
| 1  | DIMACS10/luxembourg_osm  | 114,599     | 239,332     | 393,261     | 632,185       |
| 2  | SNAP/amazon0312          | 400,727     | 3,200,440   | 14,390,544  | 14,968,909    |
| 3  | LAW/amazon-2008          | 735,323     | 5,158,388   | 25,366,745  | 26,402,678    |
| 4  | SNAP/web-Google          | 916,428     | 5,105,039   | 29,710,164  | 30,811,855    |
| 5  | SNAP/roadNet-PA          | 1,090,920   | 3,083,796   | 7,238,920   | 9,931,528     |
| 6  | SNAP/roadNet-TX	        | 1,393,383   | 3,843,320   | 8,903,897   | 12,264,987    |
| 7  | DIMACS10/belgium_osm     | 1,441,295   | 3,099,940   | 5,323,073   | 8,408,599     |
| 8  | SNAP/roadNet-CA	        | 1,971,281   | 5,533,214   | 12,908,450  | 17,743,342    |
| 9  | DIMACS10/netherlands_osm | 2,216,688   | 4,882,476   | 8,755,758   | 13,626,132    |

## Results

For performance evaluation, PC with Ubuntu 20.04 installed was used. 
It had Intel core i7-4790 CPU, 3.6GHz, DDR4 32Gb RAM and GeForce GTX 1070 GPU with 8Gb VRAM.
Only the execution time of the operations themselves was measured.
The actual data were assumed to be loaded into the VRAM or RAM respectively in the appropriate format, 
required for the target tested framework.

**Matrix-matrix multiplication evaluation results.**  
Time in milliseconds, Memory in megabytes.

| №  | cuBool     | CUSP       | cuSPARSE   | clBool     | clSPARSE   | SuiteSparse |
|--- |---         |---         |---         |---         |---         |---          |
| M  | Time&Mem   | Time&Mem   | Time&Mem   | Time&Mem   | Time&Mem   | Time        |
| 0  | 2.2 215    | 5.8 125    | 20.2 155   | 60.5 95    | 127.9 109  | 10.0        |
| 1  | 2.9 213    | 4.1 111    | 1.7 149    | 16.0 91    | 10.8 99    | 2.5         |
| 2  | 24.6 215   | 110.3 897  | 411.6 301  | 97.6 279   | 65.7 459   | 238.2       |
| 3  | 38.9 341   | 173.5 1409 | 182.8 407  | 110.1 401  | 104.4 701  | 339.4       |
| 4  | 50.1 341   | 2408 1717  | 4756.4 439 | 277.8 491  | 409.2 1085 | 644.6       |
| 5  | 21.7 215   | 43.3 481   | 37.6 247   | 45.6 203   | 85.5 283   | 63.0        |
| 6  | 26.6 215   | 52.1 581   | 46.8 271   | 55.8 229   | 107.4 329  | 74.9        |
| 7  | 26.9 215   | 33.8 397   | 26.7 235   | 68.6 183   | 104.9 259  | 57.8        |
| 8  | 37.6 215   | 76.4 771   | 67.2 325   | 77.7 279   | 151.5 433  | 110.5       |
| 9  | 40.4 215   | 51.9 585   | 51.1 291   | 78.1 251   | 158.2 361  | 93.0        |

**Element-wise matrix-matrix addition evaluation results.**  
Time in milliseconds, Memory in megabytes.

| №  | cuBool     | CUSP       | cuSPARSE   | clBool     | clSPARSE   | SuiteSparse |
|--- |---         |---         |---         |---         |---         |---          |
| M  | Time&Mem   | Time&Mem   | Time&Mem   | Time&Mem   | Time&Mem   | Time        |
| 0  | 1.1 97     | 1.5 103    | 2.4 163    | 22.8 105   | - -        | 4.0         |
| 1  | 1.7 103    | 1.1 103    | 0.9 159    | 4.5 103    | - -        | 1.5         |
| 2  | 9.4 237    | 16.5 455   | 24.1 405   | 84.0 543   | - -        | 35.1        |
| 3  | 16.2 347   | 29.1 723   | 23.6 595   | 163.3 877  | - -        | 61.2        |
| 4  | 18.7 379   | 32.3 815   | 88.7 659   | 176.7 989  | - -        | 72.5        | 
| 5  | 15.4 207   | 11.6 329   | 11.8 317   | 66.4 359   | - -        | 34.0        |
| 6  | 19.3 231   | 14.0 385   | 14.7 357   | 73.2 429   | - -        | 41.8        |
| 7  | 19.6 197   | 10.2 303   | 10.3 297   | 61.8 321   | - -        | 26.8        |
| 8  | 27.1 289   | 19.5 513   | 20.3 447   | 135.3 579  | - -        | 61.4        |
| 9  | 33.1 263   | 15.2 423   | 18.2 385   | 76.1 457   | - -        | 47.0        | 

## License

This project is licensed under MIT license. Full license text can be found in the license 
[file](https://github.com/EgorOrachyov/SpBench/blob/master/LICENSE.md). 
For the detailed license information of the dependencies refer to these third party projects official web pages.
