# SpBench

Benchmarks for sparse linear algebra libraries for GPGPU.

## How to build

The following code assumes, that you are currently in the root directory
of the project (in the SpBench folder).

Firstly, create build directory and go into it.
The example demonstrates, how to build benchmarks in `debug` mode. 

```shell script
$ mkdir build-debug
$ cd build-debug
```

Configure build and run actual compilation.
Number of threads to run compilation is chosen randomly (replace `4` with desired values).

```shell script
$ cmake . -DCMAKE_BUILD_TYPE=Debug
$ cmake --build . --target benchmark -j 4
```

## Dataset

| Matrix file name  | Type       | Size        | Nnz         | Nnz M += M * M |
| :---              | :---       | :---        | :---        | :---           |
