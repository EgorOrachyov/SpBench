#ifndef RUN

#include "clion_defines.cl"
#define GROUP_SIZE 256

#endif


// TODO: maybe split task to call less threads

__kernel void kronecker(__global unsigned int* rowsRes,
                        __global unsigned int* colsRes,
                        __global const unsigned int* rowsA,
                        __global const unsigned int* colsA,
                        __global const unsigned int* rowsB,
                        __global const unsigned int* colsB,

                        unsigned int nnzB,
                        unsigned int nRowsB,
                        unsigned int nColsB
                        ) {
    unsigned int global_id = get_global_id(0);

    unsigned int block_id = global_id / nnzB;
    unsigned int elem_id = global_id % nnzB;

    unsigned int rowA = rowsA[block_id];
    unsigned int colA = colsA[block_id];

    unsigned int rowB = rowsB[elem_id];
    unsigned int colB = colsB[elem_id];

    rowsRes[global_id] = nRowsB * rowA + rowB;
    colsRes[global_id] = nColsB * colA + colB;
}



