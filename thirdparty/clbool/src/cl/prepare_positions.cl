#ifndef RUN

#include "clion_defines.cl"
#define GROUP_SIZE 256

#endif

__kernel void prepare_array_for_positions(__global unsigned int* result,
                                          __global const unsigned int* rows,
                                          __global const unsigned int* cols,
                                          unsigned int size
                                          ) {

    unsigned int global_id = get_global_id(0);

    if (global_id >= size) {
        return;
    }

    // if on global_id - 1 we have the same value, we write 0 in result, otherwise 1
    result[global_id] = global_id == 0 ? 1 :
                        (cols[global_id] == cols[global_id - 1]) && (rows[global_id] == rows[global_id - 1]) ?
                        0 : 1;
}


__kernel void prepare_array_for_rows_positions(__global unsigned int* result,
                                               __global const unsigned int* rows,
                                               unsigned int size
) {

    unsigned int global_id = get_global_id(0);

    if (global_id >= size) {
        return;
    }

    // if on global_id - 1 we have the same value, we write 0 in result, otherwise 1
    result[global_id] = global_id == 0 ? 1 : (rows[global_id] == rows[global_id - 1]) ?
                        0 : 1;

}



__kernel void prepare_array_for_shift(__global unsigned int* result,
                                      __global const unsigned int* rows,
                                      __global const unsigned int* cols,
                                      unsigned int size
                                      ) {

    unsigned int global_id = get_global_id(0);

    if (global_id >= size) {
        return;
    }

    // if on global_id - 1 we have the same value, we write 1 in result,
    // otherwise 0
    result[global_id] = global_id == 0 ? global_id :
                        (cols[global_id] == cols[global_id - 1]) && (rows[global_id] == rows[global_id - 1]) ?
                        1 : 0;
}


__kernel void prepare_for_shift_empty_rows(__global unsigned int* result,
                                           __global const unsigned int* nnz_estimation,
                                           unsigned int size
) {

    unsigned int global_id = get_global_id(0);

    if (global_id >= size) {
        return;
    }

    result[global_id] = nnz_estimation[global_id] == nnz_estimation[global_id + 1]  ? 0 : 1;
}