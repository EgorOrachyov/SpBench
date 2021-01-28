#ifndef RUN

#include "clion_defines.cl"
#define GROUP_SIZE 256

#endif


__kernel void set_positions(__global unsigned int* newRows,
                            __global unsigned int* newCols,
                            __global const unsigned int* rows,
                            __global const unsigned int* cols,
                            __global const unsigned int* positions,
                            unsigned int size
                            ) {

    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int global_id = get_global_id(0);

    if (global_id == size - 1 && positions[global_id] != size) {
        newRows[positions[global_id]] = rows[global_id];
        newCols[positions[global_id]] = cols[global_id];
        return;
    }

    if (global_id >= size) return;

    if (positions[global_id] != positions[global_id + 1]) {
        newRows[positions[global_id]] = rows[global_id];
        newCols[positions[global_id]] = cols[global_id];
    }
}


__kernel void set_positions_pointers_and_rows(__global uint* newRowsPosition,
                                              __global uint* newRows,
                                              __global const uint* rowsPositions,
                                              __global const uint* rows,
                                              __global const uint* positions,
                                              uint nnz, // old nzr
                                              uint old_nzr,
                                              uint new_nzr
) {
    unsigned int global_id = get_global_id(0);

    if (global_id >= old_nzr) return;

    if (global_id == old_nzr - 1) {
        if (positions[global_id] != old_nzr) {
            newRowsPosition[positions[global_id]] = rowsPositions[global_id];
            newRows[positions[global_id]] = rows[global_id];
        }
        newRowsPosition[new_nzr] = nnz;
        return;
    }

    if (positions[global_id] != positions[global_id + 1]) {
        newRowsPosition[positions[global_id]] = rowsPositions[global_id];
        newRows[positions[global_id]] = rows[global_id];
    }
}


__kernel void set_positions_rows(__global uint* rows_pointers,
                                 __global uint* rows_compressed,
                                 __global const uint* rows,
                                 __global const uint* positions,
                                 uint size,
                                 uint nzr
) {
    unsigned int global_id = get_global_id(0);

    if (global_id == size - 1) {
        if (positions[global_id] != size) {
            rows_pointers[positions[global_id]] = global_id;
            rows_compressed[positions[global_id]] = rows[global_id];
        }
        rows_pointers[nzr] = size;
        return;
    }

    if (global_id >= size) return;

    if (positions[global_id] != positions[global_id + 1]) {
        rows_pointers[positions[global_id]] = global_id;
        rows_compressed[positions[global_id]] = rows[global_id];
    }
}