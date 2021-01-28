#ifndef RUN

#include "clion_defines.cl"
#define GROUP_SIZE 256

#endif

__kernel void dscr_to_coo(__global const uint* a_rows_pointers,
                          __global const uint* a_rows_compressed,
                          __global uint* c_rows_indices
                          ) {
    uint group_id = get_group_id(0);
    uint local_id = get_local_id(0);
    uint row_start = a_rows_pointers[group_id];
    uint row = a_rows_compressed[group_id];
    uint row_end = a_rows_pointers[group_id + 1];
    uint row_length = row_end - row_start;
    uint steps = (row_length + GROUP_SIZE - 1) / GROUP_SIZE;
    for (uint i = 0; i < steps; ++i) {
        uint elem_id = row_start + i * GROUP_SIZE + local_id;
        if (elem_id < row_end) {
            c_rows_indices[elem_id] = row;
        }
    }
}
