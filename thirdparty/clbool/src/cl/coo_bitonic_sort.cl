#ifndef RUN

#include "clion_defines.cl"
#define GROUP_SIZE 256

#endif


bool is_greater_local(__local const unsigned int* rows,
                      __local const unsigned int* cols,
                      unsigned int line_id,
                      unsigned int twin_id) {

    return (rows[line_id] > rows[twin_id]) ||
            ((rows[line_id] == rows[twin_id]) && (cols[line_id] > cols[twin_id]));
}

bool is_greater_global(__global const unsigned int* rows,
                       __global const unsigned int* cols,
                        unsigned int line_id,
                        unsigned int twin_id) {

    return (rows[line_id] > rows[twin_id]) ||
           ((rows[line_id] == rows[twin_id]) && (cols[line_id] > cols[twin_id]));
}

__kernel void local_bitonic_begin(__global unsigned int* rows,
                                  __global unsigned int* cols,
                                  unsigned int n) {

    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int global_id = get_global_id(0);
    unsigned int work_size = GROUP_SIZE * 2;
//    if (global_id == 0) {
//        printf("GROUP_SIZE: %d\n", GROUP_SIZE);
//        printf("get_global_size: %d\n", get_global_size(0));
//        printf("get_num_groups: %d\n", get_num_groups(0));
//    }
    __local unsigned int local_rows[GROUP_SIZE * 2];
    __local unsigned int local_cols[GROUP_SIZE * 2];

    unsigned int tmp_row = 0;
    unsigned int tmp_col = 0;

    unsigned int read_idx = work_size * group_id + local_id;

    local_cols[local_id] = read_idx < n ? cols[read_idx] : 0;
    local_rows[local_id] = read_idx < n ? rows[read_idx] : 0;

    read_idx += GROUP_SIZE;

    local_cols[local_id + GROUP_SIZE] = read_idx < n ? cols[read_idx] : 0;
    local_rows[local_id + GROUP_SIZE] = read_idx < n ? rows[read_idx] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned int last_group = n / work_size;
    unsigned int real_array_size = (group_id == last_group && (n % work_size)) ? (n % work_size) : work_size;
    unsigned int outer = pow(2, ceil(log2((float) real_array_size)));

    unsigned int segment_length = 2;
    while (outer != 1) {
        unsigned int local_line_id = local_id % (segment_length / 2);
        unsigned int local_twin_id = segment_length - local_line_id - 1;
        unsigned int group_line_id = local_id / (segment_length / 2);
        unsigned int line_id = segment_length * group_line_id + local_line_id;
        unsigned int twin_id = segment_length * group_line_id + local_twin_id;

        if (twin_id < real_array_size && is_greater_local(local_rows, local_cols, line_id, twin_id)) {
            tmp_row = local_rows[line_id];
            tmp_col = local_cols[line_id];

            local_rows[line_id] = local_rows[twin_id];
            local_cols[line_id] = local_cols[twin_id];

            local_rows[twin_id] = tmp_row;
            local_cols[twin_id] = tmp_col;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int j = segment_length / 2; j > 1; j >>= 1) {
            local_line_id = local_id % (j / 2);
            local_twin_id = local_line_id + (j / 2);
            group_line_id = local_id / (j / 2);
            line_id = j * group_line_id + local_line_id;
            twin_id = j * group_line_id + local_twin_id;
            if (twin_id < real_array_size && is_greater_local(local_rows, local_cols, line_id, twin_id)) {
                tmp_row = local_rows[line_id];
                tmp_col = local_cols[line_id];

                local_rows[line_id] = local_rows[twin_id];
                local_cols[line_id] = local_cols[twin_id];

                local_rows[twin_id] = tmp_row;
                local_cols[twin_id] = tmp_col;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        outer >>= 1;
        segment_length <<= 1;
    }

    unsigned int glob_id = get_global_id(0);

//    if (glob_id == 0) {
//        int stop = GROUP_SIZE * 2;
//        printf("local_id: %d, group_id: %d, stop: %d \n", local_id, group_id, stop);
//
//        for (int i = 0; i < 2 * stop; ++i){
//            printf("%d: (%d, %d) ", i,  local_rows[i], local_cols[i]);
//        }
//        printf("\n");
//    }

    unsigned int write_idx = work_size * group_id + local_id;
    if (write_idx < n) {
        cols[write_idx] = local_cols[local_id];
        rows[write_idx] = local_rows[local_id];
    }

    write_idx += GROUP_SIZE;
    if (write_idx < n) {
        cols[write_idx] = local_cols[local_id + GROUP_SIZE];
        rows[write_idx] = local_rows[local_id + GROUP_SIZE];
    }
}


__kernel void bitonic_global_step(__global unsigned int* rows,
                                  __global unsigned int* cols,
                                  unsigned int segment_length,
                                  unsigned int mirror,
                                  unsigned int n)
{
    unsigned int global_id = get_global_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int local_line_id = global_id % (segment_length / 2);
    unsigned int local_twin_id = mirror ? segment_length - local_line_id - 1 : local_line_id + (segment_length / 2);
    unsigned int group_line_id = global_id / (segment_length / 2);
    unsigned int line_id = segment_length * group_line_id + local_line_id;
    unsigned int twin_id = segment_length * group_line_id + local_twin_id;

    unsigned int tmp_row = 0;
    unsigned int tmp_col = 0;
//    if (group_id  == 359) {
//        printf("fine, twin_id: %d\n", twin_id, n);
//    }
    if ((twin_id < n) && is_greater_global(rows, cols, line_id, twin_id)) {

        tmp_row = rows[line_id];
        tmp_col = cols[line_id];

        rows[line_id] = rows[twin_id];
        cols[line_id] = cols[twin_id];

        rows[twin_id] = tmp_row;
        cols[twin_id] = tmp_col;
    }
}

__kernel void bitonic_local_endings(__global unsigned int* rows,
                                    __global unsigned int* cols,
                                    unsigned int n)
{
    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int work_size = GROUP_SIZE * 2;

    __local unsigned int local_rows[GROUP_SIZE * 2];
    __local unsigned int local_cols[GROUP_SIZE * 2];

    unsigned int tmp_row = 0;
    unsigned int tmp_col = 0;

    unsigned int read_idx = work_size * group_id + local_id;

    local_cols[local_id] = read_idx < n ? cols[read_idx] : 0;
    local_rows[local_id] = read_idx < n ? rows[read_idx] : 0;

    read_idx += GROUP_SIZE;

    local_cols[local_id + GROUP_SIZE] = read_idx < n ? cols[read_idx] : 0;
    local_rows[local_id + GROUP_SIZE] = read_idx < n ? rows[read_idx] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int segment_length = work_size;
    unsigned int last_group = n / work_size;
    unsigned int real_array_size = (group_id == last_group && (n % work_size)) ? (n % work_size) : work_size;

    for (unsigned int j = segment_length; j > 1; j >>= 1) {
        unsigned int local_line_id = local_id % (j / 2);
        unsigned int local_twin_id = local_line_id + (j / 2);
        unsigned int group_line_id = local_id / (j / 2);
        unsigned int line_id = j * group_line_id + local_line_id;
        unsigned int twin_id = j * group_line_id + local_twin_id;

        if (twin_id < real_array_size && is_greater_local(local_rows, local_cols, line_id, twin_id)) {
            tmp_row = local_rows[line_id];
            tmp_col = local_cols[line_id];

            local_rows[line_id] = local_rows[twin_id];
            local_cols[line_id] = local_cols[twin_id];

            local_rows[twin_id] = tmp_row;
            local_cols[twin_id] = tmp_col;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    unsigned int write_idx = work_size * group_id + local_id;
    if (write_idx < n) {
        cols[write_idx] = local_cols[local_id];
        rows[write_idx] = local_rows[local_id];
    }

    write_idx += GROUP_SIZE;
    if (write_idx < n) {
        cols[write_idx] = local_cols[local_id + GROUP_SIZE];
        rows[write_idx] = local_rows[local_id + GROUP_SIZE];
    }
}



