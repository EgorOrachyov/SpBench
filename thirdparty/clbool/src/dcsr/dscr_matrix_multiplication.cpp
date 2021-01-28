#include <numeric>
#include "dscr_matrix_multiplication.hpp"
#include "../coo/coo_matrix_addition.hpp"
#include "../coo/coo_utils.hpp"

const uint32_t BINS_NUM = 38;
const uint32_t HEAP_MERGE_BLOCK_SIZE = 32;
typedef std::vector<uint32_t> cpu_buffer;

uint32_t esc_estimation(uint32_t group) {
    switch (group) {
        case 33:
            return 64;
        case 34:
            return 128;
        case 35:
            return 256;
        case 36:
            return 512;
        default:
            throw std::runtime_error("A group should be in range 33-36!");
    }
}

auto get_to_result_matrix_single_thread(Controls &controls,
                                        uint32_t group_length) {
    cl::Program program;
    try {
        program = controls.create_program_from_file("thirdparty/clbool/to_result_matrix_single_thread.cl");
        uint32_t block_size = std::min(controls.block_size, std::min(32u, utils::ceil_to_power2(group_length)));

        std::stringstream options;
        options << "-D RUN " << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, group_length);

        cl::Kernel to_result_kernel(program, "to_result");

        using KernelType = cl::KernelFunctor<cl::Buffer, uint32_t, uint32_t,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>;

        KernelType to_result(to_result_kernel);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        return std::pair<KernelType, cl::EnqueueArgs>(to_result_kernel, eargs);
    } catch (const cl::Error &e) {
        utils::program_handler(e, program, controls.device, "get_to_result_matrix_single_thread");
    }
}

auto get_to_result_matrix_work_group(Controls &controls,
                                     uint32_t group_length) {
    cl::Program program;
    try {
        program = controls.create_program_from_file("thirdparty/clbool/to_result_matrix_work_group.cl");
        // TODO: этот размер блока можно менять и смотреть, как будет быстрее
        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D RUN " << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = block_size * group_length;

        cl::Kernel to_result_kernel(program, "to_result");

        using KernelType = cl::KernelFunctor<cl::Buffer, uint32_t,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>;

        KernelType to_result(to_result_kernel);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        return std::pair<KernelType, cl::EnqueueArgs>(to_result_kernel, eargs);
//        heap_merge(eargs, workload, a_rows_pointers, a_cols, b_rows_compressed, b_rows_pointers, a_nzr, b_nzr);
    } catch (const cl::Error &e) {
        utils::program_handler(e, program, controls.device, "get_to_result_matrix_work_group");
    }
}

void matrix_multiplication(Controls &controls,
                           matrix_dcsr &matrix_out,
                           const matrix_dcsr &a,
                           const matrix_dcsr &b) {
    if (a.nnz() == 0 || b.nnz() == 0) {
        std::cout << "empty result\n";
        return;
    }
    cl::Buffer nnz_estimation;
    count_workload(controls, nnz_estimation, a, b);

    std::vector<cpu_buffer> cpu_workload_groups(BINS_NUM, cpu_buffer());
    cpu_buffer groups_pointers(BINS_NUM + 1);
    cpu_buffer groups_length(BINS_NUM);


    cl::Buffer aux_37_group_mem_pointers;
    cl::Buffer aux_37_group_mem;

    matrix_dcsr pre;
    build_groups_and_allocate_new_matrix(controls, pre, cpu_workload_groups, nnz_estimation, a, b.nCols(),
                                         aux_37_group_mem_pointers, aux_37_group_mem);

    cl::Buffer gpu_workload_groups(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a.nzr());

    write_bins_info(controls, gpu_workload_groups, cpu_workload_groups, groups_pointers, groups_length);



    run_kernels(controls, cpu_workload_groups, groups_length, groups_pointers,
                gpu_workload_groups, nnz_estimation,
                pre, a, b,
                aux_37_group_mem_pointers, aux_37_group_mem
                );


    create_final_matrix(controls, matrix_out,
                        nnz_estimation, pre,
                        gpu_workload_groups, groups_pointers, groups_length,
                        a
                        );
}


void create_final_matrix(Controls &controls,
                         matrix_dcsr &c,
                         cl::Buffer &nnz_estimation,
                         const matrix_dcsr &pre,

                         const cl::Buffer &gpu_workload_groups,
                         const cpu_buffer &groups_pointers,
                         const cpu_buffer &groups_length,

                         const matrix_dcsr &a
                         ) {
    cl::Buffer c_rows_pointers;
    cl::Buffer c_rows_compressed;
    cl::Buffer c_cols_indices;

    uint32_t c_nnz;
    uint32_t c_nzr;

    prefix_sum(controls, nnz_estimation, c_nnz, a.nzr());

    c_cols_indices = cl::Buffer(controls.context, CL_TRUE, sizeof(uint32_t) * c_nnz);

    cl::Event write;
    controls.queue.enqueueWriteBuffer(nnz_estimation, CL_TRUE, sizeof(uint32_t) * a.nzr(), sizeof(uint32_t), &c_nnz,
                                      nullptr, &write);
    write.wait();
    cl::Event singleValueEvent;
    cl::Event wgEvent ;
    if (groups_length[1] != 0) {
        auto single_value_rows_kernel = get_to_result_matrix_single_thread(controls, groups_length[1]);
        singleValueEvent = single_value_rows_kernel.first(single_value_rows_kernel.second,
                                       gpu_workload_groups, groups_pointers[1], groups_length[1],
                                       nnz_estimation, c_cols_indices, pre.rows_pointers_gpu(), pre.cols_indices_gpu());
    }

    uint32_t second_group_length = std::accumulate(groups_length.begin() + 2, groups_length.end(), 0u);

    if (second_group_length != 0) {
        auto ordinary_rows_kernel = get_to_result_matrix_work_group(controls, second_group_length);
        wgEvent = ordinary_rows_kernel.first(ordinary_rows_kernel.second,
                                   gpu_workload_groups, groups_length[0] + groups_length[1],
                                   nnz_estimation, c_cols_indices, pre.rows_pointers_gpu(), pre.cols_indices_gpu());
    }

    if (groups_length[1] != 0) singleValueEvent.wait();
    if (second_group_length != 0) wgEvent.wait();

    cl::Buffer positions(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a.nzr());

    prepare_positions(controls, positions, nnz_estimation, a.nzr(), "prepare_for_shift_empty_rows");


    // ------------------------------------  get rid of empty rows -------------------------------

    prefix_sum(controls, positions, c_nzr, a.nzr());
    c_rows_pointers = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (c_nzr + 1));
    c_rows_compressed = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * c_nzr);
    set_positions(controls, c_rows_pointers, c_rows_compressed, nnz_estimation, a.rows_compressed_gpu(), positions,
                  c_nnz, a.nzr(), c_nzr);

    c = matrix_dcsr(c_rows_pointers, c_rows_compressed, c_cols_indices, pre.nCols(), pre.nRows(), c_nnz, c_nzr);
}

void write_bins_info(Controls &controls,
                     cl::Buffer &gpu_workload_groups,
                     const std::vector<cpu_buffer> &cpu_workload_groups,
                     cpu_buffer &groups_pointers,
                     cpu_buffer &groups_length
                     ) {

    unsigned int offset = 0;
    cl::Event end_write_buffer;
    for (uint32_t workload_group_id = 0; workload_group_id < BINS_NUM; ++workload_group_id) {
        const auto group = cpu_workload_groups[workload_group_id];
        if (group.empty()) continue;
        groups_pointers[workload_group_id] = offset;
        groups_length[workload_group_id] = group.size();
        controls.queue.enqueueWriteBuffer(gpu_workload_groups, CL_TRUE, sizeof(uint32_t) * offset,
                                          sizeof(uint32_t) * group.size(), group.data()
                                          , nullptr, &end_write_buffer);
        offset += group.size();
    }

    groups_pointers[BINS_NUM] = offset;
    end_write_buffer.wait();
}

void run_kernels(Controls &controls,
                 const std::vector<cpu_buffer> &cpu_workload_groups,
                 const cpu_buffer &groups_length,
                 const cpu_buffer &groups_pointers,

                 const cl::Buffer &gpu_workload_groups,
                 cl::Buffer &nnz_estimation,

                 const matrix_dcsr &pre,
                 const matrix_dcsr &a,
                 const matrix_dcsr &b,

                 const cl::Buffer &aux_mem_pointers,
                 cl::Buffer &aux_mem

) {
    auto heap_kernel = program<cl::Buffer, uint32_t, uint32_t,
        cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
        cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
        uint32_t>("thirdparty/clbool/heap_merge.cl")
        .set_kernel_name("heap_merge")
        .set_block_size(HEAP_MERGE_BLOCK_SIZE);

    auto copy_one_value_kernel = program<cl::Buffer, uint32_t, uint32_t,
        cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
        cl::Buffer, cl::Buffer, cl::Buffer,
        uint32_t>("thirdparty/clbool/copy_one_value.cl")
        .set_kernel_name("copy_one_value");

    auto merge_kernel = program<cl::Buffer, uint32_t, cl::Buffer,
        cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
        cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
        uint32_t>("thirdparty/clbool/merge_large_rows.cl")
        .set_kernel_name("merge_large_rows")
        .set_block_size(controls.block_size);

    auto esc_kernel = program<cl::Buffer, uint32_t, uint32_t,
            cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
            cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
            uint32_t>("thirdparty/clbool/bitonic_esc.cl")
            .set_kernel_name("bitonic_esc")
            .set_block_size(controls.block_size);


    std::vector<cl::Event> events;
    for (uint32_t workload_group_id = 1; workload_group_id < BINS_NUM; ++workload_group_id) {
        const auto group = cpu_workload_groups[workload_group_id];
        if (group.empty()) continue;



        if (workload_group_id == 1) {
//            std::cout << "first group!\n";
            copy_one_value_kernel.set_needed_work_size(groups_length[workload_group_id])
            .set_block_size(std::min(controls.block_size,
                                     std::max(32u, utils::ceil_to_power2(groups_length[workload_group_id]))));
            events.push_back(
                    copy_one_value_kernel.run(controls,
                             gpu_workload_groups, groups_pointers[workload_group_id], groups_length[workload_group_id],
                             pre.rows_pointers_gpu(), pre.cols_indices_gpu(),
                             a.rows_pointers_gpu(), a.cols_indices_gpu(),
                             b.rows_pointers_gpu(), b.rows_compressed_gpu(), b.cols_indices_gpu(),
                             b.nzr()
                             )
            );
            continue;
        }



        if (workload_group_id < 33 ) {
//            std::cout << "2 - 32!: " << workload_group_id << "\n";
            heap_kernel.set_needed_work_size(groups_length[workload_group_id])
                        .add_option("NNZ_ESTIMATION", workload_group_id);
            events.push_back(heap_kernel.run(controls, gpu_workload_groups, groups_pointers[workload_group_id], groups_length[workload_group_id],
                             pre.rows_pointers_gpu(), pre.cols_indices_gpu(),
                             nnz_estimation,
                             a.rows_pointers_gpu(), a.cols_indices_gpu(),
                             b.rows_pointers_gpu(), b.rows_compressed_gpu(), b.cols_indices_gpu(),
                             b.nzr()));

            continue;
        }



        if (workload_group_id < 37 ) {
//            std::cout << "33 - 36!\n";
            uint32_t block_size = std::max(32u, esc_estimation(workload_group_id) / 2);
            esc_kernel.add_option("NNZ_ESTIMATION", esc_estimation(workload_group_id))
            .set_block_size(block_size)
            .set_needed_work_size(block_size * groups_length[workload_group_id]);
            events.push_back(esc_kernel.run(
                    controls,
                    gpu_workload_groups, groups_pointers[workload_group_id], groups_length[workload_group_id],
                    pre.rows_pointers_gpu(), pre.cols_indices_gpu(),
                    nnz_estimation,
                    a.rows_pointers_gpu(), a.cols_indices_gpu(),
                    b.rows_pointers_gpu(), b.rows_compressed_gpu(), b.cols_indices_gpu(),
                    b.nzr()
            ));
            continue;
        }


//        std::cout << "37!\n";
        merge_kernel.set_needed_work_size(groups_length[workload_group_id] * controls.block_size);
        events.push_back(merge_kernel.run(controls,
                            gpu_workload_groups, groups_pointers[workload_group_id],
                            aux_mem_pointers, aux_mem,
                            pre.rows_pointers_gpu(), pre.cols_indices_gpu(),
                            nnz_estimation,
                            a.rows_pointers_gpu(), a.cols_indices_gpu(),
                            b.rows_pointers_gpu(), b.rows_compressed_gpu(), b.cols_indices_gpu(),
                            b.nzr()
                            ));

        for (const auto &event: events) {
            event.wait();
        }
    }
}

void build_groups_and_allocate_new_matrix(Controls& controls,
                                          matrix_dcsr &pre,
                                          std::vector<cpu_buffer>& cpu_workload_groups,
                                          cl::Buffer& nnz_estimation,
                                          const matrix_dcsr &a,
                                          uint32_t b_cols,

                                          cl::Buffer &aux_pointers,
                                          cl::Buffer &aux_mem
                                          ) {

    cpu_buffer aux_pointers_cpu;
    uint32_t aux = 0;

    cpu_buffer cpu_workload(a.nzr());
    cl::Event event;
    controls.queue.enqueueReadBuffer(nnz_estimation, CL_TRUE, 0, sizeof(uint32_t) * a.nzr(), cpu_workload.data(),
                                     nullptr, &event);
    event.wait();

    uint32_t pre_nnz = 0;
    cpu_buffer rows_pointers_cpu(a.nzr() + 1);

    pre_nnz = 0;
    for (uint32_t i = 0; i < a.nzr(); ++i) {

        uint32_t current_workload = cpu_workload[i];
        uint32_t group = get_group(current_workload);
        cpu_workload_groups[group].push_back(i);
        rows_pointers_cpu[i] = pre_nnz;

        // TODO: добавить переаллокацию
        pre_nnz += current_workload;
        if (group == 37) {
            aux_pointers_cpu.push_back(aux);
            aux += current_workload;
        }
    }
    if (pre_nnz == 0) {
        std::cout << "empty result\n";
        return;
    }
    aux_pointers_cpu.push_back(aux);
    rows_pointers_cpu[a.nzr()] = pre_nnz;

    cl::Buffer pre_rows_pointers = cl::Buffer(controls.queue, rows_pointers_cpu.begin(), rows_pointers_cpu.end(), false);
    cl::Buffer pre_cols_indices_gpu = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * pre_nnz);

    if (aux != 0) {
        aux_pointers = cl::Buffer(controls.queue, aux_pointers_cpu.begin(), aux_pointers_cpu.end(),
                                  true);
        aux_mem = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * aux);
    }


    pre = matrix_dcsr(pre_rows_pointers, a.rows_compressed_gpu(), pre_cols_indices_gpu,
                      a.nRows(), b_cols, pre_nnz, a.nzr());
}


uint32_t get_group(uint32_t size) {
    if (size < 33) return size;
    if (size < 65) return 33;
    if (size < 129) return 34;
    if (size < 257) return 35;
    if (size < 513) return 36;
    return 37;
}


void count_workload(Controls &controls,
                    cl::Buffer &nnz_estimation_out,
                    const matrix_dcsr &a,
                    const matrix_dcsr &b) {

    cl::Program program;
    try {
        cl::Buffer nnz_estimation(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (a.nzr() + 1));
        program = controls.create_program_from_file("thirdparty/clbool/count_workload.cl");
        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D RUN " << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, a.nzr());


        cl::Kernel coo_count_workload_kernel(program, "count_workload");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, uint32_t, uint32_t> coo_count_workload(
                coo_count_workload_kernel);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        auto event = coo_count_workload(eargs, nnz_estimation, a.rows_pointers_gpu(), a.cols_indices_gpu(),
                           b.rows_compressed_gpu(), b.rows_pointers_gpu(), a.nzr(), b.nzr());
        event.wait();

        nnz_estimation_out = std::move(nnz_estimation);

    } catch (const cl::Error &e) {
        utils::program_handler(e, program, controls.device, "count_workload");
    }
}


void prepare_positions(Controls &controls,
                       cl::Buffer &positions,
                       const cl::Buffer &array,
                       uint32_t size,
                       const std::string &program_name
) {
    cl::Program program;
    try {
        program = controls.create_program_from_file("thirdparty/clbool/prepare_positions.cl");
        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D RUN " << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, size);

        cl::Kernel coo_prepare_positions_kernel(program, program_name.c_str());
        cl::KernelFunctor<cl::Buffer, cl::Buffer, uint32_t> coo_prepare_positions(
                coo_prepare_positions_kernel);
        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        auto event = coo_prepare_positions(eargs, positions, array, size);
        event.wait();

    } catch (const cl::Error &e) {
        utils::program_handler(e, program, controls.device, "prepare_positions");
    }
}


void set_positions(Controls &controls,
                   cl::Buffer &c_rows_pointers,
                   cl::Buffer &c_rows_compressed,
                   const cl::Buffer &nnz_estimation,
                   const cl::Buffer &a_rows_compressed,
                   const cl::Buffer &positions,
                   uint32_t c_nnz,
                   uint32_t old_nzr,
                   uint32_t c_nzr
) {

    cl::Program program;
    try {
        program = controls.create_program_from_file("thirdparty/clbool/set_positions.cl");
        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D RUN " << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        cl::Kernel set_positions_kernel(program, "set_positions_pointers_and_rows");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                            unsigned int, unsigned int, unsigned int> set_positions(
                set_positions_kernel);

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, old_nzr);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        auto event = set_positions(eargs, c_rows_pointers, c_rows_compressed,
                      nnz_estimation, a_rows_compressed, positions,
                      c_nnz, old_nzr, c_nzr  );
        event.wait();

    } catch (const cl::Error &e) {
        utils::program_handler(e, program, controls.device, "set_positions");
    }
}