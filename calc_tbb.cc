#include <tbb/global_control.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <tbb/parallel_for.h>
#include "calc.h"
#include "spmv.h"

namespace FastSPICEmv {

#ifdef ALIGN_Y
inline intptr_t misalign(double* y) {  return (- (reinterpret_cast<intptr_t>(y) >> 3)) & 7; }
inline intptr_t misalign(TaskMatrixInfoA* matrix, int first_row) { return misalign(matrix->Id + first_row); }
inline intptr_t misalign(TaskMatrixInfoB* matrix, int first_row) { return misalign(matrix->IG + first_row); }
#endif

template<typename TaskMatrixInfo>
struct Parameters;

template<>
struct Parameters<TaskMatrixInfoA> {
    static constexpr int SEP = 128;
    static constexpr int SIMD_MIN_M = 32;
    static constexpr int LONGROW_MIN_NNZ = 1000;
    static constexpr int LONGROW_AVG_NNZ = LONGROW_MIN_NNZ / SEP;
};

template<>
struct Parameters<TaskMatrixInfoB> {
    static constexpr int SEP = 128;
    static constexpr int SIMD_MIN_M = 32;
    static constexpr int LONGROW_MIN_NNZ = 1000;
    static constexpr int LONGROW_AVG_NNZ = LONGROW_MIN_NNZ / SEP;
};

/* overload task A and task B */
inline void kernel_0(TaskMatrixInfoA* ptr, int start, int len) {
    taskA(ptr->rowArray + start, ptr->rowOffset, len, ptr->columnIndice,
        ptr->S, ptr->valueNormalMatrix, ptr->Id);
}

inline void kernel_1(TaskMatrixInfoA* ptr, int start, int len) {
    spmv_rowwise_simd_taskA(ptr->rowArray[start], ptr->rowArray[start] + len,
        ptr->rowOffset, ptr->columnIndice, ptr->valueNormalMatrix, ptr->S, ptr->Id);
}

inline void kernel_2(TaskMatrixInfoA* ptr, int start, int len) {
    spmv_row_1_taskA(ptr->rowArray[start], ptr->rowArray[start] + len,
        ptr->rowOffset, ptr->columnIndice, ptr->valueNormalMatrix, ptr->S, ptr->Id);
}

inline void kernel_0(TaskMatrixInfoB* ptr, int start, int len) {
    taskB(ptr->valueSpiceMatrix, ptr->rowOffset, ptr->columnIndice,
        ptr->A, ptr->S, ptr->R, ptr->H, ptr->D, ptr->IC,
        ptr->IG, ptr->alpha, ptr->rowArray + start, len);
}

inline void kernel_1(TaskMatrixInfoB* ptr, int start, int len) {
    spmv_rowwise_simd_taskB(ptr->rowArray[start],
        ptr->rowArray[start] + len,
        ptr->rowOffset, ptr->columnIndice, ptr->valueSpiceMatrix, ptr->S,
        ptr->D, ptr->IG, ptr->IC, ptr->R, ptr->H, ptr->A, ptr->alpha);
}

inline void kernel_2(TaskMatrixInfoB* ptr, int start, int len) {
    spmv_row_1_taskB(ptr->rowArray[start],
        ptr->rowArray[start] + len,
        ptr->rowOffset, ptr->columnIndice, ptr->valueSpiceMatrix, ptr->S,
        ptr->D, ptr->IG, ptr->IC, ptr->R, ptr->H, ptr->A, ptr->alpha);
}

template <typename TaskMatrixInfo>
void calc(TaskMatrixInfo* ptr) {
    tbb::task_group tasks;
    const int row_size = ptr->rowArraySize;
    int* row_array = ptr->rowArray;
    const int* row_offset = ptr->rowOffset;
    int row_start = 0;
    while (row_start < row_size) {
#ifdef ALIGN_Y
        int row_newstart = row_start;
        for (auto mis = misalign(ptr, ptr->rowArray[row_newstart]); mis != 0; ) {
            row_newstart += static_cast<int>(mis);
            if (row_newstart > row_size) {
                row_newstart = row_size;
                break;
            }
        }
        if (row_start < row_newstart) {
            int sep = row_newstart - row_start;
            tasks.run([=] { kernel_0(ptr, row_start, sep); });
            row_start = row_newstart;
        }
        if (row_start >= row_size) {
            break;
        }
        // now Y[row_array[row_start]] is aligned
#endif
        int row_end = row_start + Parameters<TaskMatrixInfo>::SEP;
        if (row_end > row_size) {
            row_end = row_size;
        }
        int sep = row_end - row_start;
        int nnz = row_offset[row_end] - row_offset[row_start];
        int dist = row_array[row_end - 1] - row_array[row_start] + 1;
        if (sep != dist || sep < Parameters<TaskMatrixInfo>::SIMD_MIN_M) {
            tasks.run([=]() { kernel_0(ptr, row_start, sep); });
        }
        else {
            if (nnz == sep) {
                tasks.run([=] { kernel_2(ptr, row_start, sep); });
            }
            else if (nnz > sep * 10) {
                tasks.run([=] { kernel_0(ptr, row_start, sep); });
            }
            else {
                tasks.run([=] { kernel_1(ptr, row_start, sep); });
            }
        }
        row_start = row_end;
    }
    tasks.wait();
}

} // namespace FastSPICEmv

#ifdef HALF_THREADS
#define ACTIVE(t_num) (t_num >> 1)
#else
#define ACTIVE(t_num) t_num
#endif
#define GET_NUM_THREADS tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism)

void matrix_calc_taskA(TaskMatrixInfoA** ptr, int size) {
    auto num_threads = GET_NUM_THREADS;
    tbb::task_arena active_threads(ACTIVE(num_threads));
    active_threads.execute([=] {
        tbb::parallel_for(0, size, [=](int i) {
            FastSPICEmv::calc(ptr[i]);
        }, tbb::static_partitioner());
    });
}

void matrix_calc_taskB(TaskMatrixInfoB** ptr, int size) {
    auto num_threads = GET_NUM_THREADS;
    tbb::task_arena active_threads(ACTIVE(num_threads));
    active_threads.execute([=] {
        tbb::parallel_for(0, size, [=](int i) {
            FastSPICEmv::calc(ptr[i]);
        }, tbb::static_partitioner());
    });
}
