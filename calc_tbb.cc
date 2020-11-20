#include <algorithm>
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
    static constexpr int SEP = 512;
    static constexpr int SIMD_MIN_M = 16;
};

template<>
struct Parameters<TaskMatrixInfoB> {
    static constexpr int SEP = 512;
    static constexpr int SIMD_MIN_M = 16;
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
    const int row_size = ptr->rowArraySize;
#ifdef PARALLEL_PARITITION

    tbb::parallel_for(tbb::blocked_range<int>(0, (row_size - 1) / Parameters<TaskMatrixInfo>::SEP + 1),
        [=](const tbb::blocked_range<int>& range) {
            int row_start = range.begin() * Parameters<TaskMatrixInfo>::SEP;
            int row_end = std::min(range.end() * Parameters<TaskMatrixInfo>::SEP, ptr->rowArraySize);
            int sep = row_end - row_start;
            int nnz = ptr->rowOffset[row_end] - ptr->rowOffset[row_start];
            int dist = ptr->rowArray[row_end - 1] - ptr->rowArray[row_start] + 1;
            if (sep != dist || sep < Parameters<TaskMatrixInfo>::SIMD_MIN_M || nnz > sep * 10) {
                kernel_0(ptr, row_start, sep);
            } else if (nnz == sep) {
                kernel_2(ptr, row_start, sep);
            } else {
                kernel_1(ptr, row_start, sep);
            }
        }, tbb::simple_partitioner()
    );

#else

    tbb::task_group tasks;
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
        int row_end = std::min(row_start + Parameters<TaskMatrixInfo>::SEP, row_size);
        tasks.run([=] {
            int sep = row_end - row_start;
            int nnz = ptr->rowOffset[row_end] - ptr->rowOffset[row_start];
            int dist = ptr->rowArray[row_end - 1] - ptr->rowArray[row_start] + 1;
            if (sep != dist || sep < Parameters<TaskMatrixInfo>::SIMD_MIN_M || nnz > sep * 10) {
                kernel_0(ptr, row_start, sep);
            } else if (nnz == sep) {
                kernel_2(ptr, row_start, sep);
            } else {
                kernel_1(ptr, row_start, sep);
            }
        });
        row_start = row_end;
    }
    tasks.wait();

#endif
}



} // namespace FastSPICEmv

#ifdef THREAD_LIMIT
#define ACTIVE(t_num) std::min(t_num, static_cast<decltype(t_num)>(THREAD_LIMIT))
#else
#define ACTIVE(t_num) t_num
#endif
#define GET_NUM_THREADS tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism)
#define USE_NUM_THREADS(x,f) tbb::task_arena(x, 0).execute(f)

void matrix_calc_taskA(TaskMatrixInfoA** ptr, int size) {
    auto num_threads = GET_NUM_THREADS;
    USE_NUM_THREADS(ACTIVE(num_threads), [=] {
        tbb::parallel_for(0, size, [=](int i) {
            FastSPICEmv::calc(ptr[i]);
        }, tbb::static_partitioner());
    });
}

void matrix_calc_taskB(TaskMatrixInfoB** ptr, int size) {
    auto num_threads = GET_NUM_THREADS;
    USE_NUM_THREADS(ACTIVE(num_threads), [=] {
        tbb::parallel_for(0, size, [=](int i) {
            FastSPICEmv::calc(ptr[i]);
        }, tbb::static_partitioner());
    });
}
