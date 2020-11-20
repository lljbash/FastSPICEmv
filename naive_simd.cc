#include "calc.h"
#include <type_traits>
#include "spmv.h"
#include "taskA.h"
#include "taskB.h"

inline intptr_t misalign(double* y) {  return (- (reinterpret_cast<intptr_t>(y) >> 3)) & 7; }
inline intptr_t misalign(TaskMatrixInfoA* matrix, int first_row) { return misalign(matrix->Id + first_row); }
inline intptr_t misalign(TaskMatrixInfoB* matrix, int first_row) { return misalign(matrix->IG + first_row); }

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

template <class TaskMatrixInfo, int type>
void simd_calc(TaskMatrixInfo* ptr, int start, int len) {
    if constexpr (std::is_same_v<std::decay_t<TaskMatrixInfo>, TaskMatrixInfoA>) {
        if constexpr (type == 0) {
            taskA(ptr->rowArray + start, ptr->rowOffset, len, ptr->columnIndice,
                ptr->S, ptr->valueNormalMatrix, ptr->Id);
        }
        else if constexpr (type == 1) {
            spmv_rowwise_simd_taskA(ptr->rowArray[start], ptr->rowArray[start] + len,
                ptr->rowOffset, ptr->columnIndice, ptr->valueNormalMatrix, ptr->S, ptr->Id);
        }
        else {
            spmv_row_1_taskA(ptr->rowArray[start], ptr->rowArray[start] + len,
                ptr->rowOffset, ptr->columnIndice, ptr->valueNormalMatrix, ptr->S, ptr->Id);
        }
    }
    else {
        if constexpr (type == 0) {
            taskB(ptr->valueSpiceMatrix, ptr->rowOffset, ptr->columnIndice,
                ptr->A, ptr->S, ptr->R, ptr->H, ptr->D, ptr->IC,
                ptr->IG, ptr->alpha, ptr->rowArray + start, len);
        }
        else if constexpr (type == 1) {
            spmv_rowwise_simd_taskB(ptr->rowArray[start],
                ptr->rowArray[start] + len,
                ptr->rowOffset, ptr->columnIndice, ptr->valueSpiceMatrix, ptr->S,
                ptr->D, ptr->IG, ptr->IC, ptr->R, ptr->H, ptr->A, ptr->alpha);
        }
        else {
            spmv_row_1_taskB(ptr->rowArray[start],
                ptr->rowArray[start] + len,
                ptr->rowOffset, ptr->columnIndice, ptr->valueSpiceMatrix, ptr->S,
                ptr->D, ptr->IG, ptr->IC, ptr->R, ptr->H, ptr->A, ptr->alpha);
        }
    }
}

template <class TaskMatrixInfo>
void naive_simd(TaskMatrixInfo* ptr) {
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
            simd_calc<TaskMatrixInfo, 0>(ptr, row_start, sep);
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
            simd_calc<TaskMatrixInfo, 0>(ptr, row_start, sep);
        }
        else {
            if (nnz == sep) {
                simd_calc<TaskMatrixInfo, 2>(ptr, row_start, sep);
            }
            else if (nnz > sep * 10) {
                simd_calc<TaskMatrixInfo, 0>(ptr, row_start, sep);
            }
            else {
                simd_calc<TaskMatrixInfo, 1>(ptr, row_start, sep);
            }
        }
        row_start = row_end;
    }
}

void matrix_calc_taskA(TaskMatrixInfoA** ptr, int size) {
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        naive_simd(ptr[i]);
    }
}

void matrix_calc_taskB(TaskMatrixInfoB** ptr, int size) {
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        naive_simd(ptr[i]);
    }
}
