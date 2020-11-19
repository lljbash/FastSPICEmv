#include "calc.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include "taskA.h"
#include "taskB.h"
#include "padding.h"
#include "prealloc.h"
#include "spmv.h"

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

struct NumberOfThreads {
    int threads;
    NumberOfThreads() {
#pragma omp parallel
        {
#pragma omp master
            {
                threads = omp_get_max_threads();
            }
        }
    }
};

static NumberOfThreads number_of;

/*
 * example 1: 
 * | y y y y y y y y | y y y y y y y y |
 *         ^
 *     first_row
 * then misalign(matrix, first_row) returns 5;
 */
inline intptr_t misalign(double* y) {  return (- (reinterpret_cast<intptr_t>(y) >> 3)) & 7; }
inline intptr_t misalign(TaskMatrixInfoA* matrix, int first_row) { return misalign(matrix->Id + first_row); }
inline intptr_t misalign(TaskMatrixInfoB* matrix, int first_row) { return misalign(matrix->IG + first_row); }

template <class TaskMatrixInfo>
int guess_matrix_size(TaskMatrixInfo* info) {
    int n = info->rowArray[info->rowArraySize - 1] + 1;
    while (true) {
        int nnz = info->rowOffset[n + 1] - info->rowOffset[n];
        if (nnz > 0 && nnz < 10000) {
            ++n;
        }
        else {
            break;
        }
    }
    return n;
}

template<typename TaskMatrixInfo>
struct TaskInfo {
    const TaskMatrixInfo *matrix;
    int *rowArray;
    int rowArraySize;
    int id;
    int calculation;
    enum {
        Distinct = 0,
        Contiguous,
        Ones
    } type;
};

void matrix_calc_subtask(const prealloc::vector<const TaskInfo<TaskMatrixInfoA>*>& subtask) {
    for (auto st : subtask) {
        const auto& task = *st->matrix;
        const auto& info = *st;
        switch (info.type) {
            case TaskInfo<TaskMatrixInfoA>::Distinct:
                taskA(info.rowArray, task.rowOffset, info.rowArraySize, task.columnIndice,
                    task.S, task.valueNormalMatrix, task.Id);
                break;
            case TaskInfo<TaskMatrixInfoA>::Contiguous:
                spmv_rowwise_simd_taskA(info.rowArray[0], info.rowArray[0] + info.rowArraySize, 
                    task.rowOffset, task.columnIndice, task.valueNormalMatrix, task.S, task.Id);
                break;
            case TaskInfo<TaskMatrixInfoA>::Ones:
                spmv_row_1_taskA(info.rowArray[0], info.rowArray[0] + info.rowArraySize, 
                    task.rowOffset, task.columnIndice, task.valueNormalMatrix, task.S, task.Id);
                break;
            default:
                fprintf(stderr, "fuck\n");
        }
    }
}

void matrix_calc_subtask(const prealloc::vector<const TaskInfo<TaskMatrixInfoB>*>& subtask) {
    for (auto st : subtask) {
        const auto& task = *st->matrix;
        const auto& info = *st;
        switch (info.type) {
            case TaskInfo<TaskMatrixInfoB>::Distinct:
                taskB(task.valueSpiceMatrix, task.rowOffset, task.columnIndice,
                        task.A, task.S, task.R, task.H, task.D, task.IC,
                        task.IG, task.alpha, info.rowArray, info.rowArraySize);

                break;
            case TaskInfo<TaskMatrixInfoB>::Contiguous:
                spmv_rowwise_simd_taskB(info.rowArray[0], info.rowArray[0] + info.rowArraySize, 
                            task.rowOffset, task.columnIndice, task.valueSpiceMatrix, task.S,
                            task.D, task.IG, task.IC, task.R, task.H, task.A, task.alpha);
                break;
            case TaskInfo<TaskMatrixInfoB>::Ones:
                spmv_row_1_taskB(info.rowArray[0], info.rowArray[0] + info.rowArraySize, 
                            task.rowOffset, task.columnIndice, task.valueSpiceMatrix, task.S,
                            task.D, task.IG, task.IC, task.R, task.H, task.A, task.alpha);
                break;
            default:
                fprintf(stderr, "fuck\n");
        }
    }
}

template <typename TaskMatrixInfo, typename Subtasks>
int64_t init_subtask(TaskMatrixInfo** ptr, const prealloc::vector<int>& ids, Subtasks& subtasks) {
    int64_t total_calculations = 0;
    subtasks.clear();
    for (int i : ids) {
        const int row_size = ptr[i]->rowArraySize;
        int* row_array = ptr[i]->rowArray;
        const int* row_offset = ptr[i]->rowOffset;
        //std::sort(row_array, row_array + row_size);
        int row_start = 0;
        while (row_start < row_size) {
            int row_end = row_start + Parameters<TaskMatrixInfo>::SEP;
            if (row_end > row_size) {
                row_end = row_size;
            }
            int sep = row_end - row_start;
            int nnz = row_offset[row_end] - row_offset[row_start];
            int dist = row_array[row_end - 1] - row_array[row_start] + 1;
            if (sep != dist || sep < Parameters<TaskMatrixInfo>::SIMD_MIN_M) {
                subtasks.push_back({ptr[i], row_array + row_start, sep, i, sep + nnz / dist * sep, 
                        TaskInfo<TaskMatrixInfo>::Distinct});
            }
            else {
                if (nnz == sep) {
                    subtasks.push_back({ptr[i], row_array + row_start, sep, i, sep + nnz, 
                        TaskInfo<TaskMatrixInfo>::Ones});
                }
                else if (nnz > sep * 10) {
                    subtasks.push_back({ptr[i], row_array + row_start, sep, i, sep + nnz, 
                        TaskInfo<TaskMatrixInfo>::Distinct});
                }
                else {
                    subtasks.push_back({ptr[i], row_array + row_start, sep, i, sep + nnz, 
                        TaskInfo<TaskMatrixInfo>::Contiguous});
                }
            }
            total_calculations += subtasks.back().calculation;
            row_start = row_end;
        }
    }
    return total_calculations;
}

template <typename TaskMatrixInfo, typename Subinit, typename Subtasks, typename STSS>
void init_matrix(TaskMatrixInfo** ptr, int size, Subinit& subinit, Subtasks& subtasks, STSS& stss) {
    padded::vector<int> m (size);
    int64_t total_m = 0;
#pragma omp parallel reduction(+:total_m)
    {
        int tid = omp_get_thread_num();
        subinit[tid].alloc(size);
#pragma omp for
        for (int i = 0; i < size; ++i) {
            int mi = guess_matrix_size(ptr[i]);
            m[i] = mi;
            total_m += mi;
        }
    }
    int64_t m_per_job = total_m / number_of.threads;
    int current_job = 0;
    int64_t current_m = 0;
    for (int i = 0; i < size; ++i) {
        if (current_job < number_of.threads - 1 && current_m > m_per_job) {
            ++current_job;
            current_m = 0;
        }
        subinit[current_job].push_back(i);
        current_m += m[i];
    }
    int total_max_subtask = 0;
#pragma omp parallel reduction(+:total_max_subtask)
    {
        int tid = omp_get_thread_num();
        int max_subtask = 0;
        for (int id : subinit[tid]) {
            max_subtask += m[id] / Parameters<TaskMatrixInfo>::SEP + 2;
        }
        subtasks[tid].alloc(max_subtask);
        total_max_subtask += max_subtask;
    }
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        stss[tid].alloc(total_max_subtask);
    }
}

template <typename TaskMatrixInfo>
void matrix_calc(TaskMatrixInfo** ptr, int size) {
    static bool initialized = false;
    //static std::vector<segmentedsum_t> segA;
    static padded::vector<prealloc::vector<int>> subinit (number_of.threads);
    static padded::vector<prealloc::vector<TaskInfo<TaskMatrixInfo>>> subtasks (number_of.threads); // concurrent write
    static padded::vector<prealloc::vector<const TaskInfo<TaskMatrixInfo>*>> stss (number_of.threads);

    if (!initialized) {
        //segA.resize(size);
        init_matrix(ptr, size, subinit, subtasks, stss);
        initialized = true;
    }

    int64_t total_calculation = 0;
#pragma omp parallel reduction(+:total_calculation)
    {
        int tid = omp_get_thread_num();
        total_calculation += init_subtask(ptr, subinit[tid], subtasks[tid]);
    }

    int64_t calculation_per_job = total_calculation / number_of.threads;
    int current_job = 0;
    int64_t current_calculation = 0;
    for (int i = 0; i < number_of.threads; ++i) {
        stss[i].clear();
    }
    for (const auto& sts : subtasks) {
        for (const auto& st : sts) {
            if (current_job < number_of.threads - 1 && current_calculation > calculation_per_job) {
                ++current_job;
                current_calculation = 0;
            }
            stss[current_job].push_back(&st);
            current_calculation += st.calculation;
        }
    }
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        matrix_calc_subtask(stss[tid]);
    }
}

void matrix_calc_taskA(TaskMatrixInfoA** calcDataList, int size) {
    matrix_calc(calcDataList, size);
}

void matrix_calc_taskB(TaskMatrixInfoB** calcDataList, int size) {
    matrix_calc(calcDataList, size);
}
