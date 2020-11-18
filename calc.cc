#include "calc.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include "taskA.h"
#include "taskB.h"
#include "padding.h"
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

struct ExtraInfo {
    int id;
    int calculation;
    enum {
        Distinct = 0,
        Contiguous,
        Ones,
        Longrow
    } type;
};

void matrix_calc_subtask(const std::vector<const std::pair<TaskMatrixInfoA, ExtraInfo>*>& subtask) {
    //void matrix_calc_taskA_subtask(const std::pair<TaskMatrixInfoA, ExtraInfo>* subtask) {
    for (const auto& st : subtask) {
        const auto& task = st->first;
        const auto& info = st->second;
        switch (info.type) {
            case ExtraInfo::Distinct:
                taskA(task.rowArray, task.rowOffset, task.rowArraySize, task.columnIndice,
                    task.S, task.valueNormalMatrix, task.Id);
                break;
            case ExtraInfo::Contiguous:
                //if (mA[info.id] < Parameters<TaskMatrixInfo>::SIMD_MIN_M) {
                    //spmv_rowwise(task.rowArray[0], task.rowArray[0] + task.rowArraySize,
                            //task.rowOffset, task.columnIndice, task.valueNormalMatrix,
                            //task.S, task.Id);
                //}
                //else {
                    spmv_rowwise_simd_taskA(task.rowArray[0],
                            task.rowArray[0] + task.rowArraySize, task.rowOffset, task.columnIndice,
                            task.valueNormalMatrix, task.S, task.Id);
                    //spmv_segmentedsum_simd(&segA[info.id], task.rowArray[0],
                            //task.rowArray[0] + task.rowArraySize, task.rowOffset, task.columnIndice,
                            //task.valueNormalMatrix, task.S, task.Id);
                //}
                break;
            case ExtraInfo::Ones:
                spmv_row_1_taskA(task.rowArray[0], task.rowArray[0] + task.rowArraySize, task.rowOffset,
                        task.columnIndice, task.valueNormalMatrix, task.S, task.Id);
                break;
            default:
                fprintf(stderr, "fuck\n");
        }
    }
}

void matrix_calc_subtask(const std::vector<const std::pair<TaskMatrixInfoB, ExtraInfo>*>& subtask) {
    for (const auto& st : subtask) {
        const auto& task = st->first;
        const auto& info = st->second;
        switch (info.type) {
            case ExtraInfo::Distinct:
                taskB(task.valueSpiceMatrix, task.rowOffset, task.columnIndice,
                        task.A, task.S, task.R, task.H, task.D, task.IC,
                        task.IG, task.alpha, task.rowArray, task.rowArraySize);

                break;
            case ExtraInfo::Contiguous:
                //if (mB[info.id] < SIMD_MIN_M) {
                    //spmv_rowwise_taskB(task.rowArray[0], task.rowArray[0] + task.rowArraySize,
                            //task.rowOffset, task.columnIndice, task.valueSpiceMatrix, task.S,
                            //task.D, task.IG, task.IC, task.R, task.H, task.A, task.alpha);
                //}
                //else {
                    spmv_rowwise_simd_taskB(task.rowArray[0], task.rowArray[0] + task.rowArraySize,
                            task.rowOffset, task.columnIndice, task.valueSpiceMatrix, task.S,
                            task.D, task.IG, task.IC, task.R, task.H, task.A, task.alpha);
                    //spmv_segmentedsum_simd_taskB(&segB[info.id], task.rowArray[0],
                            //task.rowArray[0] + task.rowArraySize,
                            //task.rowOffset, task.columnIndice, task.valueSpiceMatrix, task.S,
                            //task.D, task.IG, task.IC, task.R, task.H, task.A, task.alpha);
                //}
                break;
            case ExtraInfo::Ones:
                spmv_row_1_taskB(task.rowArray[0], task.rowArray[0] + task.rowArraySize,
                            task.rowOffset, task.columnIndice, task.valueSpiceMatrix, task.S,
                            task.D, task.IG, task.IC, task.R, task.H, task.A, task.alpha);
                break;
            default:
                fprintf(stderr, "fuck\n");
        }
    }
}

template <typename TaskMatrixInfo>
int64_t init_subtask(TaskMatrixInfo** ptr, const std::vector<int>& ids, 
        std::vector<std::pair<TaskMatrixInfo, ExtraInfo>>& subtasks) {
    int64_t total_calculations = 0;
    subtasks.clear();
    for (int i : ids) {
        int64_t calculations = 0;
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
            if (sep != dist || sep < Parameters<TaskMatrixInfo>::SEP) {
                subtasks.push_back({*ptr[i], {i, sep + nnz / dist * sep, ExtraInfo::Distinct}});
            }
            else {
                if (nnz == sep) {
                    subtasks.push_back({*ptr[i], {i, sep + nnz, ExtraInfo::Ones}});
                }
                else if (nnz > sep * 10) {
                    subtasks.push_back({*ptr[i], {i, sep + nnz, ExtraInfo::Distinct}});
                }
                else {
                    subtasks.push_back({*ptr[i], {i, sep + nnz, ExtraInfo::Contiguous}});
                }
            }
            subtasks.back().first.rowArray += row_start;
            subtasks.back().first.rowArraySize = sep;
            calculations += subtasks.back().second.calculation;
            row_start = row_end;
        }
        total_calculations += calculations;
    }
    return total_calculations;
}

template <typename TaskMatrixInfo>
void init_matrix(TaskMatrixInfo** ptr, int size, std::vector<std::vector<int>>& subinit) {
    std::vector<int> m (size);
    int64_t total_m = 0;
#pragma omp parallel for reduction(+:total_m)
    for (int i = 0; i < size; ++i) {
        int mi = guess_matrix_size(ptr[i]);
        m[i] = mi;
        total_m += mi;
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
}

template <typename TaskMatrixInfo>
void matrix_calc(TaskMatrixInfo** ptr, int size) {
    static bool initialized = false;
    //static std::vector<segmentedsum_t> segA;
    static std::vector<std::vector<int>> subinit (number_of.threads);
    static padded::vector<std::vector<std::pair<TaskMatrixInfo, ExtraInfo>>> subtasks (number_of.threads); // concurrent write
    static std::vector<std::vector<const std::pair<TaskMatrixInfo, ExtraInfo>*>> stss (number_of.threads);

    if (!initialized) {
        //segA.resize(size);
        init_matrix(ptr, size, subinit);
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
            current_calculation += st.second.calculation;
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