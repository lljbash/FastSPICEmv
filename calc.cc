#include "calc.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include "taskA.h"
#include "taskB.h"
#include "spmv.h"

constexpr int SEP = 128;
constexpr int SIMD_MIN_M = 32;
constexpr int LONGROW_MIN_NNZ = 1000;
constexpr int LONGROW_AVG_NNZ = LONGROW_MIN_NNZ / SEP;

//std::vector<segmentedsum_t> segA;
std::vector<int> mA;
//std::vector<segmentedsum_t> segB;
std::vector<int> mB;

std::vector<std::vector<int>> subinitA;
std::vector<std::vector<int>> subinitB;

int jobs = 1;

__attribute__((constructor))
void get_jobs() {
#pragma omp parallel
    {
#pragma omp single
        {
            jobs = omp_get_max_threads();
        }
    }
}

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
        Continumous,
        Ones,
        Longrow
    } type;
};

void matrix_calc_taskA_subtask(const std::vector<const std::pair<TaskMatrixInfoA, ExtraInfo>*>& subtask) {
//void matrix_calc_taskA_subtask(const std::pair<TaskMatrixInfoA, ExtraInfo>* subtask) {
    for (const auto& st : subtask) {
        const auto& task = st->first;
        const auto& info = st->second;
        switch (info.type) {
            case ExtraInfo::Distinct:
                taskA(task.rowArray, task.rowOffset, task.rowArraySize, task.columnIndice,
                    task.S, task.valueNormalMatrix, task.Id);
                break;
            case ExtraInfo::Continumous:
                //if (mA[info.id] < SIMD_MIN_M) {
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
                printf("fuck\n");
        }
    }
}

void matrix_calc_taskB_subtask(const std::vector<const std::pair<TaskMatrixInfoB, ExtraInfo>*>& subtask) {
//void matrix_calc_taskB_subtask(const std::pair<TaskMatrixInfoB, ExtraInfo>* subtask) {
    for (const auto& st : subtask) {
        const auto& task = st->first;
        const auto& info = st->second;
        switch (info.type) {
            case ExtraInfo::Distinct:
                taskB(task.valueSpiceMatrix, task.rowOffset, task.columnIndice,
                        task.A, task.S, task.R, task.H, task.D, task.IC,
                        task.IG, task.alpha, task.rowArray, task.rowArraySize);

                break;
            case ExtraInfo::Continumous:
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
                printf("fuck\n");
        }
    }
}

template <class TaskMatrixInfo>
void init_subtask(TaskMatrixInfo** ptr, const std::vector<int>& ids,
        std::vector<std::vector<std::pair<TaskMatrixInfo, ExtraInfo>>>& subtasks,
        std::vector<int64_t>& calculations) {
    for (int i : ids) {
        subtasks[i].clear();
        calculations[i] = 0;
        const int row_size = ptr[i]->rowArraySize;
        int* row_array = ptr[i]->rowArray;
        const int* row_offset = ptr[i]->rowOffset;
        //std::sort(row_array, row_array + row_size);
        int row_start = 0;
        while (row_start < row_size) {
            int row_end = row_start + SEP;
            if (row_end > row_size) {
                row_end = row_size;
            }
            int sep = row_end - row_start;
            int nnz = row_offset[row_end] - row_offset[row_start];
            int dist = row_array[row_end - 1] - row_array[row_start] + 1;
            if (sep != dist || sep < SIMD_MIN_M) {
                subtasks[i].push_back({*ptr[i], {i, sep + nnz / dist * sep, ExtraInfo::Distinct}});
            }
            else {
                if (nnz == sep) {
                    subtasks[i].push_back({*ptr[i], {i, sep + nnz, ExtraInfo::Ones}});
                }
                else if (nnz > sep * 10) {
                    subtasks[i].push_back({*ptr[i], {i, sep + nnz, ExtraInfo::Distinct}});
                }
                else {
                    subtasks[i].push_back({*ptr[i], {i, sep + nnz, ExtraInfo::Continumous}});
                }
            }
            subtasks[i].back().first.rowArray += row_start;
            subtasks[i].back().first.rowArraySize = sep;
            calculations[i] += subtasks[i].back().second.calculation;
            row_start = row_end;
        }
    }
}

void matrix_calc_taskA(TaskMatrixInfoA** ptr, int size) {
    static bool initialized = false;
    if (!initialized) {
        //segA.resize(size);
        mA.resize(size);
#pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            mA[i] = guess_matrix_size(ptr[i]);
            //if (mA[i] >= SIMD_MIN_M) {
                //initialize_segmentedsum(&segA[i], mA[i], ptr[i]->rowOffset, ptr[i]->columnIndice, ptr[i]->valueNormalMatrix);
            //}
        }
        subinitA.resize(jobs);
        int64_t total_m = std::accumulate(mA.begin(), mA.end(), 0);
        int64_t m_per_job = total_m / jobs;
        int current_job = 0;
        int64_t current_m = 0;
        for (int i = 0; i < size; ++i) {
            if (current_job < jobs - 1 && current_m > m_per_job) {
                ++current_job;
                current_m = 0;
            }
            subinitA[current_job].push_back(i);
            current_m += mA[i];
        }
        initialized = true;
    }

    static std::vector<std::vector<std::pair<TaskMatrixInfoA, ExtraInfo>>> subtasks;
    static std::vector<int64_t> calculations;
    subtasks.resize(size);
    calculations.resize(size);
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        init_subtask(ptr, subinitA[tid], subtasks, calculations);
    }

#if 0
    FILE* debug = fopen("debug.txt", "a");
    for (const auto& sts : subtasks) {
        for (const auto& st: sts) {
            fprintf(debug, "%d:%d ", st.first.rowArraySize, st.second.type);
        }
        fprintf(debug, "\n");
    }
    fclose(debug);
#endif

    //static std::vector<const std::pair<TaskMatrixInfoA, ExtraInfo>*> stss;
    //stss.clear();
    //for (const auto& sts: subtasks) {
        //for (const auto& st : sts) {
            //stss.push_back(&st);
        //}
    //}
//#pragma omp parallel for
    //for (size_t i = 0; i < stss.size(); ++i) {
        //matrix_calc_taskA_subtask(stss[i]);
    //}

    static std::vector<std::vector<const std::pair<TaskMatrixInfoA, ExtraInfo>*>> stss;
    stss.resize(jobs);
    for (int i = 0; i < jobs; ++i) {
        stss[i].clear();
    }
    int64_t total_calculation = std::accumulate(calculations.begin(), calculations.end(), 0);
    int64_t calculation_per_job = total_calculation / jobs;
    int current_job = 0;
    int64_t current_calculation = 0;
    for (const auto& sts : subtasks) {
        for (const auto& st : sts) {
            if (current_job < jobs - 1 && current_calculation > calculation_per_job) {
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
        matrix_calc_taskA_subtask(stss[tid]);
    }
}

void matrix_calc_taskB(TaskMatrixInfoB** ptr, int size) {
    static bool initialized = false;
    if (!initialized) {
        //segB.resize(size);
        mB.resize(size);
#pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            mB[i] = guess_matrix_size(ptr[i]);
            //if (mB[i] >= SIMD_MIN_M) {
                //initialize_segmentedsum(&segB[i], mB[i], ptr[i]->rowOffset,
                        //ptr[i]->columnIndice, ptr[i]->valueSpiceMatrix);
            //}
        }
        subinitB.resize(jobs);
        int64_t total_m = std::accumulate(mB.begin(), mB.end(), 0);
        int64_t m_per_job = total_m / jobs;
        int current_job = 0;
        int64_t current_m = 0;
        for (int i = 0; i < size; ++i) {
            if (current_job < jobs - 1 && current_m > m_per_job) {
                ++current_job;
                current_m = 0;
            }
            subinitB[current_job].push_back(i);
            current_m += mB[i];
        }
        initialized = true;
    }

    static std::vector<std::vector<std::pair<TaskMatrixInfoB, ExtraInfo>>> subtasks;
    static std::vector<int64_t> calculations;
    subtasks.resize(size);
    calculations.resize(size);
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        init_subtask(ptr, subinitB[tid], subtasks, calculations);
    }

    //static std::vector<const std::pair<TaskMatrixInfoB, ExtraInfo>*> stss;
    //stss.clear();
    //for (const auto& sts: subtasks) {
        //for (const auto& st : sts) {
            //stss.push_back(&st);
        //}
    //}
//#pragma omp parallel for
    //for (size_t i = 0; i < stss.size(); ++i) {
        //matrix_calc_taskB_subtask(stss[i]);
    //}

    static std::vector<std::vector<const std::pair<TaskMatrixInfoB, ExtraInfo>*>> stss;
    stss.resize(jobs);
    for (int i = 0; i < jobs; ++i) {
        stss[i].clear();
    }
    int64_t total_calculation = std::accumulate(calculations.begin(), calculations.end(), 0);
    int64_t calculation_per_job = total_calculation / jobs;
    int current_job = 0;
    int64_t current_calculation = 0;
    for (const auto& sts : subtasks) {
        for (const auto& st : sts) {
            if (current_job < jobs - 1 && current_calculation > calculation_per_job) {
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
        matrix_calc_taskB_subtask(stss[tid]);
    }
}
