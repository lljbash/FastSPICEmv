#include "calc.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "taskA.h"
#include "taskB.h"

void matrix_calc_taskA_subtask(const std::vector<TaskMatrixInfoA>& subtask) {
    for (const auto& task : subtask) {
        taskA(
            task.rowArray,
            task.rowOffset,
            task.rowArraySize,
            task.columnIndice,
            task.S,
            task.valueNormalMatrix,
            task.Id
        );
    }
}

int64_t estimate_taskA_computation(int64_t rc, int64_t nnz) {
    return rc + nnz;
}

void matrix_calc_taskA(TaskMatrixInfoA** ptr, int size) {
    const int jobs = omp_get_max_threads();
    printf("jobs: %d\n", jobs);

    int64_t total_computation = 0;
    static std::vector<int> nnz;
    nnz.clear();
    for (int i = 0; i < size; ++i) {
        nnz.push_back(0);
        for (int j = 0; j < ptr[i]->rowArraySize; ++j) {
            nnz[i] += ptr[i]->rowOffset[j + 1] - ptr[i]->rowOffset[j];
        }
        total_computation += estimate_taskA_computation(ptr[i]->rowArraySize, nnz[i]);
    }
    int64_t computation_per_job = total_computation / jobs;

    static std::vector<std::vector<TaskMatrixInfoA>> subtasks;
    subtasks.resize(jobs);
    for (int i = 0; i < jobs; ++i) {
        subtasks[i].clear();
    }
    int current_job = 0;
    int64_t current_computation = 0;
    for (int i = 0; i < size; ++i) {
        int64_t remain_computation = estimate_taskA_computation(ptr[i]->rowArraySize, nnz[i]);
        int current_rc = 0;
        while (remain_computation > 0) {
            if (current_computation >= computation_per_job) {
                ++current_job;
                current_computation = 0;
            }
            if (current_computation + remain_computation <= computation_per_job) {
                subtasks[current_job].push_back(*ptr[i]);
                TaskMatrixInfoA& subtask = subtasks[current_job].back();
                subtask.rowArray += current_rc;
                subtask.rowArraySize -= current_rc;
                remain_computation = 0;
            }
            else {
                int next_rc = current_rc;
                int64_t next_computation = 0;
                while (current_computation + next_computation < computation_per_job) {
                    int r_nnz = ptr[i]->rowOffset[next_rc + 1] - ptr[i]->rowOffset[next_rc];
                    next_computation += estimate_taskA_computation(1, r_nnz);
                    ++next_rc;
                }
                subtasks[current_job].push_back(*ptr[i]);
                TaskMatrixInfoA& subtask = subtasks[current_job].back();
                subtask.rowArray += current_rc;
                subtask.rowArraySize = next_rc - current_rc;
                remain_computation -= next_computation;
                current_rc = next_rc;
            }
        }
    }

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        matrix_calc_taskA_subtask(subtasks[tid]);
    }
}

void matrix_calc_taskB(TaskMatrixInfoB** ptr, int size) {
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        taskB(
            ptr[i]->valueSpiceMatrix,
            ptr[i]->rowOffset,
            ptr[i]->columnIndice,
            ptr[i]->A,
            ptr[i]->S,
            ptr[i]->R,
            ptr[i]->H,
            ptr[i]->D,
            ptr[i]->IC,
            ptr[i]->IG,
            ptr[i]->alpha,
            ptr[i]->rowArray,
            ptr[i]->rowArraySize
        );
    }
}
