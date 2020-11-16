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

void matrix_calc_taskA(TaskMatrixInfoA** ptr, int size) {
    const int jobs = omp_get_max_threads();
    printf("jobs: %d\n", jobs);

    int64_t total_row = 0;
    for (int i = 0; i < size; ++i) {
        total_row += ptr[i]->rowArraySize;
    }
    const int64_t row_per_job = total_row / jobs + 1;
    const int64_t offset_threshold = row_per_job - 1 > 40 ? row_per_job - 1 : 40;

    static std::vector<std::vector<TaskMatrixInfoA>> subtasks;
    subtasks.resize(jobs);
    for (int i = 0; i < jobs; ++i) {
        subtasks[i].clear();
    }
    int current_job = 0;
    int64_t current_job_rows = 0;
    for (int i = 0; i < size; ++i) {
        const int row_size = ptr[i]->rowArraySize;
        int* row_array = ptr[i]->rowArray;
        std::sort(row_array, row_array + row_size);
        int row_start = 0;
        for (int i = 0; i < row_size; ++i) {
            if (i + 1 == row_size || row_array[i + 1] != row_array[i] + 1) {
                int row_end = i + 1;
                int current_row = row_start;
                while (current_row < row_end) {
                    int remain_rows = row_end - current_row;
                    subtasks[current_job].push_back(*ptr[i]);
                    auto& subtask = subtasks[current_job].back();
                    subtask.rowArray += current_row;
                    if (current_job_rows + remain_rows <= row_per_job + offset_threshold || current_job == jobs - 1) {
                        subtask.rowArraySize = remain_rows;
                    }
                    else {
                        subtask.rowArraySize = static_cast<int>(row_per_job - current_job_rows);
                    }
                    current_job_rows += subtask.rowArraySize;
                    current_row += subtask.rowArraySize;
                    if (current_job_rows >= row_per_job - offset_threshold && current_job != jobs - 1) {
                        current_job++;
                        current_job_rows = 0;
                    }
                }
                row_start = row_end;
            }
        }
    }

#if 1
    FILE* debug = fopen("debug.txt", "a");
    for (const auto& sts : subtasks) {
        for (const auto& st: sts) {
            fprintf(debug, "%d ", st.rowArraySize);
        }
        fprintf(debug, "\n");
    }
    fclose(debug);
#endif

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
