#include "calc.h"
#include "taskA.h"
#include "taskB.h"
#include <cstdlib>

void matrix_calc_taskA(TaskMatrixInfoA** ptr, int size) {
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        taskA(
            ptr[i]->rowArray,
            ptr[i]->rowOffset,
            ptr[i]->rowArraySize,
            ptr[i]->columnIndice,
            ptr[i]->S,
            ptr[i]->valueNormalMatrix,
            ptr[i]->Id
        );
        //exit(-1);
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
