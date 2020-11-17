#include "calc.h"

void taskA(
    int* rowArray,
    const int* rowOffset,
    int rowArraySize,
    const int* columnIndice,
    const double* S,
    const double* valueNormalMatrix,
    double* Id
) {
    for (int i = 0; i < rowArraySize; ++i) {
        const int node = rowArray[i];
        for (int j = rowOffset[node]; j < rowOffset[node + 1]; ++j) {
            Id[node] += valueNormalMatrix[j] * S[columnIndice[j]];
        }
    }
}

void taskB(
    const double* valueSpiceMatrix,
    const int* rowOffset,
    const int* columnIndice,
    double* A,
    double* S,
    double* R,
    double* H,
    const double* D,
    double* IC,
    double* IG,
    double alpha,
    int* rowArray,
    int rowArraySize
) {
    for (int i = 0; i < rowArraySize; ++i) {
        int row = rowArray[i];
        const int k1 = row * 2;

        double ig = 0;
        double ic = 0;

        for (int p = rowOffset[row]; p < rowOffset[row + 1]; ++p) {
            int col = columnIndice[p];
            const int k = p * 2;
            double cond = valueSpiceMatrix[k];
            double cap = valueSpiceMatrix[k + 1];
            ig += cond * S[col];
            ic += cap * S[col];
            A[p] = cond + alpha * cap;
        }
        IG[row] += ig;
        IC[row] += ic;
        R[row] = D[k1] - ig;
        H[row] = D[k1 + 1] - ic;
    }
}

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
