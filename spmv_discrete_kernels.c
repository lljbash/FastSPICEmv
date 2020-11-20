#include "spmv.h"

//#define LONGROW_SIMD

void taskA(
    int* rowArray,
    const int* rowOffset,
    int rowArraySize,
    const int* columnIndice,
    const double* S,
    const double* valueNormalMatrix,
    double* Id
) {
    // (6)
    for (int i = 0; i < rowArraySize; ++i) {
        const int node = rowArray[i];
#ifdef LONGROW_SIMD
        if (rowOffset[node + 1] - rowOffset[node] >= 32) {
            spmv_long_row_taskA(node, rowOffset, columnIndice, valueNormalMatrix, S, Id);
        }
        else {
#endif
            for (int j = rowOffset[node]; j < rowOffset[node + 1]; ++j) {
                Id[node] += valueNormalMatrix[j] * S[columnIndice[j]];
            }
#ifdef LONGROW_SIMD
        }
#endif
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

#ifdef LONGROW_SIMD
        if (rowOffset[row + 1] - rowOffset[row] >= 32) {
            spmv_long_row_taskB(row, rowOffset, columnIndice, valueSpiceMatrix, S, D, IG, IC, R, H, A, alpha);
        }
        else {
#endif
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
#ifdef LONGROW_SIMD
        }
#endif
    }
}

