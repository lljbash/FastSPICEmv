#include "taskA.h"
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
