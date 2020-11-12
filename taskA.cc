#include "taskA.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

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

        for (int j = rowOffset[node]; j < rowOffset[node + 1]; ++j) {
            Id[node] += valueNormalMatrix[j] * S[columnIndice[j]];
        }
    }
}
