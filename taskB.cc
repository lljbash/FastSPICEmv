#include "taskB.h"

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
#if 0
    // (4) (5)
    for (int i = 0; i < rowArraySize; ++i) {
        int row = rowArray[i];

        for (int p = rowOffset[row]; p < rowOffset[row + 1]; ++p) {
            int col = columnIndice[p];
            const int k = p * 2;
            double cond = valueSpiceMatrix[k];
            double cap = valueSpiceMatrix[k + 1];

            IG[row] += cond * S[col];
            IC[row] += cap * S[col];
        }
    }

    // (7) (8)
    for (int i = 0; i < rowArraySize; ++i) {
        int row = rowArray[i];
        const int k1 = row * 2;
        double current = D[k1];
        double charge = D[k1 + 1];

        for (int p = rowOffset[row]; p < rowOffset[row + 1]; ++p) {
            int col = columnIndice[p];
            const int k = p * 2;

            current -= valueSpiceMatrix[k] * S[col];
            charge -= valueSpiceMatrix[k + 1] * S[col];
        }
        R[row] = current;
        H[row] = charge;
    }

    // (9)
    for (int i = 0; i < rowArraySize; ++i) {
        int row = rowArray[i];

        for (int p = rowOffset[row]; p < rowOffset[row + 1]; ++p) {
            const int k = p * 2;
            double cond = valueSpiceMatrix[k];
            double cap = valueSpiceMatrix[k + 1];
            A[p] = cond + alpha * cap;
        }
    }
#else
    for (int i = 0; i < rowArraySize; ++i) {
        int row = rowArray[i];
        const int k1 = row * 2;

        double ig = 0;
        double ic = 0;

        for (int p = rowOffset[row]; p < rowOffset[row + 1]; ++p) {
            int col = columnIndice[p];
            const int k = p * 2;
            ig += valueSpiceMatrix[k] * S[col];
            ic += valueSpiceMatrix[k + 1] * S[col];
        }
        IG[row] += ig;
        IC[row] += ic;
        R[row] = D[k1] - ig;
        H[row] = D[k1 + 1] - ic;
    }

    for (int i = 0; i < rowArraySize; ++i) {
        int row = rowArray[i];

        for (int p = rowOffset[row]; p < rowOffset[row + 1]; ++p) {
            const int k = p * 2;
            double cond = valueSpiceMatrix[k];
            double cap = valueSpiceMatrix[k + 1];
            A[p] = cond + alpha * cap;
        }
    }
#endif
}
