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
    //memset(Id, 0, sizeof(double[rowArray[rowArraySize - 1]]));
    //FILE* debug = fopen("debug.txt", "w");

    // (6)
    //printf("%d\n", rowArraySize);
    for (int i = 0; i < rowArraySize; ++i) {
        const int node = rowArray[i];

        for (int j = rowOffset[node]; j < rowOffset[node + 1]; ++j) {
            Id[node] += valueNormalMatrix[j] * S[columnIndice[j]];
        }

        //double id[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        //int count = rowOffset[node + 1] - rowOffset[node];
        //int n = (count + 7) / 8;
        //int j = rowOffset[node];
        //switch (count % 8) {
        //case 0: do { id[0] += valueNormalMatrix[j] * S[columnIndice[j]]; ++j;
        //case 7:      id[7] += valueNormalMatrix[j] * S[columnIndice[j]]; ++j;
        //case 6:      id[6] += valueNormalMatrix[j] * S[columnIndice[j]]; ++j;
        //case 5:      id[5] += valueNormalMatrix[j] * S[columnIndice[j]]; ++j;
        //case 4:      id[4] += valueNormalMatrix[j] * S[columnIndice[j]]; ++j;
        //case 3:      id[3] += valueNormalMatrix[j] * S[columnIndice[j]]; ++j;
        //case 2:      id[2] += valueNormalMatrix[j] * S[columnIndice[j]]; ++j;
        //case 1:      id[1] += valueNormalMatrix[j] * S[columnIndice[j]]; ++j;
                //} while (--n > 0);
        //}
        //Id[node] += id[0] + id[1] + id[2] + id[3] + id[4] + id[5] + id[6] + id[7];
        //double temp = id[0] + id[1] + id[2] + id[3] + id[4] + id[5] + id[6] + id[7];

        //if (Id[node] != temp) {
            //printf("%a %a %e\n", Id[node], temp, fabs((temp - Id[node]) / Id[node]));
        //}

        //fprintf(debug, "%d %a\n", node, Id[node]);
    }

    //fclose(debug);
}
