#ifndef TASKA_H_
#define TASKA_H_

#include "calc.h"

void taskA(
    int* rowArray,
    const int* rowOffset,
    int rowArraySize,
    const int* columnIndice,
    const double* S,
    const double* valueNormalMatrix,
    double* Id
);

#endif /* ifndef TASKA_H_ */
