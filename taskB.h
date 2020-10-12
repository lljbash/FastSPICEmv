#ifndef TASKB_H_
#define TASKB_H_

#include "calc.h"

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
);

#endif /* ifndef TASKB_H_ */
