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

// b : row begin, e : row end
// long rows
inline void spmv_long_row_taskA(int row, const int *offsets, const int *indices, const double *values, const double *x, double *y) {
    int p, pend = offsets[row + 1];
    __m512d vv0 = _mm512_setzero_pd(), vv1 = _mm512_setzero_pd();
#ifdef UNROLL_32
    __m512d vv2 = _mm512_setzero_pd(), vv3 = _mm512_setzero_pd();
    for(p = offsets[row]; p < pend - 31; p += 32) {
        __m512i idx0 = _mm512_loadu_si512(indices + p);
        __m512i idx1 = _mm512_loadu_si512(indices + p + 16);
        vv0 = _mm512_fmadd_pd(_mm512_loadu_pd(values + p     ), _mm512_i32gather_pd(LOI(idx0), x, 8), vv0);
        vv1 = _mm512_fmadd_pd(_mm512_loadu_pd(values + p + 8 ), _mm512_i32gather_pd(HII(idx0), x, 8), vv1);
        vv2 = _mm512_fmadd_pd(_mm512_loadu_pd(values + p + 16), _mm512_i32gather_pd(LOI(idx1), x, 8), vv2);
        vv3 = _mm512_fmadd_pd(_mm512_loadu_pd(values + p + 24), _mm512_i32gather_pd(HII(idx1), x, 8), vv3);
    }
    for(; p < pend - 7; p += 8) {
        __m256i idx = _mm256_loadu_si256((const __m256i*)(indices + p));
        vv0 = _mm512_fmadd_pd(_mm512_loadu_pd(values + p     ), _mm512_i32gather_pd(idx, x, 8), vv0);
    }
    vv0 = _mm512_add_pd(_mm512_add_pd(vv0, vv1), _mm512_add_pd(vv2, vv3));
#else
    for(p = offsets[row]; p < pend - 15; p += 16) {
        __m512i idx0 = _mm512_loadu_si512(indices + p);
        vv0 = _mm512_fmadd_pd(_mm512_loadu_pd(values + p     ), _mm512_i32gather_pd(LOI(idx0), x, 8), vv0);
        vv1 = _mm512_fmadd_pd(_mm512_loadu_pd(values + p + 8 ), _mm512_i32gather_pd(HII(idx0), x, 8), vv1);
    }
    vv0 = _mm512_add_pd(vv0, vv1);
#endif
    double v = _mm512_reduce_add_pd(vv0);
    for(; p < pend; ++p) {
        v += values[p] * x[indices[p]];
    }
    y[row] += v;
}

inline void spmv_long_row_taskB(int row, const int *offsets, const int *indices, const double *values, const double *x, \
    const double *D, double *IG, double *IC, double *R, double *H, double *A, double alpha) {
    const __m512d alpha_v = _mm512_set1_pd(alpha);
    const __m512i idxl = _mm512_set_epi64(14,12,10,8,6,4,2,0);
    const __m512i idxh = _mm512_set_epi64(15,13,11,9,7,5,3,1);
    int p, pend = offsets[row + 1];
    __m512d vv0a = _mm512_setzero_pd(), vv1a = _mm512_setzero_pd();
    __m512d vv0b = _mm512_setzero_pd(), vv1b = _mm512_setzero_pd();
    for(p = offsets[row]; p < pend - 15; p += 16) {
        __m512d val0a = _mm512_loadu_pd(values + p * 2);
        __m512d val1a = _mm512_loadu_pd(values + p * 2 + 8);
        __m512d val0b = _mm512_loadu_pd(values + p * 2 + 16);
        __m512d val1b = _mm512_loadu_pd(values + p * 2 + 24);
        __m512i idx = _mm512_loadu_si512(indices + p);
        __m512d x_gthra = _mm512_i32gather_pd(LOI(idx), x, 8);
        __m512d x_gthrb = _mm512_i32gather_pd(HII(idx), x, 8);
        __m512d valax = PACKD0246(val0a, val1a);
        __m512d valay = PACKD1357(val0a, val1a);
        __m512d valbx = PACKD0246(val0b, val1b);
        __m512d valby = PACKD1357(val0b, val1b);
        vv0a = _mm512_fmadd_pd(valax, x_gthra, vv0a);
        vv1a = _mm512_fmadd_pd(valay, x_gthra, vv1a);
        vv0b = _mm512_fmadd_pd(valbx, x_gthrb, vv0b);
        vv1b = _mm512_fmadd_pd(valby, x_gthrb, vv1b);
        _mm512_storeu_pd(A + p,     _mm512_fmadd_pd(alpha_v, valay, valax));
        _mm512_storeu_pd(A + p + 8, _mm512_fmadd_pd(alpha_v, valby, valbx));
    }
    double v0 = _mm512_reduce_add_pd(_mm512_add_pd(vv0a, vv0b));
    double v1 = _mm512_reduce_add_pd(_mm512_add_pd(vv1a, vv1b));
    for(; p < pend; ++p) {
        v0 += values[p*2] * x[indices[p]];
        v1 += values[p*2+1] * x[indices[p]];
        A[p] = values[p*2] + alpha * values[p*2+1];
    }
    IG[row] += v0;
    IC[row] += v1;
    R[row] = D[row*2] - v0;
    H[row] = D[row*2+1] - v1;
}
