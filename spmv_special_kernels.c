#include "spmv.h"

// b : row begin, e : row end
// 1 nonzero per row
void spmv_row_1_taskA(int b, int e, const int *offsets, const int *indices, const double *values, const double *x, double *y) {
    int offset_0 = offsets[b] - b;
    int i;
    for(i = b; i < e - 15; i += 16) {
        const int p = offset_0 + i;
        __m512d v0 = LOAD_PD(y + i);
        __m512d v1 = LOAD_PD(y + i + 8);
        __m512d val0 = _mm512_loadu_pd(values + p);
        __m512d val1 = _mm512_loadu_pd(values + p + 8);
        __m512i idx = _mm512_loadu_si512(indices + p);
        v0 = _mm512_fmadd_pd(val0, _mm512_i32gather_pd(LOI(idx), x, 8), v0);
        v1 = _mm512_fmadd_pd(val1, _mm512_i32gather_pd(HII(idx), x, 8), v1);
        STORE_PD(y + i,     v0);
        STORE_PD(y + i + 8, v1);
    }
    /* remainder */
    for(; i < e; ++i) {
        const int p = offset_0 + i;
        y[i] += values[p] * x[indices[p]];
    }
}

void spmv_row_1_taskB(int b, int e, const int *offsets, const int *indices, const double *values, const double *x, \
    const double *D, double *IG, double *IC, double *R, double *H, double *A, double alpha) {
    const __m512d alpha_v = _mm512_set1_pd(alpha);
    const __m512i idxl = _mm512_set_epi64(14,12,10,8,6,4,2,0);
    const __m512i idxh = _mm512_set_epi64(15,13,11,9,7,5,3,1);
    int offset_0 = offsets[b] - b;
    int i;
    for(i = b; i < e - 15; i += 16) {
        const int p = offset_0 + i;
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
        __m512d v0a = _mm512_mul_pd(valax, x_gthra);
        __m512d v1a = _mm512_mul_pd(valay, x_gthra);
        __m512d v0b = _mm512_mul_pd(valbx, x_gthrb);
        __m512d v1b = _mm512_mul_pd(valby, x_gthrb);
        _mm512_storeu_pd(A + p,     _mm512_fmadd_pd(alpha_v, valay, valax));
        _mm512_storeu_pd(A + p + 8, _mm512_fmadd_pd(alpha_v, valby, valbx));
        STORE_PD        (IG + i,     _mm512_add_pd(LOAD_PD        (IG + i    ), v0a));
        STORE_PD        (IG + i + 8, _mm512_add_pd(LOAD_PD        (IG + i + 8), v0b));
        _mm512_storeu_pd(IC + i,     _mm512_add_pd(_mm512_loadu_pd(IC + i    ), v1a));
        _mm512_storeu_pd(IC + i + 8, _mm512_add_pd(_mm512_loadu_pd(IC + i + 8), v1b));
        __m512d d0a = _mm512_loadu_pd(D + i * 2     ), d1a = _mm512_loadu_pd(D + i * 2 + 8);
        __m512d d0b = _mm512_loadu_pd(D + i * 2 + 16), d1b = _mm512_loadu_pd(D + i * 2 + 24);
        _mm512_storeu_pd(R + i,     _mm512_sub_pd(PACKD0246(d0a, d1a), v0a));
        _mm512_storeu_pd(H + i,     _mm512_sub_pd(PACKD1357(d0a, d1a), v1a));
        _mm512_storeu_pd(R + i + 8, _mm512_sub_pd(PACKD0246(d0b, d1b), v0b));
        _mm512_storeu_pd(H + i + 8, _mm512_sub_pd(PACKD1357(d0b, d1b), v1b));
    }
    /* remainder */
    for(; i < e; ++i) {
        const int p = offset_0 + i;
        double v0 = values[p * 2]     * x[indices[p]];
        double v1 = values[p * 2 + 1] * x[indices[p]];
        A[p] = values[p*2] + alpha * values[p*2+1];
        IG[i] += v0;
        IC[i] += v1;
        R[i] = D[i*2] - v0;
        H[i] = D[i*2+1] - v1;
    }
}

