#include<immintrin.h>
#include "spmv.h"

// b : row begin, e : row end
void spmv_rowwise(int b, int e, const int *offsets, const int *indices, const double *values, const double *x, double *y) {
    for(int i = b; i < e; ++i) {
        double v = 0.0;
        for(int p = offsets[i]; p < offsets[i + 1]; ++p) {
            v += values[p] * x[indices[p]];
        }
        y[i] += v;
    }
}

void spmv_rowwise_taskB(int b, int e, const int *offsets, const int *indices, const double *values, const double *x, \
    const double *D, double *IG, double *IC, double *R, double *H) {
    for(int i = b; i < e; ++i) {
        double v0 = 0.0, v1 = 0.0;
        for(int p = offsets[i]; p < offsets[i + 1]; ++p) {
            v0 += values[p*2] * x[indices[p]];
            v1 += values[p*2+1] * x[indices[p]];
        }
        IG[i] += v0;
        IC[i] += v1;
        R[i] = D[i*2] - v0;
        H[i] = D[i*2+1] - v1;
    }
}

void spmv_rowwise_simd(int b, int e, const int *offsets, const int *indices, const double *values, const double *x, double *y) {
    const __m256i step = _mm256_set1_epi32(1), zeroi = _mm256_setzero_si256();
    const __m512d zerod = _mm512_setzero_pd();
    int i;
    for(i = b; i < (e - b) & (~7); i += 8) {
        __m512d v = _mm512_loadu_pd(y + i);
        __m256i p = _mm256_loadu_si256((const __m256i*)(offsets + i));
        __m256i pend = _mm256_loadu_si256((const __m256i*)(offsets + i + 1));
        for(__mmask8 active = _mm256_cmple_epi32_mask(p, pend); 0 != _ktestz_mask8_u8(0xff, active); \
            p = _mm256_add_epi32(p, step), active = _mm256_cmple_epi32_mask(p, pend)) {
            __m512d val0 = _mm512_mask_i32gather_pd(zerod, active, p, values, 8);
            __m256i idx0 = _mm256_mmask_i32gather_epi32(zeroi, active, p, indices, 4);
            v = _mm512_fmadd_pd(val0, _mm512_mask_i32gather_pd(zerod, active, idx0, x, 8), v);
        }
        _mm512_storeu_pd(y + i, v);
    }
    /* remainder */
    for(; i < e; ++i) {
        double v = 0.0;
        for(int p = offsets[i]; p < offsets[i + 1]; ++p) {
            v += values[p] * x[indices[p]];
        }
        y[i] = v;
    }
}

void spmv_rowwise_simd_taskB(int b, int e, const int *offsets, const int *indices, const double *values, const double *x, \
    const double *D, double *IG, double *IC, double *R, double *H) {
    const __m256i onei = _mm256_set1_epi32(1), zeroi = _mm256_setzero_si256();
    const __m512d zerod = _mm512_setzero_pd();
    const __m512i idxl = _mm512_set_epi64(14,12,10,8,6,4,2,0);
    const __m512i idxh = _mm512_set_epi64(15,13,11,9,7,5,3,1);
    int i;
    for(i = b; i < (e - b) & (~7); i += 8) {
        __m512d v0 = _mm512_setzero_pd(), v1 = _mm512_setzero_pd();
        __m256i p = _mm256_loadu_si256((const __m256i*)(offsets + i));
        __m256i pend = _mm256_loadu_si256((const __m256i*)(offsets + i + 1));

        for(__mmask8 active = _mm256_cmple_epi32_mask(p, pend); 0 != _ktestz_mask8_u8(0xff, active); \
            p = _mm256_add_epi32(p, onei), active = _mm256_cmple_epi32_mask(p, pend)) {
            __m256i p2 = _mm256_slli_epi32(p, 1);
            __m512d val0 = _mm512_mask_i32gather_pd(zerod, active, p2, values, 8);
            __m512d val1 = _mm512_mask_i32gather_pd(zerod, active, _mm256_add_epi32(p2, onei), values, 8);
            __m256i idx0 = _mm256_mmask_i32gather_epi32(zeroi, active, p, indices, 4);
            __m512d xx = _mm512_mask_i32gather_pd(zerod, active, idx0, x, 8);
            v0 = _mm512_fmadd_pd(val0, xx, v0);
            v1 = _mm512_fmadd_pd(val1, xx, v1);
        }

        _mm512_storeu_pd(IG + i, _mm512_add_pd(_mm512_loadu_pd(IG + i), v0));
        _mm512_storeu_pd(IC + i, _mm512_add_pd(_mm512_loadu_pd(IC + i), v1));
        __m512d d0 = _mm512_loadu_pd(D + i * 2), d1 = _mm512_loadu_pd(D + i * 2 + 8);
        _mm512_storeu_pd(R + i, _mm512_add_pd(PACKD0246(d0, d1), v0));
        _mm512_storeu_pd(H + i, _mm512_add_pd(PACKD1357(d0, d1), v1));
    }
    /* remainder */
    for(; i < e; ++i) {
        double v0 = 0.0, v1 = 0.0;
        for(int p = offsets[i]; p < offsets[i + 1]; ++p) {
            v0 += values[p*2] * x[indices[p]];
            v1 += values[p*2+1] * x[indices[p]];
        }
        IG[i] += v0;
        IC[i] += v1;
        R[i] = D[i*2] - v0;
        H[i] = D[i*2+1] - v1;
    }
}

void spmv_segmentedsum(int b, int e, const int *offsets, const int *indices, const double *values, const double *x, double *y) {
    int i, p;
    double v = y[b];
    for(i = b, p = offsets[b]; p < offsets[e]; ++p) {
        while(p == offsets[i + 1]) {
            y[i++] = v;
            v = y[i];
        }
        v += values[p] * x[indices[p]];
    }
    while(offsets[e] == offsets[i + 1]) {
        y[i++] = v;
        v = y[i];
    }
}

void spmv_segmentedsum_taskB(int b, int e, const int *offsets, const int *indices, const double *values, const double *x, \
    const double *D, double *IG, double *IC, double *R, double *H) {
    int i, p;
    double v0 = 0.0, v1 = 0.0;
    for(i = b, p = offsets[b]; p < offsets[e]; ++p) {
        while(p == offsets[i + 1]) {
            IG[i] += v0;
            IC[i] += v1;
            R[i] = D[i*2] - v0;
            H[i] = D[i*2+1] - v1;
            ++i;
            v0 = 0.0, v1 = 0.0;
        }
        v0 += values[p*2] * x[indices[p]];
        v1 += values[p*2+1] * x[indices[p]];
    }
    while(offsets[e] == offsets[i + 1]) {
        IG[i] += v0;
        IC[i] += v1;
        R[i] = D[i*2] - v0;
        H[i] = D[i*2+1] - v1;
        ++i;
        v0 = 0.0, v1 = 0.0;
    }
}




