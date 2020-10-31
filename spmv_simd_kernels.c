#include<immintrin.h>

typedef double data_t;

void spmv_rowwise(int m, int n, int nnz, int *offsets, int *indices, data_t *values, const data_t *x, data_t *y) {
    for(int i = 0; i < m; ++i) {
        data_t v = 0.0;
        for(int p = offsets[i]; p < offsets[i + 1]; ++p) {
            v += values[p] * x[indices[p]];
        }
        y[i] = v;
    }
}

void spmv_rowwise_simd(int m, int n, int nnz, int *offsets, int *indices, data_t *values, const data_t *x, data_t *y) {
    const __m256i step = _mm256_set1_epi32(1);
    int i;
    for(i = 0; i < m & (~7); i += 8) {
        __m512d v = _mm512_setzero_pd();
        __m256i p = _mm256_loadu_epi32(offsets + i);
        __m256i pend = _mm256_loadu_epi32(offsets + i + 1);
        for(; 0 != _ktestz_mask8_u8(0xff,_mm256_cmple_epi32_mask(p, pend)); p = _mm256_add_epi32(p, step)) {
            __m512d val = _mm512_i32gather_pd(   p, values, 1);
            __m256i idx = _mm256_i32gather_epi32(indices, p, 1);
            v = _mm512_fmadd_pd(val, _mm512_i32gather_pd(idx, x, 1), v);
        }
        _mm512_storeu_pd(y + i, v);
    }
    /* remainder */
    for(; i < m; ++i) {
        data_t v = 0.0;
        for(int p = offsets[i]; p < offsets[i + 1]; ++p) {
            v += values[p] * x[indices[p]];
        }
        y[i] = v;
    }
}

void spmv_segmentedsum(int m, int n, int nnz, int *offsets, int *indices, data_t *values, const data_t *x, data_t *y) {
    int i, p;
    data_t v = 0.0;
    for(i = 0, p = 0; p < nnz; ++p) {
        while(p == offsets[i + 1]) {
            y[i++] = v;
            v = 0.0;
        }
        v += values[p] * x[indices[p]];
    }
    while(nnz == offsets[i + 1]) {
        y[i++] = v;
        v = 0.0;
    }
}

#define MASK0123  0x0F
#define MASK4567  0xF0 //_mm512_knot(MASK0123)
#define MASK0145  0x33
#define MASK2367  0xCC //_mm512_knot(MASK0145)
#define MASK0246  0x55
#define MASK1357  0xaa //_mm512_knot(MASK0246)

#define F2I(x)   _mm512_castpd_si512(x)
#define I2F(x)   _mm512_castsi512_pd(x)
#define LOI(x)   _mm512_castsi512_si256(x)
#define HII(x)   _mm512_extracti32x8_epi32(x, 1)

#define BROADCASTD1357(x)  _mm512_permute_pd(x, 0xff)
#define BROADCASTD37(x)    _mm512_permutex_pd(x, 0xff)
#define BROADCASTD7(x)     _mm512_permutexvar_pd(idx7, x)
#define BROADCASTI1357(x)  F2I(_mm512_permute_pd(I2F(x), 0xff))
#define BROADCASTI37(x)    _mm512_permutex_epi64(x, 0xff)
#define BROADCASTI7(x)     _mm512_permutexvar_epi64(idx7, x)

/* READ to ROUND1 */
/* x[0],x[2],x[4],x[6],y[0],y[2],y[4],y[6] */
#define PACKD0246(x, y)     _mm512_permutex2var_pd(x, idxl, y)
#define PACKD1357(x, y)     _mm512_permutex2var_pd(x, idxh, y)
#define PACKI0246(x, y)     _mm512_permutex2var_epi64(x, idxl, y)
#define PACKI1357(x, y)     _mm512_permutex2var_epi64(x, idxh, y)

/* ROUND1 to ROUND2 */
/* x[0],y[0],x[2],y[2],x[4],y[4],x[6],y[6] */
/* valignq is faster than vunpcklpd, vunpckhpd, vshufpd and vshuff64x2, but uses an additional mask */
#define SHUFFLED0246(x, y)  _mm512_mask_movedup_pd(x, MASK1357, y)
#define SHUFFLED1357(x, y)  _mm512_mask_permute_pd(y, MASK0246, x, 0xff)
#define SHUFFLEI0246(x, y)  F2I(_mm512_mask_movedup_pd(I2F(x), MASK1357, I2F(y)))
#define SHUFFLEI1357(x, y)  F2I(_mm512_mask_permute_pd(I2F(y), MASK0246, I2F(x), 0xff))

/* ROUND2 to ROUND3 */
/* x[0],x[1],y[0],y[1],x[4],x[5],y[4],y[5] */
#define SHUFFLE2D0145(x, y) _mm512_mask_permutex_pd(x, MASK2367, y, 0x44)
#define SHUFFLE2D2367(x, y) _mm512_mask_permutex_pd(y, MASK0145, x, 0xee)
#define SHUFFLE2I0145(x, y) _mm512_mask_permutex_epi64(x, MASK2367, y, 0x44)
#define SHUFFLE2I2367(x, y) _mm512_mask_permutex_epi64(y, MASK0145, x, 0xee)

/* ROUND3 to ROUND4 */
#define PACKD0123(x, y)     I2F(_mm512_mask_alignr_epi64(F2I(x), MASK4567, F2I(y), F2I(y), 4))
#define PACKD4567(x, y)     I2F(_mm512_mask_alignr_epi64(F2I(y), MASK0123, F2I(x), F2I(x), 4))
#define PACKI0123(x, y)     _mm512_mask_alignr_epi64(x, MASK4567, y, y, 4)
#define PACKI4567(x, y)     _mm512_mask_alignr_epi64(y, MASK0123, x, x, 4)

void spmv_segmentedsum_simd(int m, int n, int nnz, int *offsets, int *indices, data_t *values, const data_t *x, data_t *y) {
    const __m512i idxl = _mm512_set_epi64(14,12,10,8,6,4,2,0);
    const __m512i idxh = _mm512_set_epi64(15,13,11,9,7,5,3,1);时
    const __m512i idx7 = _mm512_set1_epi64(7);
    int i, p;
    data_t v;
    __m512d sp = _mm512_setzero_pd();

    // (sl, fl) +> (sh, fh) = (if fh then sl + sh else sh, fl & fh)
    for(i = 0, p = offsets[0]; p < nnz & (~15); p += 16) {
        __attribute__((__aligned__(64))) double results[16];
        __mmask8 ml = 0xff, mh = 0xff;
        __m512d sl, sh, sl_old;
        __m512i fl, fh, fl_old, idx;
        int j, k;
        
        /* generate flags */
        for(j = i; offsets[j] < p + 8; ++j) {
            ml = _kandn_mask16(_kshiftli_mask16(1, offsets[j] - p), ml);
        }
        for(k = j; offsets[k] < p + 16; ++k) {
            mh = _kandn_mask16(_kshiftli_mask16(1, offsets[k] - (p + 8)), mh);
        }
        fl = _mm512_movm_epi64(ml);
        fh = _mm512_movm_epi64(mh);

        /* load Matrix A, gather Vector X */
        idx = _mm512_loadu_epi32(indices + p);
        sl = _mm512_mul_pd(_mm512_i32gather_pd(LOI(idx), x, 1), _mm512_loadu_pd(values + p));
        sh = _mm512_mul_pd(_mm512_i32gather_pd(HII(idx), x, 1), _mm512_loadu_pd(values + p + 8));

        /* prefix sum */
        sl_old = sl; fl_old = fl;
        sl = PACKD0246(sl, sh);
        fl = PACKI0246(fl, fh);
        sh = PACKD1357(sl_old, sh);
        fh = PACKI1357(fl_old, fh);
        sh = _mm512_mask_add_pd(sh, _mm512_test_epi64_mask(fh, fh), sl, sh);
        fh = _mm512_and_epi64(fl, fh);

        sl_old = sl; fl_old = fl;
        sl = SHUFFLED0246(sl, sh);
        fl = SHUFFLEI0246(fl, fh);
        sh = SHUFFLED1357(sl_old, sh);
        fh = SHUFFLEI1357(fl_old, fh);
        sh = _mm512_mask_add_pd(sh, _mm512_test_epi64_mask(fh, fh), BROADCASTD1357(sl), sh);
        fh = _mm512_and_epi64(BROADCASTI1357(fl), fh);

        sl_old = sl; fl_old = fl;
        sl = SHUFFLE2D0145(sl, sh);
        fl = SHUFFLE2I0145(fl, fh);
        sh = SHUFFLE2D2367(sl_old, sh);
        fh = SHUFFLE2I2367(fl_old, fh);
        sh = _mm512_mask_add_pd(sh, _mm512_test_epi64_mask(fh, fh), BROADCASTD37(sl), sh);
        fh = _mm512_and_epi64(BROADCASTI37(fl), fh);

        sl_old = sl; fl_old = fl;
        sl = PACKD0123(sl, sh);
        fl = PACKI0123(fl, fh);
        sh = PACKD4567(sl_old, sh);
        fh = PACKI4567(fl_old, fh);
        sl = _mm512_mask_add_pd(sl, _mm512_test_epi64_mask(fl, fl), sp, sl);
        sh = _mm512_mask_add_pd(sh, _mm512_test_epi64_mask(fh, fh), BROADCASTD7(sl), sh);
        sp = BROADCASTD7(sh);

        /* extract result from prefix sum */
        _mm512_store_pd(results,     sl);
        _mm512_store_pd(results + 8, sh);
        for(int j = i; j < k; ++j) {
            int off = offsets[j] - (p + 1);
            if(off >= 0) {
                y[j - 1] = results[off];
                results[off] = 0.0;
            }
        }
        i = k;
    }
    
    /* remainder */
    _mm_store_sd(&v, _mm512_castpd512_pd128(sp));
    for(; p < nnz; ++p) {
        while(p == offsets[i + 1]) {
            y[i++] = v;
            v = 0.0;
        }
        v += values[p] * x[indices[p]];
    }
    while(nnz == offsets[i + 1]) {
        y[i++] = v;
        v = 0.0;
    }
}

