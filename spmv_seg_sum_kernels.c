#include <stdint.h>
#include <immintrin.h>
#include "spmv.h"

#define MASKS_PER_TILE 7
#define WRITE_MASK_POS 5
#define VEC_PER_TILE 2

void initialize_segmentedsum(segmentedsum_t *seg, int m, const int *offsets, const int *indices, const double *values) {
    const __m512i idxl = _mm512_set_epi64(14,12,10,8,6,4,2,0);
    const __m512i idxh = _mm512_set_epi64(15,13,11,9,7,5,3,1);
    const __m512i idx7 = _mm512_set1_epi64(7);
    int head = (8 - ((((uintptr_t)values) >> 3) & 7)) & 7;
    int nnz = offsets[m];
    int n_tiles = (nnz - head) >> 4;
    int i, p, mi, oi;
    
    __mmask8 *masks = (__mmask8*)malloc(sizeof(__mmask8) * n_tiles * MASKS_PER_TILE);
    int *y_offsets = (int*)malloc(sizeof(int) * n_tiles * VEC_PER_TILE + 1);

    for(i = 1; offsets[i] <= head; ++i);
    y_offsets[0] = --i;
    for(p = head, mi = 0, oi = 0; p < nnz - 15; p += 16, mi += MASKS_PER_TILE, oi += VEC_PER_TILE) {
        uint16_t seg_mask = 0, write_mask = 0;
        __m512i fl, fh, fl_old, idx;
        int j, i1, i2;

        /* generate flags */
        for(j = i; offsets[j] <= p + 16; ++j) {
            int off = offsets[j] - (p + 1);
            if(off >= 0) {
                write_mask = (1 << off) | write_mask;
            }
        }
        masks[mi + WRITE_MASK_POS    ] = (__mmask8)(write_mask & 0xff);
        masks[mi + WRITE_MASK_POS + 1] = (__mmask8)((write_mask >> 8) & 0xff);
        i1 = i  + _mm_popcnt_u32((uint32_t)masks[mi + WRITE_MASK_POS    ]);
        i2 = i1 + _mm_popcnt_u32((uint32_t)masks[mi + WRITE_MASK_POS + 1]);
        y_offsets[oi + 1] = i1;
        y_offsets[oi + 2] = i2;
        
        for(j = i; offsets[j] < p + 16; ++j) {
            int off = offsets[j] - p;
            if(off >= 0) {
                seg_mask = (1 << off) | seg_mask;
            }
        }

        i = i2;
        seg_mask = ~seg_mask;
        fl = _mm512_movm_epi64((__mmask8)(seg_mask & 0xff));
        fh = _mm512_movm_epi64((__mmask8)((seg_mask >> 8) & 0xff));

        /* prefix sum */
        fl_old = fl;
        fl = PACKI0246(fl, fh);
        fh = PACKI1357(fl_old, fh);
        masks[mi    ] = _mm512_test_epi64_mask(fh, fh);
        fh = _mm512_and_epi64(fl, fh);
        fl_old = fl;
        fl = SHUFFLEI0246(fl, fh);
        fh = SHUFFLEI1357(fl_old, fh);
        masks[mi + 1] =  _mm512_test_epi64_mask(fh, fh);
        fh = _mm512_and_epi64(BROADCASTI1357(fl), fh);
        fl_old = fl;
        fl = SHUFFLE2I0145(fl, fh);
        fh = SHUFFLE2I2367(fl_old, fh);
        masks[mi + 2] =  _mm512_test_epi64_mask(fh, fh);
        fh = _mm512_and_epi64(BROADCASTI37(fl), fh);
        fl_old = fl;
        fl = PACKI0123(fl, fh);
        fh = PACKI4567(fl_old, fh);
        masks[mi + 3] = _mm512_test_epi64_mask(fl, fl);
        masks[mi + 4] = _mm512_test_epi64_mask(fh, fh);
    }

    seg->masks = masks;
    seg->y_offsets = y_offsets;
    seg->head = head;
    seg->n_tiles = n_tiles;
}

void finalize_segmentedsum(segmentedsum_t *seg) {
    free(seg->masks);
    free(seg->y_offsets);
}

void spmv_segmentedsum_simd(segmentedsum_t *seg, \
    int begin_row, int end_row, const int *offsets, const int *indices, const double *values, const double *x, double *y) {
    const __m512i idxl = _mm512_set_epi64(14,12,10,8,6,4,2,0);
    const __m512i idxh = _mm512_set_epi64(15,13,11,9,7,5,3,1);
    const __m512i idx7 = _mm512_set1_epi64(7);
    __mmask8 *masks = seg->masks;
    int head = seg->head;
    int *y_offsets = seg->y_offsets;
    int nz_begin = offsets[begin_row], nz_end = offsets[end_row];
    int p_begin = (nz_begin <= head) ? head : (nz_begin + (16 - ((nz_begin - head) & 15) & 15));
    int p, t;
    __m512d sp = _mm512_setzero_pd();

    for(int i = begin_row; offsets[i] < p_begin && i < end_row; ++i) {
        int end = (offsets[i + 1] < p_begin) ? offsets[i + 1] : p_begin;
        double v = 0.0;
        for(int p = offsets[i]; p < end; ++p) {
            v += values[p] * x[indices[p]];
        }
        y[i] += v;
    }

    for(p = p_begin, t = (p_begin - head) / 16; p < nz_end - 15; p += 16, ++t) {
        int mi = t * MASKS_PER_TILE, oi = t * VEC_PER_TILE;
        __m512d sl, sh, sl_old;
        __m512i idx = _mm512_loadu_si512(indices + p);
        sl = _mm512_mul_pd(_mm512_i32gather_pd(LOI(idx), x, 8), _mm512_load_pd(values + p));
        sh = _mm512_mul_pd(_mm512_i32gather_pd(HII(idx), x, 8), _mm512_load_pd(values + p + 8));

        sl_old = sl;
        sl = PACKD0246(sl, sh);
        sh = PACKD1357(sl_old, sh);
        sh = _mm512_mask_add_pd(sh, masks[mi], sl, sh);

        sl_old = sl;
        sl = SHUFFLED0246(sl, sh);
        sh = SHUFFLED1357(sl_old, sh);
        sh = _mm512_mask_add_pd(sh, masks[mi+1], BROADCASTD1357(sl), sh);

        sl_old = sl;
        sl = SHUFFLE2D0145(sl, sh);
        sh = SHUFFLE2D2367(sl_old, sh);
        sh = _mm512_mask_add_pd(sh, masks[mi+2], BROADCASTD37(sl), sh);
        
        sl_old = sl;
        sl = PACKD0123(sl, sh);
        sh = PACKD4567(sl_old, sh);
        sl = _mm512_mask_add_pd(sl, masks[mi+3], sp, sl);
        sh = _mm512_mask_add_pd(sh, masks[mi+4], BROADCASTD7(sl), sh);
        sp = BROADCASTD7(sh);

        sl = _mm512_add_pd(sl, _mm512_maskz_expandloadu_pd(masks[mi+5], y + y_offsets[oi]));
        sh = _mm512_add_pd(sh, _mm512_maskz_expandloadu_pd(masks[mi+6], y + y_offsets[oi+1]));
        
        _mm512_mask_compressstoreu_pd(y + y_offsets[oi  ], masks[mi+5], sl);
        _mm512_mask_compressstoreu_pd(y + y_offsets[oi+1], masks[mi+6], sh);
    }
    
    int i = y_offsets[t*2];
    double v = y[i] + _mm512_mask_reduce_add_pd((p == offsets[i]) ? 0 : 1, sp);
    for(; p < nz_end; ++p) {
        while(p == offsets[i + 1]) {
            y[i++] = v;
            v = y[i];
        }
        v += values[p] * x[indices[p]];
    }
    while(nz_end == offsets[i + 1]) {
        y[i++] = v;
        v = y[i];
    }
}
