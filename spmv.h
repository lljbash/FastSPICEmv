#ifndef SPMV_H_
#define SPMV_H_

#include <stdint.h>
#include <immintrin.h>

// controls whether the addition (A = v0 + alpha * v1) is fused in the rowwise_simd taskB kernel
//#define FUSED_LOOP

// controls whether the access to y is aligned in the rowwise_simd and row_1 taskA kernels
//#define ALIGN_Y

// controls whether the loops in the unfused addition and long_row kernel are unrolled to 32
//#define UNROLL_32

// controls whether to compile the scalar kernels (no SIMD, only for debug)
//#define SCALAR_KERNELS

typedef struct {
    unsigned char *masks; // 5 seg masks, 2 write masks
    int *y_offsets;
    int head, n_tiles;
} segmentedsum_t;


/* rename symbols */
#define taskA                        FastSPICEmv_task_A
#define taskB                        FastSPICEmv_task_B
#define spmv_rowwise_taskA           FastSPICEmv_rw_s_A
#define spmv_rowwise_taskB           FastSPICEmv_rw_s_B
#define spmv_rowwise_simd_taskA      FastSPICEmv_rw_v_A
#define spmv_rowwise_simd_taskB      FastSPICEmv_rw_v_B
#define spmv_segmentedsum_taskA      FastSPICEmv_ss_s_A
#define spmv_segmentedsum_taskB      FastSPICEmv_ss_s_B
#define initialize_segmentedsum      FastSPICEmv_ss_v_I
#define finalize_segmentedsum        FastSPICEmv_ss_v_F
#define spmv_segmentedsum_simd_taskA FastSPICEmv_ss_v_A
#define spmv_segmentedsum_simd_taskB FastSPICEmv_ss_v_B
#define spmv_row_1_taskA             FastSPICEmv_r1_v_A
#define spmv_row_1_taskB             FastSPICEmv_r1_v_B
#define spmv_long_row_taskA          FastSPICEmv_lr_v_A
#define spmv_long_row_taskB          FastSPICEmv_lr_v_B

/* common macros, only for implementation */
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

/* x[0],x[2],x[4],x[6],y[0],y[2],y[4],y[6] */
#define PACKD0246(x, y)     _mm512_permutex2var_pd(x, idxl, y)
#define PACKD1357(x, y)     _mm512_permutex2var_pd(x, idxh, y)
#define PACKI0246(x, y)     _mm512_permutex2var_epi64(x, idxl, y)
#define PACKI1357(x, y)     _mm512_permutex2var_epi64(x, idxh, y)

/* x[0],y[0],x[2],y[2],x[4],y[4],x[6],y[6] */
/* valignq is faster than vunpcklpd, vunpckhpd, vshufpd and vshuff64x2, but uses an additional mask */
#define SHUFFLED0246(x, y)  _mm512_mask_movedup_pd(x, MASK1357, y)
#define SHUFFLED1357(x, y)  _mm512_mask_permute_pd(y, MASK0246, x, 0xff)
#define SHUFFLEI0246(x, y)  F2I(_mm512_mask_movedup_pd(I2F(x), MASK1357, I2F(y)))
#define SHUFFLEI1357(x, y)  F2I(_mm512_mask_permute_pd(I2F(y), MASK0246, I2F(x), 0xff))

/* x[0],x[1],y[0],y[1],x[4],x[5],y[4],y[5] */
#define SHUFFLE2D0145(x, y) _mm512_mask_permutex_pd(x, MASK2367, y, 0x44)
#define SHUFFLE2D2367(x, y) _mm512_mask_permutex_pd(y, MASK0145, x, 0xee)
#define SHUFFLE2I0145(x, y) _mm512_mask_permutex_epi64(x, MASK2367, y, 0x44)
#define SHUFFLE2I2367(x, y) _mm512_mask_permutex_epi64(y, MASK0145, x, 0xee)

/* x[0],x[1],x[2],x[3],y[0],y[1],y[2],y[3] */
#define PACKD0123(x, y)     I2F(_mm512_mask_alignr_epi64(F2I(x), MASK4567, F2I(y), F2I(y), 4))
#define PACKD4567(x, y)     I2F(_mm512_mask_alignr_epi64(F2I(y), MASK0123, F2I(x), F2I(x), 4))
#define PACKI0123(x, y)     _mm512_mask_alignr_epi64(x, MASK4567, y, y, 4)
#define PACKI4567(x, y)     _mm512_mask_alignr_epi64(y, MASK0123, x, x, 4)

#ifdef ALIGN_Y // aligned Id and IG
#define LOAD_PD(arg) _mm512_load_pd(arg)
#define STORE_PD(arg0, arg1) _mm512_store_pd(arg0, arg1)
#else
#define LOAD_PD(arg) _mm512_loadu_pd(arg)
#define STORE_PD(arg0, arg1) _mm512_storeu_pd(arg0, arg1)
#endif


#ifdef __cplusplus
extern "C" {
#endif

// for taskB
// IG and IC are the summed result, R and H are the substracted result
// D is the substracted vector, A is the summed matrix

// b : row begin, e : row end
// b <= the computed row indices < e
// for example, b = 10, e = 15 means 5 rows [10,11,12,13,14]

void spmv_rowwise_simd_taskA(int b, int e, const int *offsets, const int *indices, const double *values, const double *x, double *y);
void spmv_rowwise_simd_taskB(int b, int e, const int *offsets, const int *indices, const double *values, const double *x, \
    const double *D, double *IG, double *IC, double *R, double *H, double *A, double alpha);

int initialize_segmentedsum(segmentedsum_t *seg, int m, const int *offsets, const int *indices, const double *values);
void finalize_segmentedsum(segmentedsum_t *seg);
void spmv_segmentedsum_simd_taskA(segmentedsum_t *seg, \
    int b, int e, const int *offsets, const int *indices, const double *values, const double *x, double *y);
void spmv_segmentedsum_simd_taskB(segmentedsum_t *seg, \
    int begin_row, int end_row, const int *offsets, const int *indices, const double *values, const double *x, \
    const double *D, double *IG, double *IC, double *R, double *H, double *A, double alpha);

// 1 nonzero per row
void spmv_row_1_taskA(int b, int e, const int *offsets, const int *indices, const double *values, const double *x, double *y);
void spmv_row_1_taskB(int b, int e, const int *offsets, const int *indices, const double *values, const double *x, \
    const double *D, double *IG, double *IC, double *R, double *H, double *A, double alpha);

// long row
// row: the index of the long row
void spmv_long_row_taskA(int row, const int *offsets, const int *indices, const double *values, const double *x, double *y);
void spmv_long_row_taskB(int row, const int *offsets, const int *indices, const double *values, const double *x, \
    const double *D, double *IG, double *IC, double *R, double *H, double *A, double alpha);

#ifdef SCALAR_KERNELS
void spmv_rowwise_taskA(int b, int e, const int *offsets, const int *indices, const double *values, const double *x, double *y);
void spmv_rowwise_taskB(int b, int e, const int *offsets, const int *indices, const double *values, const double *x, \
    const double *D, double *IG, double *IC, double *R, double *H, double *A, double alpha);
    
void spmv_segmentedsum_taskA(int b, int e, const int *offsets, const int *indices, const double *values, const double *x, double *y);
void spmv_segmentedsum_taskB(int b, int e, const int *offsets, const int *indices, const double *values, const double *x, \
    const double *D, double *IG, double *IC, double *R, double *H, double *A, double alpha);
#endif

void taskA(
    int* rowArray,
    const int* rowOffset,
    int rowArraySize,
    const int* columnIndice,
    const double* S,
    const double* valueNormalMatrix,
    double* Id
);

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

#ifdef __cplusplus
}
#endif


#endif /* ifndef SPMV_H_ */
