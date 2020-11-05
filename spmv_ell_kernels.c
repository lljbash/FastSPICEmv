#include <stdint.h>
#include <immintrin.h>
#include "spmv.h"

void initialize_ell(ell_t *ell, int m, const int *offsets, const int *indices, const double *values) {

}

void finalize_ell(ell_t *ell) {
    
}

void spmv_ell_simd(ell_t *ell, int b, int e, const int *offsets, const int *indices, const double *values, const double *x, double *y) {


}


