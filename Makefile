CXX ?= g++
SRCS = calc.cc taskA.cc taskB.cc spmv_csr_kernels.c  spmv_special_kernels.c #spmv_seg_sum_kernels.c
NAIVE_SIMD_SRCS = naive_simd.cc taskA.cc taskB.cc spmv_csr_kernels.c  spmv_special_kernels.c
LIBNAME = libcompetition.so
CXXOPTS = -O3 -g -DNDEBUG -fopenmp -mavx512f -mavx512dq -mavx512vl #-march=native
#CXXOPTS = -O3 -DNDEBUG -march=native
#CXXOPTS = -O3 -DNDEBUG -fopenmp -funroll-loops
#CXXOPTS = -g -fopenmp -march=native
CXXOPTS += -std=c++17 -Wall -Wextra -Wconversion

CALCDEFS =
#CALCDEFS += -DLONGROW_SIMD
#CALCDEFS += -DFUSED_LOOP
#CALCDEFS += -DALIGN_Y
#CALCDEFS += -DUNROLL_32
CALCDEFS += -DCPP17_ALIGNED_NEW
#CALCDEFS += -DSCALAR_KERNELS

TESTDEFS =
#TESTDEFS += -DBASELINE -DBASELINE_TEST

all:
	${CXX} ${CXXOPTS} ${CALCDEFS} ${SRCS} -shared -fPIC -o ${LIBNAME}

.PHONY: test clean

test: mytest.cc
	${CXX} ${CXXOPTS} ${TESTDEFS} -o mytest $< -ldl -L. -lcompetition -Wl,-rpath=.

naive: naive.cc
	${CXX} ${CXXOPTS} $< -shared -fPIC -o libnaive.so

naive_simd:
	${CXX} ${CXXOPTS} ${CALCDEFS} ${NAIVE_SIMD_SRCS} -shared -fPIC -o ${LIBNAME}

clean:
	rm -f ${LIBNAME} mytest libnaive.so
