CXX ?= g++
SRCS = calc.cc taskA.cc taskB.cc spmv_csr_kernels.c spmv_seg_sum_kernels.c spmv_special_kernels.c
LIBNAME = libcompetition.so
CXXOPTS = -O3 -DNDEBUG -fopenmp -march=native
#CXXOPTS = -O3 -DNDEBUG -march=native
#CXXOPTS = -O3 -DNDEBUG -fopenmp -funroll-loops
#CXXOPTS = -g -fopenmp -march=native
CXXOPTS += -std=c++17 -Wall -Wextra -Wconversion

CALCDEFS =
CALCDEFS += -DLONGROW_SIMD
CALCDEFS += -DFUSED_LOOP
CALCDEFS += -DALIGN_Y
CALCDEFS += -DUNROLL_32
#CALCDEFS += -DSCALAR_KERNELS

TESTDEFS =
TESTDEFS += -DBASELINE

all:$(OBJS)
	${CXX} ${CXXOPTS} ${CALCDEFS} ${SRCS} -shared -fPIC -o ${LIBNAME}

.PHONY: test clean

test: mytest.cc
	${CXX} ${CXXOPTS} ${TESTDEFS} -o mytest $< -ldl -L. -lcompetition -Wl,-rpath=.

naive: naive.cc
	${CXX} ${CXXOPTS} $< -shared -fPIC -o libnaive.so

clean:
	rm -f ${LIBNAME} mytest
