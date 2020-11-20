CXX ?= g++
SRCS = calc_tbb.cc spmv_discrete_kernels.c spmv_csr_kernels.c spmv_special_kernels.c
LIBNAME = libcompetition.so
CXXOPTS = -O3 -DNDEBUG -mavx512f -mavx512dq -mavx512vl #-march=native
#CXXOPTS = -O3 -DNDEBUG -march=native
#CXXOPTS = -O3 -DNDEBUG -fopenmp -funroll-loops
#CXXOPTS = -g -fopenmp -march=native
CXXOPTS += -std=c++17 -Wall -Wextra #-Wconversion

CALCDEFS =
CALCDEFS += -DLONGROW_SIMD
CALCDEFS += -DTHREAD_LIMIT=32
#CALCDEFS += -DFUSED_LOOP
#CALCDEFS += -DALIGN_Y
#CALCDEFS += -DUNROLL_32
#CALCDEFS += -DCPP17_ALIGNED_NEW
#CALCDEFS += -DSCALAR_KERNELS

TESTDEFS =
TESTDEFS += -DBASELINE
TESTDEFS += -DBASELINE_TEST

all: ${LIBNAME} mytest libnaive.so

${LIBNAME}:$(OBJS)
	${CXX} ${CXXOPTS} ${CALCDEFS} ${SRCS} -shared -fPIC -o ${LIBNAME} -ltbb

.PHONY: test clean

mytest: mytest.cc
	${CXX} ${CXXOPTS} ${TESTDEFS} -o $@ $< -ldl -ltbb -L. -lcompetition -Wl,-rpath=.

libnaive.so: naive.cc
	${CXX} ${CXXOPTS} -fopenmp $< -shared -fPIC -o $@

clean:
	rm -f ${LIBNAME} mytest libnaive.so
