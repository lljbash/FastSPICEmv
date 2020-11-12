CXX ?= g++-7
SRCS = calc.cc taskA.cc taskB.cc
LIBNAME = libcompetition.so
CXXOPTS = -O3 -DNDEBUG -fopenmp -march=native#-mavx -mavx2 -mfma
#CXXOPTS = -O3 -DNDEBUG -march=native
#CXXOPTS = -O3 -DNDEBUG -fopenmp -funroll-loops
#CXXOPTS = -g
CXXOPTS += -std=c++17 -Wall -Wextra -Wconversion

all:$(OBJS)
	${CXX} ${CXXOPTS} ${SRCS} -shared -fPIC -o ${LIBNAME}

.PHONY: test clean

test: mytest.cc
	${CXX} ${CXXOPTS} -o mytest $< -ldl -L. -lcompetition -Wl,-rpath=.

clean:
	rm -f ${LIBNAME} mytest
