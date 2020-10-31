CXX ?= g++
SRCS = calc.cc taskA.cc taskB.cc
LIBNAME = libcompetition.so
CXXOPTS = -O3 -DNDEBUG -fopenmp -funroll-loops -mavx512f
#CXXOPTS = -g -fopenmp
#CXXOPTS = -O2 -DNDEBUG -fopenmp
CXXOPTS += -std=c++17 -Wall -Wextra -Wconversion

all:$(OBJS)
	${CXX} ${CXXOPTS} ${SRCS} -shared -fPIC -o ${LIBNAME}

clean:
	rm -f ${LIBNAME}
