CXX ?= g++-7
SRCS = calc.cc taskA.cc taskB.cc
LIBNAME = libcompetition.so
CXXOPTS = -O3 -DNDEBUG -fopenmp -funroll-loops -march=native#-mavx -mavx2 -mfma
#CXXOPTS = -O3 -DNDEBUG -fopenmp -funroll-loops
CXXOPTS += -std=c++17 -Wall -Wextra -Wconversion

all:$(OBJS)
	${CXX} ${CXXOPTS} ${SRCS} -shared -fPIC -o ${LIBNAME}

clean:
	rm -f ${LIBNAME}
