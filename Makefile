CXXFLAGS=-I/usr/include/eigen3 -O3 -std=c++17
OBJS = main.o tabulate.o

tabulate.o: tabulate.cpp polyn.h
main.o: main.cpp

all:: tabulate
tabulate:: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)
