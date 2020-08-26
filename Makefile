CXXFLAGS=-I/usr/include/eigen3 -O3 -std=c++17
OBJS = main.o tabulate.o

all:: tabulate
tabulate:: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)
