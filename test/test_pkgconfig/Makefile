CXX      = $(shell pkg-config --variable compiler basix)
CXXFLAGS = $(shell pkg-config --cflags basix)
LDLIBS   = $(shell pkg-config --libs basix)
LDFLAGS  = -Wl,-rpath,$(shell pkg-config --variable=libdir basix)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

main: main.o
	$(CXX) $(LDFLAGS) $< -o $@ $(LDLIBS)

run: main
	./main

clean:
	rm -f main main.o

.PHONY: run clean
