CXX           = $(shell pkg-config --variable compiler basix)
CXXFLAGS      = $(shell pkg-config --cflags basix)
LDFLAGS       = -Wl,-rpath,$(shell pkg-config --variable=libdir basix)
LDLIBS_SHARED = $(shell pkg-config --libs basix)
# --static also pulls in Libs.private (BLAS/LAPACK), needed only for a
# static basix.
LDLIBS_STATIC = $(shell pkg-config --static --libs basix)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

main-shared: main.o
	$(CXX) $(LDFLAGS) $< -o $@ $(LDLIBS_SHARED)

main-static: main.o
	$(CXX) $(LDFLAGS) $< -o $@ $(LDLIBS_STATIC)

run-shared: main-shared
	./main-shared

run-static: main-static
	./main-static

clean:
	rm -f main-shared main-static main.o

.PHONY: run-shared run-static clean
