# Installation

## Standard

Basix can be installed using

```console
pip install .
```

## Advanced

In the standard install, the C++ library is built and installed inside the
Python package.  This method is suitable for the majority of use cases.

It is also possible to install the C++ and Python interfaces separately
(see below). This is useful if you only need the C++ interface, and can
be helpful during development.

### C++ library

In the `cpp/` directory:

```console
cmake -DCMAKE_BUILD_TYPE=Release -B build-dir -S .
cmake --build build-dir
cmake --install build-dir
```

You may need to use `sudo` for the final install step. Using the CMake
build type `Release` is strongly recommended for performance.


### Python interface

After installing the C++ library, install the Python interface by running in
the directory `python/`:

```console
pip install .
```

## Running the unit tests

To install Basix and the extra dependencies required to run the Python unit tests:

```console
pip install .[test]
```

From the directory `python/` the tests can be run with:

```console
pytest test/
```

## Dependencies

### C++

Basix requires a C++20 compiler and depends on BLAS and LAPACK.

### Python

When using the standard install approach all build and runtime
dependencies for the C++ and Python parts of Basix will be fetched
automatically.

Building the Python interface requires
[`nanobind`](https://github.com/wjakob/nanobind).

At runtime Basix requires [`numpy`](https://numpy.org).

Basix specifies sets of optional extras `docs`, `lint`, `optional`, `test`, and
`ci` for building documentation, linting, enabling optional features, testing
and for continuous integration, respectively, e.g.:

```console
pip install .[docs,lint]
```
