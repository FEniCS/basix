# Installation

## Standard

Basix can be installed using
```console
pip install .
```

## Advanced

It is also possible to install the C++ and Python interfaces separately
(see below). This is useful if you only need the C++ interface or during
development.

### C++ library

In the `cpp/` directory:
```console
cmake -DCMAKE_BUILD_TYPE=Release -B build-dir -S .
cmake --build build-dir
cmake --install build-dir
```
Using the CMake build type `Release` is strongly recommended for
performance.


### Python interface

After installing the C++ library, install the Python interface by
running in the directory `python/`:
```console
pip install .
```

For a debug and editable build for development:
```console
pip -v install --check-build-dependencies --config-settings=build-dir="build" --config-settings=cmake.build-type="Debug"  --config-settings=install.strip=false --no-build-isolation -e .
```
When using the `--no-build-isolation` option all build and runtime
dependencies must already be installed.

## Running the unit tests

To install Basix and the extra dependencies required to run the Python
unit tests:

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
dependencies for the C++ and Python parts of Basix are fetched
automatically.

Basix specifies sets of optional extras `docs`, `lint`, `optional`,
`test`, and `ci` for building documentation, linting, enabling optional
features, testing and for continuous integration, respectively, e.g.:
```console
pip install .[docs,lint]
```
