# Installation

## Standard

Basix can be installed using:
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

Using the CMake build type `Release` or `RelWithDebInfo` is strongly recommended
for performance.


### Python interface

After installing the C++ library, install the Python interface by
running in the directory `python/`:
```console
pip install .
```

For a debug and editable build for development:
```console
pip -v install --check-build-dependencies -Cbuild-dir="build" -Ccmake.build-type="Developer" -Cinstall.strip=false --no-build-isolation -e .
```
When using the `--no-build-isolation` option all build dependencies must
already be installed, e.g. via `pip install --group build`
(see `python/pyproject.toml`).

RPATH manipulation can be disabled by passing
`-Ccmake.args=-DBASIX_SET_INSTALL_RPATH=FALSE`.

## Running the unit tests

To install Basix and the extra dependencies required to run the Python
unit tests:

```console
pip install --group test .
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

Basix specifies the optional extra `optional` for enabling optional
features, and the dependency groups `build`, `docs`, `lint`, `test`,
and `ci` for installing build dependencies, building documentation,
linting, testing and continuous integration, respectively, e.g.:
```console
pip install --group docs --group lint .
```
