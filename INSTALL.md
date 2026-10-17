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

### nanobind split mode (optional)

The Python extension can be built in [nanobind split
mode](https://nanobind.readthedocs.io/en/latest/split_mode.html) by setting
`BASIX_NANOBIND_SPLIT=true` in the environment, for either the combined install
from the repository root or the separate Python install from `python/`:

```console
BASIX_NANOBIND_SPLIT=true pip install .
```

It requires nanobind >= 3.0 and adds `nanobind-backend>=1.0` as a build and
runtime dependency. When using `--no-build-isolation`, install these two
dependencies first. Split mode produces a `cp311-abi3` wheel usable on
supported CPython versions from 3.11 onwards. It is off by default and enabled
for the wheels built by `.github/workflows/build-wheels.yml`. Without split
mode, the existing interpreter-specific and Python 3.12 stable ABI builds
remain available.

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

The C++ unit tests in `cpp/test/` are built against an installed Basix
and require [Catch2](https://github.com/catchorg/Catch2) 3. From the
repository root:

```console
cmake -DCMAKE_BUILD_TYPE=Developer -B build-dir-test -S cpp/test
cmake --build build-dir-test
ctest --test-dir build-dir-test --output-on-failure
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
