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

### External dependency metadata

Both `pyproject.toml` files declare non-Python build dependencies in an
`[external]` table using the draft [PEP 725](https://peps.python.org/pep-0725/)
format and dependency identifiers described by
[PEP 804](https://peps.python.org/pep-0804/). The root project declares CMake,
a C++ compiler, BLAS and LAPACK. The Python-only project instead declares
CMake, a C++ compiler and the installed Basix C++ library, identified by
`dep:github/fenics/basix`. The compiler must support C++20; CMake enforces
this requirement.

Libraries used for linking are listed under `host-requires`, which covers
dependencies needed at build time and potentially at runtime. This keeps
the metadata independent of the BLAS/LAPACK implementation and whether
shared libraries are bundled into a wheel.

These PEPs are still drafts. Tools supporting them can read the tables
from the source tree or source distribution; pip does not automatically
install these external dependencies. Package name mappings are maintained
outside Basix, and the Basix C++ dependency may need a mapping supplied by
the packaging ecosystem. Install external libraries and their development
headers before building from source. CMake can also be provided by
scikit-build-core's PyPI fallback.

### Python

Python build and runtime dependencies are fetched automatically by pip
when using the standard install approach. Building from source also
requires the external dependencies described above.

Basix specifies the optional extra `optional` for enabling optional
features, and the dependency groups `build`, `docs`, `lint`, `test`,
and `ci` for installing build dependencies, building documentation,
linting, testing and continuous integration, respectively, e.g.:

```console
pip install --group docs --group lint .
```

On `main`, `optional` and `test` need the development version of UFL,
so install it first:

```console
pip install git+https://github.com/FEniCS/ufl.git
```
