# AGENTS.md

Guidance for AI coding agents working in the Basix repository.

## What this project is

Basix is a finite element definition and tabulation runtime library. It has
a C++ core (`cpp/`) and a Python interface (`python/`) built with nanobind,
and is one of the FEniCSx components (alongside UFL, FFCx, DOLFINx).

Basix follows the same design and coding style as
[DOLFINx](https://github.com/FEniCS/dolfinx) — when in doubt about a style
or API design choice, prefer consistency with DOLFINx conventions over
introducing a new pattern.

## Repository layout

- `cpp/basix/` — C++ core library (element definitions `e-*.cpp/h`, cell
  geometry, quadrature, polysets, interpolation, dof transformations, etc.).
- `cpp/CMakeLists.txt` — C++ library build.
- `python/basix/` — Python package (thin wrappers around the C++ core).
- `python/wrapper.cpp` — nanobind bindings exposing the C++ core to Python.
- `test/` — Python unit tests (pytest), plus `test/test_cmake` and
  `test/test_pkgconfig` integration tests for the installed C++ library.
- `demo/` — C++ and Python demos, also exercised in CI.
- `doc/` — Doxygen (C++) and Sphinx (Python) documentation sources.

## Build and test

Standard install (builds C++ and Python together):

```console
pip install .
```

Editable/development install:

```console
cd python
pip -v install --check-build-dependencies -Cbuild-dir="build" \
  -Ccmake.build-type="Development" -Cinstall.strip=false \
  --no-build-isolation -e .
```

(`--no-build-isolation` requires build dependencies to already be installed,
e.g. via `pip install --group build`; see `python/pyproject.toml`.)

C++ only:

```console
cd cpp
cmake -DCMAKE_BUILD_TYPE=Release -B build-dir -S .
cmake --build build-dir
```

Requires a C++20 compiler, BLAS, and LAPACK.

Dependency groups (`build`, `docs`, `lint`, `test`, `ci`) use PEP 735
syntax and require `pip >= 25.1` (or another PEP 735-compliant build
frontend) for the `--group` flag.

Python unit tests (from repo root, after installing with `pip install --group test .`):

```console
pytest test/
```

## Linting and formatting

Run before opening a PR — CI enforces all of these:

```console
ruff format .          # Python formatting
ruff check .           # Python linting
mypy python/basix      # type checking
mypy demo/python
gersemi --check .      # CMake file formatting (2-space indent, see .gersemirc)
```

C++ formatting follows `.clang-format`; `clang-tidy` runs in CI with
`WarningsAsErrors: "*"` (performance checks only, see `.clang-tidy`).

Ruff config (`pyproject.toml`): line length 100, Google-style docstrings
(`D` rules apply outside `test/`, `demo/`, `doc/`).

## Conventions

- `cpp/basix/mdspan.hpp` is a vendored third-party header (reference
  implementation of `std::mdspan`) — never edit it. It's also excluded from
  `clang-tidy` via `HeaderFilterRegex` in `.clang-tidy`.
- C++ source files use a `// Copyright (c) <years> <authors>` + FEniCS
  Project + `SPDX-License-Identifier: MIT` header block.
- New finite elements go in `cpp/basix/e-<name>.{cpp,h}` and are wired into
  `element-families.h` and the Python/nanobind wrapper as needed.
- License: MIT.

## Useful references

- `INSTALL.md` — detailed build/install/test instructions.
- `CONTRIBUTING.md` — PR process and linting expectations.
- `.github/workflows/pythonapp.yml` — canonical CI steps (lint, build, test,
  docs) — the most reliable source of exact commands.
- Issue tracker: <https://github.com/FEniCS/basix/issues>
