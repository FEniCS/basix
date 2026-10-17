# AGENTS.md

Guidance for AI coding agents working in the Basix repository. This
complements [CONTRIBUTING.md](CONTRIBUTING.md), which covers the PR process
and code of conduct.

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
- `python/basix/wrappers/basix_wrappers.cpp` — nanobind bindings exposing
  the C++ core to Python; the shared helpers and templates live alongside
  it in `python/basix/wrappers/basix_wrappers/basix_wrappers.h`. The header
  sits inside the Python package so that `wheel.packages` ships it;
  downstream nanobind wrappers locate it with `basix.get_include()`.
- `cpp/test/` — C++ unit tests (Catch2 + CTest), built as a standalone
  CMake project against an installed Basix.
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
  -Ccmake.build-type="Developer" -Cinstall.strip=false \
  --no-build-isolation -e .
```

(`--no-build-isolation` requires build dependencies to already be installed,
e.g. via `pip install --group build`; see `python/pyproject.toml`.)

C++ only:

```console
cd cpp
cmake -DCMAKE_BUILD_TYPE=Developer -B build-dir -S .
cmake --build build-dir
```

Requires a C++20 compiler, BLAS, and LAPACK.

Use `Developer` mode for correctness testing of both the C++ library and
the Python interface (`-Ccmake.build-type=Developer`). It enables compiler
warnings and standard-library checks where supported. Use `Release` for
performance profiling. After changing the C++ library in a split build,
rebuild and install it before rebuilding or testing the Python interface.

Dependency groups (`build`, `docs`, `lint`, `test`, `ci`) use PEP 735
syntax and require `pip >= 25.1` (or another PEP 735-compliant build
frontend) for the `--group` flag.

Python unit tests (from repo root, after installing with `pip install --group test .`):

```console
pytest test/
```

C++ unit tests (requires Catch2 3 and an installed Basix):

```console
cmake -DCMAKE_BUILD_TYPE=Developer -B build-dir-test -S cpp/test
cmake --build build-dir-test
ctest --test-dir build-dir-test --output-on-failure
```

The C++ library itself must also be built in `Developer` mode to enable
its checks; setting the test project's build type does not rebuild Basix.
On Windows with a multi-configuration generator, pass a configuration
matching the installed Basix to both the build (`--config Release`) and
CTest (`-C Release`).

## Linting and formatting

Run before opening a PR — CI enforces all of these:

```console
ruff format .          # Python formatting
ruff check .           # Python linting
gersemi --check .      # CMake file formatting (2-space indent, see .gersemirc)
```

Type checking uses [Pyrefly](https://pyrefly.org) (`[tool.pyrefly]` in the
repository-root `pyproject.toml`, run over `python/basix`, `demo/python`,
and `test`); it needs Basix built and installed first (it resolves
`basix`'s own compiled-extension types), so run it from an installed
environment, from the repository root:

```console
pip install pyrefly
pyrefly check
```

Pre-existing diagnostics are recorded in `pyrefly-baseline.json` and do not
fail CI; new diagnostics do. Regenerate the baseline after fixing or adding
to it with `pyrefly check --update-baseline`.

C++ formatting follows `.clang-format`; `clang-tidy` runs in CI with
`WarningsAsErrors: "*"` (performance checks only, see `.clang-tidy`).

Ruff config (`pyproject.toml`): line length 100, Google-style docstrings
(`D` rules apply outside `test/`, `demo/`, `doc/`).

## C++ style

- Use modern C++20, with concepts such as `std::floating_point` to
  constrain templates rather than SFINAE. Avoid C-style casts. Use
  `const`, `constexpr`, and `consteval` where appropriate.
- Prefer explicit types when they help readers understand a value's
  role. Avoid `auto` for simple types where it hides useful information;
  use it for structured bindings and complicated iterator types.
- Follow `.clang-format`: 2-space indentation, an 80-column limit, and
  Allman braces. Run `clang-format -i` on touched C++ source and header
  files, including nanobind wrappers.
- Braces may be omitted when both the control statement and its body
  each fit on one line. This also applies to nested loops satisfying
  the same condition. Brace bodies when the control statement or body
  spans multiple lines; judge each arm of an `if`/`else` separately.
- Use `#pragma once` in headers. Include the headers for the symbols
  used directly, including standard-library types, rather than relying
  on transitive includes. Let `clang-format` sort includes, keeping
  the paired header first in `.cpp` files.
- Library code lives in `basix` and its sub-namespaces. Follow the
  surrounding file's namespace conventions. Use `snake_case` for
  functions and variables, and a leading underscore for private data
  members; preserve existing public type and enum naming conventions.
- Use `std::size_t` for container sizes and extents. Preserve the `int`
  types used by existing APIs for degrees, derivative orders, entity
  indices, and DOF indices. Choose types by their role and avoid
  unnecessary changes to interface or storage types.
- Use `std::span` for contiguous array views and `basix::md::mdspan`
  or the surrounding code's mdspan aliases for multidimensional
  views. Use `std::string_view` for read-only string parameters when
  ownership is unnecessary. Document shapes and ordering of flat
  buffers; row-major storage is the usual convention.
- Prefer `std::ranges` algorithms when they improve clarity without
  hurting performance. For generic ranges, use `std::ranges::distance`
  and `std::ranges::advance` so dispatch follows C++20 iterator concepts.
  Pass a matching free function directly as a callback instead of
  wrapping it in a forwarding lambda.
- List lambda captures explicitly. Capture by reference only when the
  lambda cannot outlive the referenced data; avoid copying containers
  into captures. Move long or reused lambdas to named helper functions
  in an anonymous namespace.
- Avoid unnecessary copies of large buffers and element data. Use
  moves when transferring ownership, and check that cached views stay
  valid across moves. When declaring a special member function,
  explicitly consider all five copy/move/destructor operations; prefer
  defaulted implementations where appropriate.
- Use Doxygen `///` comments above public declarations, with
  `@param[in]`/`@param[in,out]`, `@return`, and `@note` as appropriate.
  Keep parameter names in sync across declarations, definitions, and
  documentation. Explain array shapes and mathematical conventions.
  Keep implementation comments concise and focused on non-obvious facts.
- Use descriptive standard exceptions for API errors and `assert` for
  internal invariants. Keep expensive checks in debug-only code and
  avoid introducing checks or exceptions inside hot tabulation loops.
  Follow the surrounding API's exception conventions.

## Python and nanobind style

- Follow the root `pyproject.toml` Ruff configuration: 4-space
  indentation, line length 100, and Google-style docstrings. Document
  public modules and APIs and provide public API type hints, including
  NumPy array dtypes where relevant.
- Let Ruff order imports: future, standard library, third-party,
  first-party (`basix`, `ufl`), then local imports.
- Keep the user-facing API in the pure-Python modules. New demos should
  use that API rather than importing `basix._basixcpp` directly.
- Bind functions with named nanobind arguments (`nb::arg` or the `_a`
  literal), checking their names and ordering against the C++ signature.
  Put Python-visible defaults in the pure-Python layer.
- Reuse the shared wrapper helpers in
  `python/basix/wrappers/basix_wrappers/basix_wrappers.h` for array
  conversion and ownership. Ensure returned NumPy arrays retain valid
  backing storage and expose the documented shape and dtype.

## CMake style

- Follow the minimum version in `cpp/CMakeLists.txt`; do not introduce
  newer CMake features without updating requirements or guarding them.
  Format touched CMake files with `gersemi`.
- Add library sources to `target_sources(basix PRIVATE ...)` and
  installed headers to `HEADERS_basix` in `cpp/CMakeLists.txt`.
- Use imported targets such as `Basix::basix` for consumers and keep
  build requirements on targets. Share Python installation discovery
  through `cmake/BasixPythonHints.cmake` rather than duplicating it.
- C++ tests and demos build against an installed Basix. Add new test
  source files to the `unittests` target in `cpp/test/CMakeLists.txt`;
  Catch2 test cases are discovered individually by CTest.

## Verifying changes locally

- Run the relevant formatters, linters, and affected tests before
  considering a change done. Use `ruff check .`, `ruff format --check .`,
  `gersemi --check .`, and `clang-format --dry-run --Werror` on touched
  C++ files to verify formatting without editing files.
- Check C++ changes with an actual build rather than relying on
  editor/clangd diagnostics, which may have incomplete include paths.
- Run the C++ unit tests for core changes and the Python tests for
  wrapper or public API changes. Rebuild and reinstall the library or
  extension as needed so tests exercise the changed code.
- Exercise affected demos with `pytest demo/python/test.py` and
  `pytest demo/cpp/test.py`. Demo text should explain the mathematics
  clearly and agree with the code and intended public API.

## Conventions

- `cpp/basix/mdspan.hpp` is a vendored third-party header (reference
  implementation of `std::mdspan`) — never edit it. It's also excluded from
  `clang-tidy` via `HeaderFilterRegex` in `.clang-tidy`.
- C++ source files use a `// Copyright (c) <years> <authors>` + FEniCS
  Project + `SPDX-License-Identifier: MIT` header block.
- Preserve existing copyright owners. AI agents must not add themselves,
  their model names, or their providers as copyright owners. Python files
  use the corresponding `#` comment header followed by a module docstring.
- New finite elements go in `cpp/basix/e-<name>.{cpp,h}` and are wired into
  `element-families.h` and the Python/nanobind wrapper as needed.
- License: MIT.

## Useful references

- `INSTALL.md` — detailed build/install/test instructions.
- `CONTRIBUTING.md` — PR process and linting expectations.
- `.github/workflows/pythonapp.yml` — canonical CI steps (lint, build, test,
  docs) — the most reliable source of exact commands.
- Issue tracker: <https://github.com/FEniCS/basix/issues>
