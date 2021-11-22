# Installation

The Basix Python and C++ can be installed by

```console
pip install .
```

It is also possible to install the C++ and Python interfaces separately
(see below). This is useful if you only need the C++ interface, and can
be helpful during development.

## Advanced

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

To install the Python interface, in the directory `python/`:

```console
pip install .
```


## Running the unit tests

Once the Basix Python interface has been installed, from the 
directory `python/` the tests can be run with:

```console
pytest test/
```


## Dependencies

Basix depends on [`xtensor`](https://github.com/xtensor-stack/xtensor).
CMake will download install these packages if they cannot be found.

Building the Python interface requires
[`pybind11`](https://github.com/pybind/pybind11).
