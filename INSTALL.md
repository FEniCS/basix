# Installation instructions

## Installing the Basix C++ library

To install Basix, you must first install the C++ interface and library:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -B build-dir -S .
cmake --build build-dir
cmake --install build-dir
```

You may need to use `sudo` for the final install step. Using the CMake
build type `Release` is strongly recommended for performance.


## Installing the Python interface

To use Basix via Python, you will need to install the Basix Python
interface. First, you will need to install pybind11: `pip install
pybind11`. You can then install the Basix Python interface with:

```bash
cd python
pip install .
```


## Running the Basix tests

Once you have installed the Basix Python interface, you can run the
Basix tests to check that everything is running correctly. First,
install pytest: `pip install pytest`. You can then run the tests with:

```bash
pytest test/
```


## Dependencies

Basix depends on [`xtensor`](https://github.com/xtensor-stack/xtensor)
and [`xtensor-blass`](https://github.com/xtensor-stack/xtensor-blas).
CMake will download install these packages if they cannot be found.

Building the Python interface required
[`pybind11`](https://github.com/pybind/pybind11).