# Installation instructions

Before installing Basix, you will need to install eigen3 and ninja. On Ubuntu, you can do this by
running `apt install libeigen3-dev ninja-build`; on macOS, you can use `brew install eigen ninja`.

## Installing Basix C++ library
To install Basix, you must first install the C++ library. You can do by running:

```bash
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -O3 -B build-dir -S .
cmake --build build-dir
cmake --install build-dir
```

You may need to use `sudo` for the final install line

## Installing the Python interface
If you want use Basix via Python, you will need to install the Basix Python interface. First, you
will need to install pybind11: `pip install pybind11`. You can then install the Basix Python
interface with:

```bash
cd python
pip install .
```

## Running the Basix tests
Once you have installed the Basix Python interface, you can run the Basix tests to check that everything
is running correctly. First, install pytest: `pip install pytest`. You can then run the tests with:

```bash
pytest test/
```
