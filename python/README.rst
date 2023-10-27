Basix Python interface
========================

This document explains how to install the Basix Python interface. Note that
this interface must be built without PEP517 build isolation by passing
`--no-build-isolation` flag to `pip install`.

1. Build and install the Basix C++ library in the usual way.

2. Ensure the build time requirements are installed::

     python3 -m pip -v install -r build-requirements.txt

3. Build Basix Python interface::

    python3 -m pip -v install --no-build-isolation .

   To build in debug and editable mode for development::

    python3 -m pip -v install --config-settings=build-dir="build" --config-settings=cmake.build-type="Debug" --no-build-isolation -e .
