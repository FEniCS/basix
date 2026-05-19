Basix documentation
===================

C++ documentation
-----------------
To build the documentation of the Basix C++ library, run the following
commands:

```bash
cd cpp
doxygen
```

The documentation will then be built in the folder `cpp/html`.


Python documentation
--------------------
To build the documentation of the Basix Python interface, run the
following commands:

```bash
cd python
python -m sphinx -W -b html source/ build/html/
```

The documentation will then be built in the folder `python/build/html`.
