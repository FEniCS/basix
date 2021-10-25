Basix documentation
===================

C++ documentation
-----------------
To build the documentation of the Basix C++ library, run the following commands:

```bash
cd cpp
doxygen
```

The documentation will then be built in the folder `cpp/html/`.


Python documentation
--------------------
To build the documentation of the Basix Python interface, run the following commands:

```bash
cd python
make html
```

The documentation will then be built in the folder `python/html/`.


Documentation for website
-------------------------
To build the Basix documentation as it appears at
[docs.fenicsproject.org/basix/main/](https://docs.fenicsproject.org/basix/main/),
run the following commands:

```bash
cd web
python3 make_html.py
```

The documentation will then be built in the folder `web/html/`.
