# Basix C++ documentation

Welcome to the Basix C++ documentation.

Basix is a finite element definition and tabulation runtime library.
It is part of [FEniCSx](https://docs.fenicsproject.org),
alongside [FFCx](https://docs.fenicsproject.org/ffcx) and [DOLFINx](https://docs.fenicsproject.org/dolfinx/cpp).

Basix can create finite elements on intervals, triangles, quadrilaterals, tetrahedra, hexahedra, prisms, and pyramids.

### Using Basix
A Basix element can be created using the function `basix::create_element()`.
This function will return a `basix::FiniteElement` object.

The element can be tabulated using the function `basix::FiniteElement::tabulate()`.

### Table of contents
- [Index of namespaces](namespaces.html)
- [Index of classes](annotated.html)
- [Index of files](files.html)
