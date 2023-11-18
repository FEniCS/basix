# Basix

[![Basix CI](https://github.com/FEniCS/basix/actions/workflows/pythonapp.yml/badge.svg)](https://github.com/FEniCS/basix/actions/workflows/pythonapp.yml)
[![Spack install](https://github.com/FEniCS/basix/actions/workflows/spack.yml/badge.svg)](https://github.com/FEniCS/basix/actions/workflows/spack.yml)

Basix is a finite element definition and tabulation runtime library.
Basix allows users to:

- evaluate finite element basis functions and their derivatives at a set
  of points;
- access geometric and topological information about reference cells;
- apply push forward and pull back operations to map data between a
  reference cell and a physical cell;
- permute and transform DOFs to allow higher-order elements to be use on
  arbitrary meshes; and
- interpolate into and between finite element spaces.

Basix includes a range of built-in elements, and also allows the user to
define their own custom elements.

Basix is one of the components of FEniCSx, alongside
[UFL](https://github.com/fenics/ufl),
[FFCx](https://github.com/fenics/ffcx), and
[DOLFINx](https://github.com/fenics/dolfinx).


## Installation

To install Basix:
```console
pip install fenics-basix
```
We currently build binary wheels for Linux and macOS x86-64 architectures.

For advanced and developer installation instructions see the more
detailed [install instructions](INSTALL.md)


## Documentation

Documentation of Basix can be found at
https://docs.fenicsproject.org/basix/main/.


## Support

If you find a bug in Basix, you can report it on the [GitHub issue
tracker](https://github.com/fenics/basix/issues/new?labels=bug).

Questions about using Basix can be asked on the [FEniCS discourse
group](https://fenicsproject.discourse.group/).


## Contributing

Information about how to contribute to Basix can be found
[here](CONTRIBUTING.md).

## Supported elements

### Interval

In Basix, the sub-entities of the reference interval are numbered as
follows:

![The numbering of a reference interval](joss/img/interval_numbering.png)

The following elements are supported on an interval:

  - [Lagrange](https://defelement.com/elements/lagrange.html)
  - [Bubble](https://defelement.com/elements/bubble.html)
  - [Serendipity](https://defelement.com/elements/serendipity.html)
  - [Hermite](https://defelement.com/elements/hermite.html)
  - [iso](https://defelement.com/elements/p1-iso-p2.html)


### Triangle

In Basix, the sub-entities of the reference triangle are numbered as
follows:

![The numbering of a reference triangle](joss/img/triangle_numbering.png)

The following elements are supported on a triangle:

  - [Lagrange](https://defelement.com/elements/lagrange.html)
  - [Nédélec first kind](https://defelement.com/elements/nedelec1.html)
  - [Raviart-Thomas](https://defelement.com/elements/raviart-thomas.html)
  - [Nédélec second kind](https://defelement.com/elements/nedelec2.html)
  - [Brezzi-Douglas-Marini](https://defelement.com/elements/brezzi-douglas-marini.html)
  - [Regge](https://defelement.com/elements/regge.html)
  - [Hellan-Herrmann-Johnson](https://defelement.com/elements/hellan-hermann-johnson.html)
  - [Crouzeix-Raviart](https://defelement.com/elements/crouzeix-raviart.html)
  - [Bubble](https://defelement.com/elements/bubble.html)
  - [Hermite](https://defelement.com/elements/hermite.html)
  - [iso](https://defelement.com/elements/p1-iso-p2.html)


### Quadrilateral

In Basix, the sub-entities of the reference quadrilateral are numbered
as follows:

![The numbering of a reference quadrilateral](joss/img/quadrilateral_numbering.png)

The following elements are supported on a quadrilateral:

  - [Lagrange](https://defelement.com/elements/lagrange.html)
  - [Nédélec first kind](https://defelement.com/elements/nedelec1.html)
  - [Raviart-Thomas](https://defelement.com/elements/qdiv.html)
  - [Nédélec second kind](https://defelement.com/elements/scurl.html)
  - [Brezzi-Douglas-Marini](https://defelement.com/elements/sdiv.html)
  - [Bubble](https://defelement.com/elements/bubble.html)
  - [DPC](https://defelement.com/elements/dpc.html)
  - [Serendipity](https://defelement.com/elements/serendipity.html)
  - [iso](https://defelement.com/elements/p1-iso-p2.html)


### Tetrahedron

In Basix, the sub-entities of the reference tetrahedron are numbered as
follows:

![The numbering of a reference tetrahedron](joss/img/tetrahedron_numbering.png)

The following elements are supported on a tetrahedron:

  - [Lagrange](https://defelement.com/elements/lagrange.html)
  - [Nédélec first kind](https://defelement.com/elements/nedelec1.html)
  - [Raviart-Thomas](https://defelement.com/elements/raviart-thomas.html)
  - [Nédélec second kind](https://defelement.com/elements/nedelec2.html)
  - [Brezzi-Douglas-Marini](https://defelement.com/elements/brezzi-douglas-marini.html)
  - [Regge](https://defelement.com/elements/regge.html)
  - [Crouzeix-Raviart](https://defelement.com/elements/crouzeix-raviart.html)
  - [Bubble](https://defelement.com/elements/bubble.html)
  - [Hermite](https://defelement.com/elements/hermite.html)
  - [iso](https://defelement.com/elements/p1-iso-p2.html)


### Hexahedron

In Basix, the sub-entities of the reference hexahedron are numbered as
follows:

![The numbering of a reference hexahedron](joss/img/hexahedron_numbering.png)

The following elements are supported on a hexahedron:

  - [Lagrange](https://defelement.com/elements/lagrange.html)
  - [Nédélec first kind](https://defelement.com/elements/nedelec1.html)
  - [Raviart-Thomas](https://defelement.com/elements/qdiv.html)
  - [Nédélec second kind](https://defelement.com/elements/scurl.html)
  - [Brezzi-Douglas-Marini](https://defelement.com/elements/sdiv.html)
  - [Bubble](https://defelement.com/elements/bubble.html)
  - [DPC](https://defelement.com/elements/dpc.html)
  - [Serendipity](https://defelement.com/elements/serendipity.html)
  - [iso](https://defelement.com/elements/p1-iso-p2.html)


### Prism

In Basix, the sub-entities of the reference prism are numbered as
follows:

![The numbering of a reference prism](joss/img/prism_numbering.png)

The following elements are supported on a prism:

  - [Lagrange](https://defelement.com/elements/lagrange.html)


### Pyramid

In Basix, the sub-entities of the reference pyramid are numbered as
follows:

![The numbering of a reference pyramid](joss/img/pyramid_numbering.png)

The following elements are supported on a pyramid:

  - [Lagrange](https://defelement.com/elements/lagrange.html)
