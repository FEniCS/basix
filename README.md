# Basix

![Basix CI](https://github.com/FEniCS/basix/workflows/Basix%20CI/badge.svg)

Basix is a finite element definition and tabulation runtime library.

## Supported elements

### Triangle
In Basix, the sub-entities of the reference triangle are numbered as follows:

![The numbering of a reference triangle](img/triangle_numbering.png)

Basix currently supports the following finite elements:

  - Lagrange (interval, triangle, tetrahedron, prism, pyramid, quadrilateral, hexahedron)
  - Nédélec (triangle, tetrahedron, quadrilateral, hexahedron)
  - Raviart-Thomas (triangle, tetrahedron, quadrilateral, hexahedron)
  - Nédélec Second Kind (triangle, tetrahedron)
  - Brezzi-Douglas-Marini (triangle, tetrahedron)
  - Regge (triangle, tetrahedron)
  - Crouzeix-Raviart (triangle, tetrahedron)
  - Bubble (interval, triangle, tetrahedron, quadrilateral, hexahedron)
  - DPC (interval, quadrilateral, hexahedron)
  - Serendipity (interval, quadrilateral, hexahedron)
