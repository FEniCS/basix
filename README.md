# basix

## Finite Element Basis Function Definition Runtime Library

* computes FE basis functions and derivatives for the following elements:
  - Lagrange (interval, triangle, tetrahedron, prism, pyramid, quadrilateral, hexahedron)
  - Nédélec (triangle, tetrahedron)
  - Nédélec Second Kind (triangle, tetrahedron)
  - Raviart-Thomas (triangle, tetrahedron)
  - Regge (triangle, tetrahedron)
  - Crouzeix-Raviart (triangle, tetrahedron)

* computes quadrature rules on different cell types
* provides reference topology and geometry for reference cells of each type
* python wrapper with pybind11


![Basix CI](https://github.com/FEniCS/basix/workflows/Basix%20CI/badge.svg)
