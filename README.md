# libtab

## Finite Element Tabulation Runtime Library

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


![Libtab CI](https://github.com/FEniCS/libtab/workflows/Libtab%20CI/badge.svg)
