#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "cell.h"
#include "crouzeix-raviart.h"
#include "indexing.h"
#include "lagrange.h"
#include "nedelec-second-kind.h"
#include "nedelec.h"
#include "polynomial-set.h"
#include "quadrature.h"
#include "raviart-thomas.h"
#include "regge.h"
#include "tp.h"

namespace py = pybind11;
using namespace libtab;

const std::string tabdoc = R"(
Tabulate the finite element basis function and derivatives at points.
If no derivatives are required, use nderiv=0. In 2D and 3D, the derivatives are ordered
in triangular (or tetrahedral) order, i.e. (0,0),(1,0),(0,1),(2,0),(1,1),(0,2) etc. in 2D.

Parameters
==========
nderiv : int
    Number of derivatives required

points : numpy.ndarray
    Array of points

Returns
=======
List[numpy.ndarray]
    List of basis values and derivatives at points. Returns a list of length `(n+d)!/(n!d!)`
    where `n` is the number of derivatives and `d` is the topological dimension.
)";

PYBIND11_MODULE(libtab, m)
{
  m.doc() = R"(
Libtab provides information about finite elements on the reference cell. It has support for
interval (1D), triangle and quadrilateral (2D), and tetrahedron, hexahedron, prism and pyramid (3D) reference cells.
Elements are available in several different types, typically as `ElementName(celltype, degree)`. Not all elements are available
on all cell types, and an error should be thrown if an invalid combination is requested.
Each element has a `tabulate` function which returns the basis functions and a number of their derivatives, as desired.

)";

  py::enum_<cell::Type>(m, "CellType")
      .value("interval", cell::Type::interval)
      .value("triangle", cell::Type::triangle)
      .value("tetrahedron", cell::Type::tetrahedron)
      .value("quadrilateral", cell::Type::quadrilateral)
      .value("hexahedron", cell::Type::hexahedron)
      .value("prism", cell::Type::prism)
      .value("pyramid", cell::Type::pyramid);

  m.def("topology", &cell::topology);
  m.def("geometry", &cell::geometry);
  m.def("sub_entity_geometry", &cell::sub_entity_geometry);

  m.def("simplex_type", &cell::simplex_type,
        "Simplex CellType of given dimension");
  m.def("create_lattice", &cell::create_lattice,
        "Create a uniform lattice of points on a reference cell");

  m.def("create_new_element",
        [](cell::Type celltype, int degree, int value_size,
           const Eigen::MatrixXd& dualmat,
           const Eigen::MatrixXd& coeffs) -> FiniteElement {
          auto new_coeffs
              = FiniteElement::apply_dualmat_to_basis(coeffs, dualmat);
          return FiniteElement(celltype, degree, value_size, new_coeffs);
        });

  py::class_<FiniteElement>(m, "FiniteElement", "Finite Element")
      .def("tabulate", &FiniteElement::tabulate, tabdoc.c_str())
      .def_property_readonly("degree", &FiniteElement::degree)
      .def_property_readonly("cell_type", &FiniteElement::cell_type)
      .def_property_readonly("value_size", &FiniteElement::value_size);

  // Create FiniteElement of different types
  m.def("Nedelec", &Nedelec::create, "Create Nedelec Element (first kind)");
  m.def("Lagrange", &Lagrange::create, "Create Lagrange Element");
  m.def("CrouzeixRaviart", &CrouzeixRaviart::create,
        "Create Crouzeix-Raviart Element");
  m.def("TensorProduct", &TensorProduct::create,
        "Create TensorProduct Element");
  m.def("RaviartThomas", &RaviartThomas::create,
        "Create Raviart-Thomas Element");
  m.def("NedelecSecondKind", &NedelecSecondKind::create,
        "Create Nedelec Element (second kind)");
  m.def("Regge", &Regge::create, "Create Regge Element");

  m.def("tabulate_polynomial_set", &polyset::tabulate,
        "Tabulate orthonormal polynomial expansion set");

  m.def("compute_jacobi_deriv", &quadrature::compute_jacobi_deriv,
        "Compute jacobi polynomial and derivatives at points");

  m.def("make_quadrature",
        py::overload_cast<const Eigen::Array<double, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>&,
                          int>(&quadrature::make_quadrature),
        "Compute quadrature points and weights on a simplex defined by points");

  m.def("index", py::overload_cast<int, int>(&libtab::idx),
        "Indexing for triangular arrays")
      .def("index", py::overload_cast<int, int, int>(&libtab::idx),
           "Indexing for tetrahedral arrays");
}
