// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "tp.h"
#include "polynomial-set.h"
#include <Eigen/Dense>

using namespace libtab;

TensorProduct::TensorProduct(Cell::Type celltype, int degree)
    : FiniteElement(celltype, degree)
{
  if (celltype != Cell::Type::quadrilateral and celltype != Cell::Type::prism
      and celltype != Cell::Type::pyramid
      and celltype != Cell::Type::hexahedron)
    throw std::runtime_error("Invalid celltype");

  // Tabulate basis at nodes
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pt
      = Cell::create_lattice(celltype, degree, true);

  Eigen::MatrixXd dualmat = PolynomialSet::tabulate(celltype, degree, 0, pt)[0];

  int ndofs = pt.rows();
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(ndofs, ndofs);
  apply_dualmat_to_basis(coeffs, dualmat);
}
//-----------------------------------------------------------------------------
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
TensorProduct::tabulate(
    int nderiv,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts) const
{
  const int tdim = Cell::topological_dimension(_cell_type);
  if (pts.cols() != tdim)
    throw std::runtime_error(
        "Point dimension does not match element dimension");

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dbasis_at_pts = PolynomialSet::tabulate(_cell_type, _degree, nderiv, pts);

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dresult(dbasis_at_pts.size());

  for (std::size_t p = 0; p < dresult.size(); ++p)
    dresult[p] = dbasis_at_pts[p].matrix() * _coeffs.transpose();

  return dresult;
}
