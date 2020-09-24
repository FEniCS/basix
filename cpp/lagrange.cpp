// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "lagrange.h"
#include "polynomial-set.h"
#include <Eigen/Dense>

Lagrange::Lagrange(Cell::Type celltype, int degree)
    : FiniteElement(celltype, degree)
{
  if (celltype != Cell::Type::interval and celltype != Cell::Type::triangle
      and celltype != Cell::Type::tetrahedron)
    throw std::runtime_error("Invalid celltype");

  // Tabulate basis at nodes
  Cell c(celltype);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pt
      = c.create_lattice(degree, true);

  Eigen::MatrixXd dualmat
      = PolynomialSet::tabulate_polynomial_set(celltype, degree, pt);

  const int ndofs = pt.rows();
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(ndofs, ndofs);
  apply_dualmat_to_basis(coeffs, dualmat);
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Lagrange::tabulate_basis(
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts) const
{
  const int tdim = Cell::topological_dimension(_cell_type);
  if (pts.cols() != tdim)
    throw std::runtime_error(
        "Point dimension does not match element dimension");

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      basis_at_pts
      = PolynomialSet::tabulate_polynomial_set(_cell_type, _degree, pts);
  const int psize = basis_at_pts.cols();
  const int ndofs = _new_coeffs.rows();

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
      pts.rows(), ndofs);
  result.setZero();

  for (int i = 0; i < ndofs; ++i)
    for (int k = 0; k < psize; ++k)
      result.col(i) += basis_at_pts.col(k) * _new_coeffs(i, k);

  return result;
}
