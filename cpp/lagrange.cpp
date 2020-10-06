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

  // Create points at nodes
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pt
      = Cell::create_lattice(celltype, degree, true);
  const int ndofs = pt.rows();

  // Coefficients are Identity Matrix
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(ndofs, ndofs);

  // Point evaluation of basis
  Eigen::MatrixXd dualmat
      = PolynomialSet::tabulate_polynomial_set(celltype, degree, pt);

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

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      basis_at_pts
      = PolynomialSet::tabulate_polynomial_set(_cell_type, _degree, pts);

  return basis_at_pts * _coeffs.transpose();
}
//-----------------------------------------------------------------------------
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
Lagrange::tabulate_basis_derivatives(
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
      dbasis_at_pts = PolynomialSet::tabulate_polynomial_set_deriv(
          _cell_type, _degree, nderiv, pts);
  const int ndofs = _coeffs.rows();

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dresult(dbasis_at_pts.size());

  for (std::size_t p = 0; p < dresult.size(); ++p)
  {
    auto& result = dresult[p];
    result.resize(pts.rows(), ndofs);
    result.setZero();

    for (int i = 0; i < ndofs; ++i)
      for (int k = 0; k < dbasis_at_pts[p].cols(); ++k)
        result.col(i) += dbasis_at_pts[p].col(k) * _coeffs(i, k);
  }

  return dresult;
}
