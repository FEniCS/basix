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
  const int ndofs = pt.rows();
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(ndofs, ndofs);

  Eigen::MatrixXd dualmat
      = PolynomialSet::tabulate_polynomial_set(celltype, degree, pt);

  std::vector<Polynomial> basis
      = PolynomialSet::compute_polynomial_set(celltype, degree);

  apply_dualmat_to_basis(coeffs, dualmat, basis, 1);
}
