// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "tp.h"
#include "polynomial-set.h"
#include <Eigen/Dense>

TensorProduct::TensorProduct(Cell::Type celltype, int degree)
    : FiniteElement(celltype, degree)
{
  if (celltype != Cell::Type::quadrilateral
      and celltype != Cell::Type::hexahedron)
    throw std::runtime_error("Invalid celltype");

  // Create orthonormal basis on cell
  std::vector<Polynomial> basis
      = PolynomialSet::compute_polynomial_set(celltype, degree);

  // Tabulate basis at nodes
  Cell c(celltype);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pt
      = c.create_lattice(degree, true);

  int ndofs = pt.rows();
  assert(ndofs == (int)basis.size());

  Eigen::MatrixXd dualmat(ndofs, ndofs);
  for (int j = 0; j < ndofs; ++j)
    dualmat.col(j) = basis[j].tabulate(pt);

  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(ndofs, ndofs);
  apply_dualmat_to_basis(coeffs, dualmat, basis, 1);
}
