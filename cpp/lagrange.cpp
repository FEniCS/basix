// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "lagrange.h"
#include "simplex.h"
#include <Eigen/Dense>

Lagrange::Lagrange(int dim, int degree) : FiniteElement(dim, degree)
{
  // Reference simplex vertices
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> simplex
      = ReferenceSimplex::create_simplex(dim);

  // Create orthonormal basis on simplex
  std::vector<Polynomial> basis
      = ReferenceSimplex::compute_polynomial_set(dim, degree);

  // Tabulate basis at nodes and get inverse
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pt
      = ReferenceSimplex::create_lattice(simplex, degree, true);
  const int ndofs = pt.rows();
  assert(ndofs == (int)basis.size());

  Eigen::MatrixXd dualmat(ndofs, ndofs);
  for (int j = 0; j < ndofs; ++j)
    dualmat.col(j) = basis[j].tabulate(pt);

  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(ndofs, ndofs);
  apply_dualmat_to_basis(coeffs, dualmat, basis, 1);
}
