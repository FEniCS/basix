// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "lagrange.h"
#include "polynomial-set.h"
#include "simplex.h"
#include <Eigen/Dense>

Lagrange::Lagrange(CellType celltype, int degree) : FiniteElement(0, degree)
{

  if (celltype == CellType::interval)
    _dim = 1;
  else if (celltype == CellType::triangle)
    _dim = 2;
  else if (celltype == CellType::tetrahedron)
    _dim = 3;
  else
    throw std::runtime_error("Invalid celltype");

  // Reference simplex vertices
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> simplex
      = ReferenceSimplex::create_simplex(_dim);

  // Create orthonormal basis on simplex
  std::vector<Polynomial> basis
      = PolynomialSet::compute_polynomial_set(celltype, degree);

  // Tabulate basis at nodes
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
