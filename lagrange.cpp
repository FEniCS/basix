// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "lagrange.h"
#include "simplex.h"

Lagrange::Lagrange(int dim, int degree) : FiniteElement(dim, degree)
{
  // Reference simplex vertices
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> simplex
      = ReferenceSimplex::create_simplex(dim);

  // Create orthonormal basis on simplex
  std::vector<Polynomial> bset
      = ReferenceSimplex::compute_polynomial_set(dim, degree);

  // Tabulate basis at nodes and get inverse
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pt
      = ReferenceSimplex::create_lattice(simplex, degree, true);
  assert(pt.rows() == bset.size());

  Eigen::MatrixXd dualmat(bset.size(), pt.rows());
  for (std::size_t j = 0; j < bset.size(); ++j)
    dualmat.row(j) = bset[j].tabulate(pt);

  Eigen::MatrixXd new_coeffs = dualmat.transpose().inverse();

  // Matrix multiply basis by new_coeffs
  poly_set.resize(bset.size(), Polynomial::zero(dim));
  for (std::size_t j = 0; j < bset.size(); ++j)
  {
    for (std::size_t k = 0; k < bset.size(); ++k)
      poly_set[j] += bset[k] * new_coeffs(k, j);
  }
}
