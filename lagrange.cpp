// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "lagrange.h"
#include "simplex.h"

Lagrange::Lagrange(int dim, int degree) : _dim(dim), _degree(degree)
{
  // Reference simplex
  ReferenceSimplex simplex(dim);

  // Create orthonormal basis on simplex
  std::vector<Polynomial> bset = simplex.compute_polynomial_set(degree);

  // Tabulate basis at nodes and get inverse
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pt
      = simplex.lattice(degree);
  Eigen::MatrixXd dualmat(pt.rows(), bset.size());
  for (std::size_t j = 0; j < bset.size(); ++j)
    dualmat.col(j) = bset[j].tabulate(pt);
  Eigen::MatrixXd w = dualmat.inverse();

  // Matrix multiply basis by w
  poly_set.resize(bset.size(), Polynomial::zero(dim));
  for (std::size_t j = 0; j < bset.size(); ++j)
  {
    for (std::size_t k = 0; k < bset.size(); ++k)
      poly_set[j] += bset[k] * w(k, j);
  }
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Lagrange::tabulate_basis(
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts) const
{
  if (pts.cols() != _dim)
    throw std::runtime_error(
        "Point dimension does not match element dimension");

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
      pts.rows(), poly_set.size());
  for (std::size_t j = 0; j < poly_set.size(); ++j)
    result.col(j) = poly_set[j].tabulate(pts);

  return result;
}
