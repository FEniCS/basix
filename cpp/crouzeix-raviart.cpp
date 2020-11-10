
// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "crouzeix-raviart.h"
#include "polynomial-set.h"
#include "quadrature.h"
#include <Eigen/Dense>
#include <iostream>
#include <numeric>
#include <vector>

using namespace libtab;

//-----------------------------------------------------------------------------
FiniteElement CrouzeixRaviart::create(cell::Type celltype, int degree)
{
  if (degree != 1)
    throw std::runtime_error("Degree must be 1 for Crouzeix-Raviart");

  // Compute facet midpoints
  const int tdim = cell::topological_dimension(celltype);
  const std::vector<std::vector<int>> facet_topology
      = cell::topology(celltype)[tdim - 1];
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> geometry
      = cell::geometry(celltype);

  const int ndofs = facet_topology.size();
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pts(
      ndofs, tdim);
  pts.setZero();

  int c = 0;
  for (const std::vector<int>& f : facet_topology)
  {
    for (int i : f)
      pts.row(c) += geometry.row(i);
    pts.row(c) /= static_cast<double>(f.size());
    ++c;
  }

  // Initial coefficients are Identity Matrix
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(ndofs, ndofs);

  Eigen::MatrixXd dualmat = polyset::tabulate(celltype, 1, 0, pts)[0];

  int perm_count = 0;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      base_permutations(perm_count, Eigen::MatrixXd::Identity(ndofs, ndofs));

  auto new_coeffs
      = FiniteElement::compute_expansion_coefficents(coeffs, dualmat);
  FiniteElement el(celltype, 1, 1, new_coeffs, base_permutations);
  return el;
}
//-----------------------------------------------------------------------------
