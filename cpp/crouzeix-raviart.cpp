
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

  const int tdim = cell::topological_dimension(celltype);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const std::vector<std::vector<int>> facet_topology = topology[tdim - 1];
  const Eigen::ArrayXXd geometry = cell::geometry(celltype);

  const int ndofs = facet_topology.size();
  Eigen::ArrayXXd pts = Eigen::ArrayXXd::Zero(ndofs, tdim);

  // Compute facet midpoints
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
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      base_permutations(perm_count, Eigen::MatrixXd::Identity(ndofs, ndofs));

  const Eigen::MatrixXd new_coeffs
      = FiniteElement::compute_expansion_coefficents(coeffs, dualmat);

  // Crouzeix-Raviart has one dof on each entity of tdim-1.
  std::vector<std::vector<int>> entity_dofs(topology.size());
  entity_dofs[0].resize(topology[0].size(), 0);
  entity_dofs[1].resize(topology[1].size(), 0);
  entity_dofs[tdim - 1].resize(topology[tdim - 1].size(), 1);
  entity_dofs[tdim].resize(topology[tdim].size(), 0);

  return FiniteElement(celltype, 1, {1}, new_coeffs, entity_dofs,
                       base_permutations);
}
//-----------------------------------------------------------------------------
