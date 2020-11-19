
// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "crouzeix-raviart.h"
#include "polynomial-set.h"
#include "quadrature.h"
#include <Eigen/Dense>
#include <numeric>
#include <vector>

using namespace libtab;

//-----------------------------------------------------------------------------
FiniteElement cr::create(cell::Type celltype, int degree)
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

  Eigen::MatrixXd dual = polyset::tabulate(celltype, 1, 0, pts)[0];
  int perm_count = 0;
  std::vector<Eigen::MatrixXd> base_permutations(
      perm_count, Eigen::MatrixXd::Identity(ndofs, ndofs));

  const Eigen::MatrixXd coeffs = FiniteElement::compute_expansion_coefficients(
      Eigen::MatrixXd::Identity(ndofs, ndofs), dual);

  // Crouzeix-Raviart has one dof on each entity of tdim-1.
  std::vector<std::vector<int>> entity_dofs(topology.size());
  entity_dofs[0].resize(topology[0].size(), 0);
  entity_dofs[1].resize(topology[1].size(), (tdim == 2) ? 1 : 0);
  entity_dofs[2].resize(topology[2].size(), (tdim == 3) ? 1 : 0);
  if (tdim == 3)
    entity_dofs[3] = {0};

  return FiniteElement(cr::family_name, celltype, 1, {1}, coeffs, entity_dofs,
                       base_permutations);
}
//-----------------------------------------------------------------------------
