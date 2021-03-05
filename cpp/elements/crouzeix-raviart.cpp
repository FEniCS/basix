// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "crouzeix-raviart.h"
#include "core/element-families.h"
#include "core/mappings.h"
#include "core/polyset.h"
#include "core/quadrature.h"
#include <Eigen/Dense>
#include <numeric>
#include <vector>

using namespace basix;

//-----------------------------------------------------------------------------
FiniteElement basix::create_cr(cell::type celltype, int degree)
{
  if (degree != 1)
    throw std::runtime_error("Degree must be 1 for Crouzeix-Raviart");

  const int tdim = cell::topological_dimension(celltype);
  if (tdim < 2)
    throw std::runtime_error("Tdim must be 2 or 3 for Crouzeix-Raviart");

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const std::vector<std::vector<int>> facet_topology = topology[tdim - 1];
  const ndarray<double, 2> geometry = cell::geometry(celltype);

  const int ndofs = facet_topology.size();
  Eigen::ArrayXXd pts = Eigen::ArrayXXd::Zero(ndofs, tdim);

  // Compute facet midpoints
  int c = 0;
  for (const std::vector<int>& f : facet_topology)
  {
    for (int i : f)
    {
      for (std::size_t j = 0; j < geometry.shape[1]; ++j)
        pts(c, j) += geometry(i, j);
      // pts.row(c) += geometry.row(i);
    }

    for (std::size_t j = 0; j < geometry.shape[1]; ++j)
      pts(c, j) /= static_cast<double>(f.size());
    // pts.row(c) /= static_cast<double>(f.size());
    ++c;
  }

  Eigen::MatrixXd dual = polyset::tabulate(celltype, 1, 0, pts)[0];
  int transform_count = tdim == 2 ? 3 : 14;
  std::vector<Eigen::MatrixXd> base_transformations(
      transform_count, Eigen::MatrixXd::Identity(ndofs, ndofs));

  // Crouzeix-Raviart has one dof on each entity of tdim-1.
  std::vector<std::vector<int>> entity_dofs(topology.size());
  entity_dofs[0].resize(topology[0].size(), 0);
  entity_dofs[1].resize(topology[1].size(), (tdim == 2) ? 1 : 0);
  entity_dofs[2].resize(topology[2].size(), (tdim == 3) ? 1 : 0);
  if (tdim == 3)
    entity_dofs[3] = {0};

  const Eigen::MatrixXd coeffs = compute_expansion_coefficients(
      celltype, Eigen::MatrixXd::Identity(ndofs, ndofs),
      Eigen::MatrixXd::Identity(ndofs, ndofs), pts, degree);

  return FiniteElement(element::family::CR, celltype, 1, {1}, coeffs,
                       entity_dofs, base_transformations, pts,
                       Eigen::MatrixXd::Identity(ndofs, ndofs),
                       mapping::type::identity);
}
//-----------------------------------------------------------------------------
