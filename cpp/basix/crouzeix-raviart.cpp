// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "crouzeix-raviart.h"
#include "element-families.h"
#include "mappings.h"
#include "polyset.h"
#include "quadrature.h"
#include <numeric>
#include <vector>
#include <xtensor/xpad.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

//-----------------------------------------------------------------------------
FiniteElement basix::create_cr(cell::type celltype, int degree)
{
  if (degree != 1)
    throw std::runtime_error("Degree must be 1 for Crouzeix-Raviart");

  const std::size_t tdim = cell::topological_dimension(celltype);
  if (tdim < 2)
    throw std::runtime_error("Tdim must be 2 or 3 for Crouzeix-Raviart");

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const std::vector<std::vector<int>> facet_topology = topology[tdim - 1];
  const xt::xtensor<double, 2> geometry = cell::geometry(celltype);

  const std::size_t ndofs = facet_topology.size();
  xt::xtensor<double, 2> pts = xt::zeros<double>({ndofs, tdim});

  // Compute facet midpoints
  int c = 0;
  for (const std::vector<int>& f : facet_topology)
  {
    for (int i : f)
    {
      for (std::size_t j = 0; j < geometry.shape()[1]; ++j)
        pts(c, j) += geometry(i, j);
    }

    for (std::size_t j = 0; j < geometry.shape()[1]; ++j)
      pts(c, j) /= static_cast<double>(f.size());
    // pts.row(c) /= static_cast<double>(f.size());
    ++c;
  }

  xt::xtensor<double, 2> dual = xt::view(polyset::tabulate(celltype, 1, 0, pts),
                                         0, xt::all(), xt::all());
  std::size_t transform_count = tdim == 2 ? 3 : 14;
  auto base_transformations
      = xt::tile(xt::expand_dims(xt::eye<double>(ndofs), 0), transform_count);

  // Crouzeix-Raviart has one dof on each entity of tdim-1.
  std::vector<std::vector<int>> entity_dofs(topology.size());
  entity_dofs[0].resize(topology[0].size(), 0);
  entity_dofs[1].resize(topology[1].size(), (tdim == 2) ? 1 : 0);
  entity_dofs[2].resize(topology[2].size(), (tdim == 3) ? 1 : 0);
  if (tdim == 3)
    entity_dofs[3] = {0};

  const xt::xtensor<double, 2> coeffs = compute_expansion_coefficients(
      celltype, xt::eye<double>(ndofs), xt::eye<double>(ndofs), pts, degree);
  return FiniteElement(element::family::CR, celltype, 1, {1}, coeffs,
                       entity_dofs, base_transformations, pts,
                       xt::eye<double>(ndofs), mapping::type::identity);
}
//-----------------------------------------------------------------------------
