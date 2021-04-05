// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "crouzeix-raviart.h"
#include "cell.h"
#include "element-families.h"
#include "maps.h"
#include <array>
#include <vector>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>

using namespace basix;

//-----------------------------------------------------------------------------
FiniteElement basix::create_cr(cell::type celltype, int degree)
{
  if (degree != 1)
    throw std::runtime_error("Degree must be 1 for Crouzeix-Raviart");

  const std::size_t tdim = cell::topological_dimension(celltype);
  if (tdim < 2)
  {
    throw std::runtime_error(
        "opological dim must be 2 or 3 for Crouzeix-Raviart");
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const std::vector<std::vector<int>>& facet_topology = topology[tdim - 1];
  const xt::xtensor<double, 2> geometry = cell::geometry(celltype);

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;
  x[tdim - 1].resize(facet_topology.size(),
                     xt::zeros<double>({static_cast<std::size_t>(1), tdim}));

  // Compute facet midpoints
  for (std::size_t f = 0; f < facet_topology.size(); ++f)
  {
    auto v = xt::view(geometry, xt::keep(facet_topology[f]), xt::all());
    xt::row(x[tdim - 1][f], 0) = xt::mean(v, 0);
  }

  const std::size_t ndofs = facet_topology.size();
  const std::size_t transform_count = tdim == 2 ? 3 : 14;
  const auto base_transformations
      = xt::tile(xt::expand_dims(xt::eye<double>(ndofs), 0), transform_count);

  // Crouzeix-Raviart has one dof on each entity of tdim - 1.
  std::vector<std::vector<int>> entity_dofs(topology.size());
  entity_dofs[0].resize(topology[0].size(), 0);
  entity_dofs[1].resize(topology[1].size(), (tdim == 2) ? 1 : 0);
  entity_dofs[2].resize(topology[2].size(), (tdim == 3) ? 1 : 0);
  if (tdim == 3)
    entity_dofs[3] = {0};

  M[tdim - 1].resize(facet_topology.size(), xt::ones<double>({1, 1, 1}));
  const xt::xtensor<double, 3> coeffs = compute_expansion_coefficients(
      celltype, xt::eye<double>(ndofs), {M[tdim - 1]}, {x[tdim - 1]}, degree);
  return FiniteElement(element::family::CR, celltype, 1, {1}, coeffs,
                       entity_dofs, base_transformations, x, M,
                       maps::type::identity);
}
//-----------------------------------------------------------------------------
