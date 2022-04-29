// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-crouzeix-raviart.h"
#include "cell.h"
#include "element-families.h"
#include "maps.h"
#include <array>
#include <vector>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>

using namespace basix;

//-----------------------------------------------------------------------------
FiniteElement basix::element::create_cr(cell::type celltype, int degree,
                                        bool discontinuous)
{
  if (degree != 1)
    throw std::runtime_error("Degree must be 1 for Crouzeix-Raviart");

  const std::size_t tdim = cell::topological_dimension(celltype);
  if (tdim < 2)
  {
    throw std::runtime_error(
        "topological dim must be 2 or 3 for Crouzeix-Raviart");
  }
  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron)
  {
    throw std::runtime_error(
        "Crouzeix-Raviart is only defined on triangles and tetrahedra.");
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const std::vector<std::vector<int>>& facet_topology = topology[tdim - 1];
  const std::size_t ndofs = facet_topology.size();
  const xt::xtensor<double, 2> geometry = cell::geometry(celltype);

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  for (std::size_t i = 0; i < tdim - 1; ++i)
  {
    x[i] = std::vector<xt::xtensor<double, 2>>(
        topology[i].size(), xt::xtensor<double, 2>({0, tdim}));
    M[i] = std::vector<xt::xtensor<double, 3>>(
        topology[i].size(), xt::xtensor<double, 3>({0, 1, 0}));
  }
  x[tdim] = std::vector<xt::xtensor<double, 2>>(
      topology[tdim].size(), xt::xtensor<double, 2>({0, tdim}));
  M[tdim] = std::vector<xt::xtensor<double, 3>>(
      topology[tdim].size(), xt::xtensor<double, 3>({0, 1, 0}));

  x[tdim - 1].resize(facet_topology.size(),
                     xt::zeros<double>({static_cast<std::size_t>(1), tdim}));

  // Compute facet midpoints
  for (std::size_t f = 0; f < facet_topology.size(); ++f)
  {
    auto v = xt::view(geometry, xt::keep(facet_topology[f]), xt::all());
    xt::row(x[tdim - 1][f], 0) = xt::mean(v, 0);
  }

  M[tdim - 1].resize(facet_topology.size(), xt::ones<double>({1, 1, 1}));

  if (discontinuous)
  {
    std::tie(x, M) = element::make_discontinuous(x, M, tdim, 1);
  }

  return FiniteElement(element::family::CR, celltype, 1, {},
                       xt::eye<double>(ndofs), x, M, maps::type::identity,
                       discontinuous, degree, degree);
}
//-----------------------------------------------------------------------------
