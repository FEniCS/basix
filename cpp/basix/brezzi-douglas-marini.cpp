// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "brezzi-douglas-marini.h"
#include "element-families.h"
#include "lagrange.h"
#include "maps.h"
#include "moments.h"
#include "nedelec.h"
#include "polyset.h"
#include "quadrature.h"
#include <numeric>
#include <vector>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

//----------------------------------------------------------------------------
FiniteElement basix::create_bdm(cell::type celltype, int degree)
{
  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron)
    throw std::runtime_error("Unsupported cell type");

  const std::size_t tdim = cell::topological_dimension(celltype);
  const cell::type facettype
      = (tdim == 2) ? cell::type::interval : cell::type::triangle;

  // The number of order (degree) scalar polynomials
  const std::size_t npoly = polyset::dim(celltype, degree);
  const std::size_t ndofs = npoly * tdim;

  // quadrature degree
  int quad_deg = 5 * degree;

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  // Add integral moments on facets
  const std::size_t facet_count = tdim + 1;
  const std::size_t facet_dofs = polyset::dim(facettype, degree);
  const int internal_dofs = ndofs - facet_count * facet_dofs;

  FiniteElement facet_moment_space = create_dlagrange(facettype, degree);
  std::tie(x[tdim - 1], M[tdim - 1]) = moments::make_normal_integral_moments(
      facet_moment_space, celltype, tdim, quad_deg);

  xt::xtensor<double, 3> facet_transforms
      = moments::create_normal_moment_dof_transformations(facet_moment_space);

  // Add integral moments on interior
  if (degree > 1)
  {
    // Interior integral moment
    auto ele = create_nedelec(celltype, degree - 1);
    std::tie(x[tdim], M[tdim])
        = moments::make_dot_integral_moments(ele, celltype, tdim, quad_deg);
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::size_t transform_count = 0;
  for (std::size_t i = 1; i < tdim; ++i)
    transform_count += topology[i].size() * i;
  auto base_transformations
      = xt::tile(xt::expand_dims(xt::eye<double>(ndofs), 0), transform_count);
  switch (tdim)
  {
  case 2:
    for (std::size_t edge = 0; edge < facet_count; ++edge)
    {
      const std::size_t start = facet_dofs * edge;
      auto range = xt::range(start, start + facet_dofs);
      xt::view(base_transformations, edge, range, range)
          = xt::view(facet_transforms, 0, xt::all(), xt::all());
    }
    break;
  case 3:
    for (std::size_t face = 0; face < facet_count; ++face)
    {
      const std::size_t start = facet_dofs * face;
      auto range = xt::range(start, start + facet_dofs);
      xt::view(base_transformations, 6 + 2 * face, range, range)
          = xt::view(facet_transforms, 0, xt::all(), xt::all());
      xt::view(base_transformations, 6 + 2 * face + 1, range, range)
          = xt::view(facet_transforms, 1, xt::all(), xt::all());
    }
    break;
  default:
    throw std::runtime_error("Invalid topological dimension.");
  }

  // BDM has facet_dofs dofs on each facet, and ndofs - facet_count *
  // facet_dofs in the interior
  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t i = 0; i < tdim - 1; ++i)
    entity_dofs[i].resize(topology[i].size(), 0);
  entity_dofs[tdim - 1].resize(topology[tdim - 1].size(), facet_dofs);
  entity_dofs[tdim] = {internal_dofs};

  // Create coefficients for order (degree-1) vector polynomials
  xt::xtensor<double, 2> B = xt::eye<double>(ndofs);
  xt::xtensor<double, 3> coeffs = compute_expansion_coefficients(
      celltype, B, {M[tdim - 1], M[tdim]}, {x[tdim - 1], x[tdim]}, degree);

  return FiniteElement(element::family::BDM, celltype, degree, {tdim}, coeffs,
                       entity_dofs, base_transformations, x, M,
                       maps::type::contravariantPiola);
}
//-----------------------------------------------------------------------------
