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
  int quad_deg = 2 * degree;

  // Add integral moments on facets
  const std::size_t facet_count = tdim + 1;
  const std::size_t facet_dofs = polyset::dim(facettype, degree);
  const int internal_dofs = ndofs - facet_count * facet_dofs;

  xt::xtensor<double, 2> points_facet, matrix_facet;
  FiniteElement facet_moment_space = create_dlagrange(facettype, degree);
  std::tie(points_facet, matrix_facet) = moments::make_normal_integral_moments(
      facet_moment_space, celltype, tdim, quad_deg);

  auto [points_facet_new, M_facet_new]
      = moments::make_normal_integral_moments_new(facet_moment_space, celltype,
                                                  tdim, quad_deg);

  std::vector<xt::xtensor<double, 4>> M = {M_facet_new};
  std::vector<xt::xtensor<double, 3>> x = {points_facet_new};

  xt::xtensor<double, 3> facet_transforms
      = moments::create_normal_moment_dof_transformations(facet_moment_space);

  // Add integral moments on interior
  xt::xtensor<double, 2> points_cell, matrix_cell;
  xt::xtensor<double, 3> points_cell_new;
  xt::xtensor<double, 4> M_cell_new;
  if (degree > 1)
  {
    // Interior integral moment
    std::tie(points_cell, matrix_cell) = moments::make_dot_integral_moments(
        create_nedelec(celltype, degree - 1), celltype, tdim, quad_deg);

    auto [points_cell_new, M_cell_new] = moments::make_dot_integral_moments_new(
        create_nedelec(celltype, degree - 1), celltype, tdim, quad_deg);
    x.push_back(points_cell_new);
    M.push_back(M_cell_new);
  }

  // Interpolation points and matrix
  xt::xtensor<double, 2> points, matrix;
  std::tie(points, matrix) = combine_interpolation_data(
      points_facet, points_cell, {}, matrix_facet, matrix_cell, {}, tdim, tdim);

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
  xt::xtensor<double, 3> coeffs
      = compute_expansion_coefficients_new(celltype, B, M, x, degree);
  return FiniteElement(element::family::BDM, celltype, degree, {tdim}, coeffs,
                       entity_dofs, base_transformations, points, matrix,
                       maps::type::contravariantPiola);
}
//-----------------------------------------------------------------------------
