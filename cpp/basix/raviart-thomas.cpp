// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "raviart-thomas.h"
#include "element-families.h"
#include "lagrange.h"
#include "maps.h"
#include "moments.h"
#include "polyset.h"
#include "quadrature.h"
#include <numeric>
#include <vector>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

//----------------------------------------------------------------------------
FiniteElement basix::create_rt(cell::type celltype, int degree)
{
  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron)
    throw std::runtime_error("Unsupported cell type");

  const std::size_t tdim = cell::topological_dimension(celltype);

  const cell::type facettype
      = (tdim == 2) ? cell::type::interval : cell::type::triangle;

  // The number of order (degree-1) scalar polynomials
  const std::size_t nv = polyset::dim(celltype, degree - 1);

  // The number of order (degree-2) scalar polynomials
  const std::size_t ns0 = polyset::dim(celltype, degree - 2);

  // The number of additional polynomials in the polynomial basis for
  // Raviart-Thomas
  const std::size_t ns = polyset::dim(facettype, degree - 1);

  // Evaluate the expansion polynomials at the quadrature points
  auto [pts, _wts]
      = quadrature::make_quadrature("default", celltype, 2 * degree);
  auto wts = xt::adapt(_wts);
  auto phi = xt::view(polyset::tabulate(celltype, degree, 0, pts), 0, xt::all(),
                      xt::all());

  // The number of order (degree) polynomials
  const std::size_t psize = phi.shape(1);

  // Create coefficients for order (degree-1) vector polynomials
  xt::xtensor<double, 2> B = xt::zeros<double>({nv * tdim + ns, psize * tdim});
  for (std::size_t j = 0; j < tdim; ++j)
  {
    xt::view(B, xt::range(nv * j, nv * j + nv),
             xt::range(psize * j, psize * j + nv))
        = xt::eye<double>(nv);
  }

  // Create coefficients for additional polynomials in Raviart-Thomas
  // polynomial basis
  for (std::size_t i = 0; i < ns; ++i)
  {
    auto p = xt::col(phi, ns0 + i);
    for (std::size_t k = 0; k < psize; ++k)
    {
      auto pk = xt::col(phi, k);
      for (std::size_t j = 0; j < tdim; ++j)
      {
        B(nv * tdim + i, k + psize * j)
            = xt::sum(wts * p * xt::col(pts, j) * pk)();
      }
    }
  }

  // quadrature degree
  int quad_deg = 5 * degree;

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  // Add integral moments on facets
  FiniteElement facet_moment_space = create_dlagrange(facettype, degree - 1);
  std::tie(x[tdim - 1], M[tdim - 1]) = moments::make_normal_integral_moments(
      facet_moment_space, celltype, tdim, quad_deg);
  xt::xtensor<double, 3> facet_transforms
      = moments::create_normal_moment_dof_transformations(facet_moment_space);

  const std::size_t facet_dofs = facet_transforms.shape(1);

  // Add integral moments on interior
  xt::xtensor<double, 2> points_cell, matrix_cell;
  if (degree > 1)
  {
    // Interior integral moment
    std::tie(x[tdim], M[tdim]) = moments::make_integral_moments(
        create_dlagrange(celltype, degree - 2), celltype, tdim, quad_deg);
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  const std::size_t facet_count = tdim + 1;
  const std::size_t ndofs = nv * tdim + ns;
  std::size_t transform_count = 0;
  for (std::size_t i = 1; i < tdim; ++i)
    transform_count += topology[i].size() * i;

  xt::xtensor<double, 3> base_transformations
      = xt::zeros<double>({transform_count, ndofs, ndofs});
  for (std::size_t i = 0; i < base_transformations.shape(0); ++i)
  {
    xt::view(base_transformations, i, xt::all(), xt::all())
        = xt::eye<double>(ndofs);
  }
  if (tdim == 2)
  {
    for (std::size_t edge = 0; edge < facet_count; ++edge)
    {
      const std::size_t start = facet_dofs * edge;
      auto range = xt::range(start, start + facet_dofs);
      xt::view(base_transformations, edge, range, range)
          = xt::view(facet_transforms, 0, xt::all(), xt::all());
    }
  }
  else if (tdim == 3)
  {
    for (std::size_t face = 0; face < facet_count; ++face)
    {
      const std::size_t start = facet_dofs * face;
      auto range = xt::range(start, start + facet_dofs);
      xt::view(base_transformations, 6 + 2 * face, range, range)
          = xt::view(facet_transforms, 0, xt::all(), xt::all());
      xt::view(base_transformations, 6 + 2 * face + 1, range, range)
          = xt::view(facet_transforms, 1, xt::all(), xt::all());
    }
  }

  xt::xtensor<double, 3> coeffs = compute_expansion_coefficients_new(
      celltype, B, {M[tdim - 1], M[tdim]}, {x[tdim - 1], x[tdim]}, degree);

  // Raviart-Thomas has ns dofs on each facet, and ns0*tdim in the
  // interior
  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t i = 0; i < tdim - 1; ++i)
    entity_dofs[i].resize(topology[i].size(), 0);
  entity_dofs[tdim - 1].resize(topology[tdim - 1].size(), facet_dofs);
  entity_dofs[tdim] = {(int)(ns0 * tdim)};

  return FiniteElement(element::family::RT, celltype, degree, {tdim}, coeffs,
                       entity_dofs, base_transformations, x, M,
                       maps::type::contravariantPiola);
}
//-----------------------------------------------------------------------------
