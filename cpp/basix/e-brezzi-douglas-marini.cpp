// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-brezzi-douglas-marini.h"
#include "e-lagrange.h"
#include "e-nedelec.h"
#include "element-families.h"
#include "maps.h"
#include "moments.h"
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
FiniteElement basix::create_bdm(cell::type celltype, int degree,
                                element::variant variant)
{
  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron)
    throw std::runtime_error("Unsupported cell type");

  const std::size_t tdim = cell::topological_dimension(celltype);
  const cell::type facettype = sub_entity_type(celltype, tdim - 1, 0);

  // The number of order (degree) scalar polynomials
  const std::size_t ndofs = tdim * polyset::dim(celltype, degree);

  // quadrature degree
  int quad_deg = 5 * degree;

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  // Add integral moments on facets
  const FiniteElement facet_moment_space
      = create_dlagrange(facettype, degree, variant);
  std::tie(x[tdim - 1], M[tdim - 1]) = moments::make_normal_integral_moments(
      facet_moment_space, celltype, tdim, quad_deg);

  const xt::xtensor<double, 3> facet_transforms
      = moments::create_normal_moment_dof_transformations(facet_moment_space);

  // Add integral moments on interior
  if (degree > 1)
  {
    // Interior integral moment
    std::tie(x[tdim], M[tdim]) = moments::make_dot_integral_moments(
        create_nedelec(celltype, degree - 1, variant), celltype, tdim,
        quad_deg);
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::vector<xt::xtensor<double, 2>> entity_transformations;
  switch (tdim)
  {
  case 2:
    entity_transformations.push_back(
        xt::view(facet_transforms, 0, xt::all(), xt::all()));
    break;
  case 3:
    entity_transformations.push_back(xt::xtensor<double, 2>({0, 0}));
    entity_transformations.push_back(
        xt::view(facet_transforms, 0, xt::all(), xt::all()));
    entity_transformations.push_back(
        xt::view(facet_transforms, 1, xt::all(), xt::all()));
    break;
  default:
    throw std::runtime_error("Invalid topological dimension.");
  }

  // Create coefficients for order (degree-1) vector polynomials
  xt::xtensor<double, 3> coeffs = compute_expansion_coefficients(
      celltype, xt::eye<double>(ndofs), {M[tdim - 1], M[tdim]},
      {x[tdim - 1], x[tdim]}, degree);

  return FiniteElement(element::family::BDM, celltype, degree, {tdim}, coeffs,
                       entity_transformations, x, M,
                       maps::type::contravariantPiola);
}
//-----------------------------------------------------------------------------
