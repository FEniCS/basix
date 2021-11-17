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
#include <vector>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>

using namespace basix;

//----------------------------------------------------------------------------
FiniteElement basix::element::create_bdm(cell::type celltype, int degree,
                                         bool discontinuous)
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
  const FiniteElement facet_moment_space = element::create_lagrange(
      facettype, degree, element::lagrange_variant::equispaced, true);
  std::tie(x[tdim - 1], M[tdim - 1]) = moments::make_normal_integral_moments(
      facet_moment_space, celltype, tdim, quad_deg);

  const xt::xtensor<double, 3> facet_transforms
      = moments::create_normal_moment_dof_transformations(facet_moment_space);

  // Add integral moments on interior
  if (degree > 1)
  {
    // Interior integral moment
    std::tie(x[tdim], M[tdim]) = moments::make_dot_integral_moments(
        element::create_nedelec(celltype, degree - 1, true), celltype, tdim,
        quad_deg);
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::map<cell::type, xt::xtensor<double, 3>> entity_transformations;
  switch (tdim)
  {
  case 2:
    entity_transformations[cell::type::interval] = facet_transforms;
    break;
  case 3:
    entity_transformations[cell::type::interval]
        = xt::xtensor<double, 3>({1, 0, 0});
    entity_transformations[cell::type::triangle] = facet_transforms;
    break;
  default:
    throw std::runtime_error("Invalid topological dimension.");
  }

  if (discontinuous)
  {
    std::tie(x, M, entity_transformations)
        = element::make_discontinuous(x, M, entity_transformations, tdim, tdim);
  }

  return FiniteElement(element::family::BDM, celltype, degree, {tdim},
                       xt::eye<double>(ndofs), entity_transformations, x, M,
                       maps::type::contravariantPiola, discontinuous);
}
//-----------------------------------------------------------------------------
