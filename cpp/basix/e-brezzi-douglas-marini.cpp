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

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  for (std::size_t i = 0; i < tdim - 1; ++i)
  {
    x[i] = std::vector<xt::xtensor<double, 2>>(
        cell::num_sub_entities(celltype, i), xt::xtensor<double, 2>({0, tdim}));
    M[i] = std::vector<xt::xtensor<double, 3>>(
        cell::num_sub_entities(celltype, i),
        xt::xtensor<double, 3>({0, tdim, 0}));
  }

  // Add integral moments on facets
  const FiniteElement facet_moment_space = element::create_lagrange(
      facettype, degree, element::lagrange_variant::legendre, true);
  std::tie(x[tdim - 1], M[tdim - 1]) = moments::make_normal_integral_moments(
      facet_moment_space, celltype, tdim, degree * 2);

  // Add integral moments on interior
  if (degree > 1)
  {
    // Interior integral moment
    std::tie(x[tdim], M[tdim]) = moments::make_dot_integral_moments(
        element::create_nedelec(celltype, degree - 1, true), celltype, tdim,
        2 * degree - 1);
  }
  else
  {
    x[tdim] = std::vector<xt::xtensor<double, 2>>(
        cell::num_sub_entities(celltype, tdim),
        xt::xtensor<double, 2>({0, tdim}));
    M[tdim] = std::vector<xt::xtensor<double, 3>>(
        cell::num_sub_entities(celltype, tdim),
        xt::xtensor<double, 3>({0, tdim, 0}));
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  if (discontinuous)
  {
    std::tie(x, M) = element::make_discontinuous(x, M, tdim, tdim);
  }

  return FiniteElement(element::family::BDM, celltype, degree, {tdim},
                       xt::eye<double>(ndofs), x, M,
                       maps::type::contravariantPiola, discontinuous, degree);
}
//-----------------------------------------------------------------------------
