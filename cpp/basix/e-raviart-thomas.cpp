// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-raviart-thomas.h"
#include "e-lagrange.h"
#include "element-families.h"
#include "maps.h"
#include "moments.h"
#include "polyset.h"
#include "quadrature.h"
#include <vector>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

//----------------------------------------------------------------------------
FiniteElement basix::element::create_rt(cell::type celltype, int degree,
                                        bool discontinuous)
{
  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron)
    throw std::runtime_error("Unsupported cell type");

  if (degree < 1)
    throw std::runtime_error("Degree must be at least 1");

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
  const auto [pts, _wts] = quadrature::make_quadrature(
      quadrature::type::Default, celltype, 2 * degree);
  auto wts = xt::adapt(_wts);
  const auto phi = xt::view(polyset::tabulate(celltype, degree, 0, pts), 0,
                            xt::all(), xt::all());

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

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  // Add integral moments on facets
  const FiniteElement facet_moment_space = element::create_lagrange(
      facettype, degree - 1, element::lagrange_variant::equispaced, true);
  std::tie(x[tdim - 1], M[tdim - 1]) = moments::make_normal_integral_moments(
      facet_moment_space, celltype, tdim, 2 * degree - 1);

  // Add integral moments on interior
  if (degree > 1)
  {
    // Interior integral moment
    std::tie(x[tdim], M[tdim]) = moments::make_integral_moments(
        element::create_lagrange(celltype, degree - 2,
                                 element::lagrange_variant::equispaced, true),
        celltype, tdim, 2 * degree - 2);
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  if (discontinuous)
  {
    std::tie(x, M) = element::make_discontinuous(x, M, tdim, tdim);
  }

  return FiniteElement(element::family::RT, celltype, degree, {tdim}, B, x, M,
                       maps::type::contravariantPiola, discontinuous, degree,
                       degree - 1);
}
//-----------------------------------------------------------------------------
