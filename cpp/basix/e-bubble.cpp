// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-bubble.h"
#include "element-families.h"
#include "lattice.h"
#include "maps.h"
#include "polyset.h"
#include "quadrature.h"
#include <array>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

//----------------------------------------------------------------------------
FiniteElement basix::create_bubble(cell::type celltype, int degree,
                                   element::variant variant)
{
  switch (celltype)
  {
  case cell::type::interval:
    if (degree < 2)
      throw std::runtime_error(
          "Bubble element on an interval must have degree at least 2");
    break;
  case cell::type::triangle:
    if (degree < 3)
      throw std::runtime_error(
          "Bubble element on a triangle must have degree at least 3");
    break;
  case cell::type::tetrahedron:
    if (degree < 4)
      throw std::runtime_error(
          "Bubble element on a tetrahedron must have degree at least 4");
    break;
  case cell::type::quadrilateral:
    if (degree < 2)
      throw std::runtime_error("Bubble element on a quadrilateral interval "
                               "must have degree at least 2");
    break;
  case cell::type::hexahedron:
    if (degree < 2)
      throw std::runtime_error(
          "Bubble element on a hexahedron must have degree at least 2");
    break;
  default:
    throw std::runtime_error("Unsupported cell type");
  }

  // For quads and hexes, use GLL points by default. Otherwise use equispaced
  if (variant == element::variant::DEFAULT)
  {
    if (celltype == cell::type::quadrilateral
        or celltype == cell::type::hexahedron)
      variant = element::variant::GLL;
    else
      variant = element::variant::EQ;
  }

  const std::size_t tdim = cell::topological_dimension(celltype);

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  // Evaluate the expansion polynomials at the quadrature points
  auto [pts, _wts]
      = quadrature::make_quadrature("default", celltype, 2 * degree);
  auto wts = xt::adapt(_wts);

  const xt::xtensor<double, 2> phi = xt::view(
      polyset::tabulate(celltype, degree, 0, pts), 0, xt::all(), xt::all());

  // The number of order (degree) polynomials
  const std::size_t psize = phi.shape(1);

  // Create points at nodes on interior
  const auto points = lattice::create(
      celltype, degree, element::variant_to_lattice(variant), false);
  const std::size_t ndofs = points.shape(0);
  x[tdim].push_back(points);

  // Create coefficients for order (degree-1) vector polynomials
  xt::xtensor<double, 2> phi1;
  xt::xtensor<double, 1> bubble;
  switch (celltype)
  {
  case cell::type::interval:
  {
    phi1 = xt::view(polyset::tabulate(celltype, degree - 2, 0, pts), 0,
                    xt::all(), xt::all());
    auto p = pts;
    bubble = p * (1.0 - p);
    break;
  }
  case cell::type::triangle:
  {
    phi1 = xt::view(polyset::tabulate(celltype, degree - 3, 0, pts), 0,
                    xt::all(), xt::all());
    auto p0 = xt::col(pts, 0);
    auto p1 = xt::col(pts, 1);
    bubble = p0 * p1 * (1 - p0 - p1);
    break;
  }
  case cell::type::tetrahedron:
  {
    phi1 = xt::view(polyset::tabulate(celltype, degree - 4, 0, pts), 0,
                    xt::all(), xt::all());
    auto p0 = xt::col(pts, 0);
    auto p1 = xt::col(pts, 1);
    auto p2 = xt::col(pts, 2);
    bubble = p0 * p1 * p2 * (1 - p0 - p1 - p2);
    break;
  }
  case cell::type::quadrilateral:
  {
    phi1 = xt::view(polyset::tabulate(celltype, degree - 2, 0, pts), 0,
                    xt::all(), xt::all());
    auto p0 = xt::col(pts, 0);
    auto p1 = xt::col(pts, 1);
    bubble = p0 * (1 - p0) * p1 * (1 - p1);
    break;
  }
  case cell::type::hexahedron:
  {
    phi1 = xt::view(polyset::tabulate(celltype, degree - 2, 0, pts), 0,
                    xt::all(), xt::all());
    auto p0 = xt::col(pts, 0);
    auto p1 = xt::col(pts, 1);
    auto p2 = xt::col(pts, 2);
    bubble = p0 * (1 - p0) * p1 * (1 - p1) * p2 * (1 - p2);
    break;
  }
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize});
  for (std::size_t i = 0; i < phi1.shape(1); ++i)
  {
    auto integrand = xt::col(phi1, i) * bubble;
    for (std::size_t k = 0; k < psize; ++k)
      wcoeffs(i, k) = xt::sum(wts * integrand * xt::col(phi, k))();
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  M[tdim].push_back(xt::xtensor<double, 3>({ndofs, 1, ndofs}));
  xt::view(M[tdim][0], xt::all(), 0, xt::all()) = xt::eye<double>(ndofs);

  const int num_transformations = tdim * (tdim - 1) / 2;
  std::vector<xt::xtensor<double, 2>> entity_transformations(
      num_transformations);

  xt::xtensor<double, 3> coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, {M[tdim]}, {x[tdim]}, degree);
  return FiniteElement(element::family::Bubble, celltype, degree, {1}, coeffs,
                       entity_transformations, x, M, maps::type::identity);
}
//-----------------------------------------------------------------------------
