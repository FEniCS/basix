// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "bubble.h"
#include "element-families.h"
#include "lattice.h"
#include "mappings.h"
#include "polyset.h"
#include "quadrature.h"
#include <numeric>
#include <xtensor/xadapt.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

//----------------------------------------------------------------------------
FiniteElement basix::create_bubble(cell::type celltype, int degree)
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
      throw std::runtime_error(
          "Bubble element on a quadrilateral interval must "
          "have degree at least 2");
    break;
  case cell::type::hexahedron:
    if (degree < 2)
      throw std::runtime_error(
          "Bubble element on a hexahedron must have degree at least 2");
    break;
  default:
    throw std::runtime_error("Unsupported cell type");
  }

  // Evaluate the expansion polynomials at the quadrature points
  auto [Qpts, _Qwts]
      = quadrature::make_quadrature("default", celltype, 2 * degree);
  auto Qwts = xt::adapt(_Qwts);

  xt::xtensor<double, 2> polyset_at_Qpts = xt::view(
      polyset::tabulate(celltype, degree, 0, Qpts), 0, xt::all(), xt::all());

  // The number of order (degree) polynomials
  const std::size_t psize = polyset_at_Qpts.shape(1);

  // Create points at nodes on interior
  const auto points
      = lattice::create(celltype, degree, lattice::type::equispaced, false);

  const std::size_t ndofs = points.shape(0);

  // Create coefficients for order (degree-1) vector polynomials
  xt::xtensor<double, 2> lower_polyset_at_Qpts;
  xt::xtensor<double, 1> bubble;
  switch (celltype)
  {
  case cell::type::interval:
  {
    lower_polyset_at_Qpts
        = xt::view(polyset::tabulate(celltype, degree - 2, 0, Qpts), 0,
                   xt::all(), xt::all());
    auto p = Qpts;
    bubble = p * (1.0 - p);
    break;
  }
  case cell::type::triangle:
  {
    lower_polyset_at_Qpts
        = xt::view(polyset::tabulate(celltype, degree - 3, 0, Qpts), 0,
                   xt::all(), xt::all());
    auto p0 = xt::col(Qpts, 0);
    auto p1 = xt::col(Qpts, 1);
    bubble = p0 * p1 * (1 - p0 - p1);
    break;
  }
  case cell::type::tetrahedron:
  {
    lower_polyset_at_Qpts
        = xt::view(polyset::tabulate(celltype, degree - 4, 0, Qpts), 0,
                   xt::all(), xt::all());
    auto p0 = xt::col(Qpts, 0);
    auto p1 = xt::col(Qpts, 1);
    auto p2 = xt::col(Qpts, 2);
    bubble = p0 * p1 * p2 * (1 - p0 - p1 - p2);
    break;
  }
  case cell::type::quadrilateral:
  {
    lower_polyset_at_Qpts
        = xt::view(polyset::tabulate(celltype, degree - 2, 0, Qpts), 0,
                   xt::all(), xt::all());
    auto p0 = xt::col(Qpts, 0);
    auto p1 = xt::col(Qpts, 1);
    bubble = p0 * (1 - p0) * p1 * (1 - p1);
    break;
  }
  case cell::type::hexahedron:
  {
    lower_polyset_at_Qpts
        = xt::view(polyset::tabulate(celltype, degree - 2, 0, Qpts), 0,
                   xt::all(), xt::all());
    auto p0 = xt::col(Qpts, 0);
    auto p1 = xt::col(Qpts, 1);
    auto p2 = xt::col(Qpts, 2);
    bubble = p0 * (1 - p0) * p1 * (1 - p1) * p2 * (1 - p2);
    break;
  }
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize});
  for (std::size_t i = 0; i < lower_polyset_at_Qpts.shape(1); ++i)
  {
    auto integrand = xt::col(lower_polyset_at_Qpts, i) * bubble;
    for (std::size_t k = 0; k < psize; ++k)
    {
      double w_sum = xt::sum(Qwts * integrand * xt::col(polyset_at_Qpts, k))();
      wcoeffs(i, k) = w_sum;
    }
  }

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t i = 0; i < tdim; ++i)
    for (std::size_t j = 0; j < topology[i].size(); ++j)
      entity_dofs[i].push_back(0);
  entity_dofs[tdim].push_back(points.shape(0));

  std::size_t transform_count = 0;
  for (std::size_t i = 1; i < topology.size() - 1; ++i)
    transform_count += topology[i].size() * i;

  auto base_transformations
      = xt::tile(xt::expand_dims(xt::eye<double>(ndofs), 0), transform_count);
  xt::xtensor<double, 2> coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, xt::eye<double>(ndofs), points, degree);
  return FiniteElement(element::family::Bubble, celltype, degree, {1}, coeffs,
                       entity_dofs, base_transformations, points,
                       xt::eye<double>(ndofs), mapping::type::identity);
}
//-----------------------------------------------------------------------------
