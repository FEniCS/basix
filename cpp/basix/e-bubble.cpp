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
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

//----------------------------------------------------------------------------
FiniteElement basix::element::create_bubble(cell::type celltype, int degree,
                                            bool discontinuous)
{
  if (discontinuous)
    throw std::runtime_error("Cannot create a discontinuous bubble element.");

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

  const std::size_t tdim = cell::topological_dimension(celltype);

  std::array<std::vector<impl::mdarray2_t>, 4> x;
  std::array<std::vector<impl::mdarray4_t>, 4> M;

  for (std::size_t i = 0; i < tdim; ++i)
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, i);
    x[i] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
    M[i] = std::vector(num_ent, impl::mdarray4_t(0, 1, 0, 1));
  }

  // Evaluate the expansion polynomials at the quadrature points
  auto [pts, wts] = quadrature::make_quadrature(quadrature::type::Default,
                                                celltype, 2 * degree);
  const xt::xtensor<double, 2> phi = xt::view(
      polyset::tabulate(celltype, degree, 0, pts), 0, xt::all(), xt::all());

  // The number of order (degree) polynomials
  const std::size_t psize = phi.shape(0);

  // Create points at nodes on interior
  std::size_t ndofs = 0;
  {
    const auto [points, pshape] = lattice::create_new(
        celltype, degree, lattice::type::equispaced, false);
    ndofs = pshape[0];
    x[tdim].emplace_back(points, pshape[0], pshape[1]);
  }

  auto create_phi1 = [](auto& phi, auto& buffer)
  {
    buffer.resize(phi.shape(1) * phi.shape(2));
    impl::mdspan2_t phi1(buffer.data(), phi.shape(1), phi.shape(2));
    for (std::size_t i = 0; i < phi1.extent(0); ++i)
      for (std::size_t j = 0; j < phi1.extent(1); ++j)
        phi1(i, j) = phi(0, i, j);
    return phi1;
  };

  // Create coefficients for order (degree-1) vector polynomials
  std::vector<double> phi1_buffer;
  impl::mdspan2_t phi1;
  std::vector<double> bubble;
  switch (celltype)
  {
  case cell::type::interval:
  {
    auto _phi1 = polyset::tabulate(celltype, degree - 2, 0, pts);
    phi1 = create_phi1(_phi1, phi1_buffer);
    for (std::size_t i = 0; i < pts.shape(0); ++i)
    {
      double x0 = pts(i, 0);
      bubble.push_back(x0 * (1.0 - x0));
    }
    break;
  }
  case cell::type::triangle:
  {
    auto _phi1 = polyset::tabulate(celltype, degree - 3, 0, pts);
    phi1 = create_phi1(_phi1, phi1_buffer);
    for (std::size_t i = 0; i < pts.shape(0); ++i)
    {
      double x0 = pts(i, 0);
      double x1 = pts(i, 1);
      bubble.push_back(x0 * x1 * (1.0 - x0 - x1));
    }
    break;
  }
  case cell::type::tetrahedron:
  {
    auto _phi1 = polyset::tabulate(celltype, degree - 4, 0, pts);
    phi1 = create_phi1(_phi1, phi1_buffer);
    for (std::size_t i = 0; i < pts.shape(0); ++i)
    {
      double x0 = pts(i, 0);
      double x1 = pts(i, 1);
      double x2 = pts(i, 2);
      bubble.push_back(x0 * x1 * x2 * (1 - x0 - x1 - x2));
    }
    break;
  }
  case cell::type::quadrilateral:
  {
    auto _phi1 = polyset::tabulate(celltype, degree - 2, 0, pts);
    phi1 = create_phi1(_phi1, phi1_buffer);
    for (std::size_t i = 0; i < pts.shape(0); ++i)
    {
      double x0 = pts(i, 0);
      double x1 = pts(i, 1);
      bubble.push_back(x0 * (1 - x0) * x1 * (1 - x1));
    }
    break;
  }
  case cell::type::hexahedron:
  {
    auto _phi1 = polyset::tabulate(celltype, degree - 2, 0, pts);
    phi1 = create_phi1(_phi1, phi1_buffer);
    for (std::size_t i = 0; i < pts.shape(0); ++i)
    {
      double x0 = pts(i, 0);
      double x1 = pts(i, 1);
      double x2 = pts(i, 2);
      bubble.push_back(x0 * (1 - x0) * x1 * (1 - x1) * x2 * (1 - x2));
    }
    break;
  }
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  impl::mdarray2_t wcoeffs(ndofs, psize);
  for (std::size_t i = 0; i < phi1.extent(0); ++i)
    for (std::size_t j = 0; j < psize; ++j)
      for (std::size_t k = 0; k < wts.size(); ++k)
        wcoeffs(i, j) += wts[k] * phi1(i, k) * bubble[k] * phi(j, k);

  auto& _M = M[tdim].emplace_back(ndofs, 1, ndofs, 1);
  for (std::size_t i = 0; i < _M.extent(0); ++i)
    _M(i, 0, i, 0) = 1.0;

  impl::mdspan2_t wview(wcoeffs.data(), wcoeffs.extents());
  return FiniteElement(element::family::bubble, celltype, degree, {}, wview,
                       impl::to_mdspan(x), impl::to_mdspan(M), 0,
                       maps::type::identity, discontinuous, -1, degree);
}
//-----------------------------------------------------------------------------
