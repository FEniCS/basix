// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-bubble.h"
#include "element-families.h"
#include "lattice.h"
#include "maps.h"
#include "polyset.h"
#include "quadrature.h"
#include "sobolev-spaces.h"
#include <array>
#include <vector>

using namespace basix;

//----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T> basix::element::create_bubble(cell::type celltype, int degree,
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

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;

  for (std::size_t i = 0; i < tdim; ++i)
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, i);
    x[i] = std::vector(num_ent, impl::mdarray_t<T, 2>(0, tdim));
    M[i] = std::vector(num_ent, impl::mdarray_t<T, 4>(0, 1, 0, 1));
  }

  // Evaluate the expansion polynomials at the quadrature points
  const auto [_pts, wts] = quadrature::make_quadrature<T>(
      quadrature::type::Default, celltype, polyset::type::standard, 2 * degree);
  impl::mdspan_t<const T, 2> pts(_pts.data(), wts.size(),
                                 _pts.size() / wts.size());
  const auto [_phi, shape]
      = polyset::tabulate(celltype, polyset::type::standard, degree, 0, pts);
  impl::mdspan_t<const T, 3> phi(_phi.data(), shape);

  // The number of order (degree) polynomials
  const std::size_t psize = phi.extent(1);

  // Create points at nodes on interior
  std::size_t ndofs = 0;
  {
    const auto [points, pshape] = lattice::create<T>(
        celltype, degree, lattice::type::equispaced, false);
    ndofs = pshape[0];
    x[tdim].emplace_back(pshape, points);
  }

  auto create_phi1 = [](auto& phi, auto& buffer)
  {
    buffer.resize(phi.extent(1) * phi.extent(2));
    impl::mdspan_t<T, 2> phi1(buffer.data(), phi.extent(1), phi.extent(2));
    for (std::size_t i = 0; i < phi1.extent(0); ++i)
      for (std::size_t j = 0; j < phi1.extent(1); ++j)
        phi1[i, j] = phi[0, i, j];
    return phi1;
  };

  // Create coefficients for order (degree-1) vector polynomials
  std::vector<T> phi1_buffer;
  impl::mdspan_t<T, 2> phi1;
  std::vector<T> bubble;
  switch (celltype)
  {
  case cell::type::interval:
  {
    const auto [_phi1, shape] = polyset::tabulate(
        celltype, polyset::type::standard, degree - 2, 0, pts);
    impl::mdspan_t<const T, 3> p1(_phi1.data(), shape);
    phi1 = create_phi1(p1, phi1_buffer);
    for (std::size_t i = 0; i < pts.extent(0); ++i)
    {
      T x0 = pts[i, 0];
      bubble.push_back(x0 * (1.0 - x0));
    }
    break;
  }
  case cell::type::triangle:
  {
    const auto [_phi1, shape] = polyset::tabulate(
        celltype, polyset::type::standard, degree - 3, 0, pts);
    impl::mdspan_t<const T, 3> p1(_phi1.data(), shape);
    phi1 = create_phi1(p1, phi1_buffer);
    for (std::size_t i = 0; i < pts.extent(0); ++i)
    {
      T x0 = pts[i, 0];
      T x1 = pts[i, 1];
      bubble.push_back(x0 * x1 * (1.0 - x0 - x1));
    }
    break;
  }
  case cell::type::tetrahedron:
  {
    const auto [_phi1, shape] = polyset::tabulate(
        celltype, polyset::type::standard, degree - 4, 0, pts);
    impl::mdspan_t<const T, 3> p1(_phi1.data(), shape);
    phi1 = create_phi1(p1, phi1_buffer);
    for (std::size_t i = 0; i < pts.extent(0); ++i)
    {
      T x0 = pts[i, 0];
      T x1 = pts[i, 1];
      T x2 = pts[i, 2];
      bubble.push_back(x0 * x1 * x2 * (1 - x0 - x1 - x2));
    }
    break;
  }
  case cell::type::quadrilateral:
  {
    const auto [_phi1, shape] = polyset::tabulate(
        celltype, polyset::type::standard, degree - 2, 0, pts);
    impl::mdspan_t<const T, 3> p1(_phi1.data(), shape);
    phi1 = create_phi1(p1, phi1_buffer);
    for (std::size_t i = 0; i < pts.extent(0); ++i)
    {
      T x0 = pts[i, 0];
      T x1 = pts[i, 1];
      bubble.push_back(x0 * (1 - x0) * x1 * (1 - x1));
    }
    break;
  }
  case cell::type::hexahedron:
  {
    const auto [_phi1, shape] = polyset::tabulate(
        celltype, polyset::type::standard, degree - 2, 0, pts);
    impl::mdspan_t<const T, 3> p1(_phi1.data(), shape);
    phi1 = create_phi1(p1, phi1_buffer);
    for (std::size_t i = 0; i < pts.extent(0); ++i)
    {
      T x0 = pts[i, 0];
      T x1 = pts[i, 1];
      T x2 = pts[i, 2];
      bubble.push_back(x0 * (1 - x0) * x1 * (1 - x1) * x2 * (1 - x2));
    }
    break;
  }
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  impl::mdarray_t<T, 2> wcoeffs(ndofs, psize);
  for (std::size_t i = 0; i < phi1.extent(0); ++i)
    for (std::size_t j = 0; j < psize; ++j)
      for (std::size_t k = 0; k < wts.size(); ++k)
        wcoeffs[i, j] += wts[k] * phi1[i, k] * bubble[k] * phi[0, j, k];

  math::orthogonalise<T>(wcoeffs);

  auto& _M = M[tdim].emplace_back(ndofs, 1, ndofs, 1);
  for (std::size_t i = 0; i < _M.extent(0); ++i)
    _M[i, 0, i, 0] = 1.0;

  impl::mdspan_t<T, 2> wview(wcoeffs.data(), wcoeffs.extents());
  sobolev::space space
      = discontinuous ? sobolev::space::L2 : sobolev::space::H1;
  return FiniteElement<T>(
      element::family::bubble, celltype, polyset::type::standard, degree, {},
      wview, impl::to_mdspan(x), impl::to_mdspan(M), 0, maps::type::identity,
      space, discontinuous, -1, degree, element::lagrange_variant::unset,
      element::dpc_variant::unset);
}
//-----------------------------------------------------------------------------
template FiniteElement<float> element::create_bubble(cell::type, int, bool);
template FiniteElement<double> element::create_bubble(cell::type, int, bool);
//-----------------------------------------------------------------------------
