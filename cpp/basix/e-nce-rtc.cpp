// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-nce-rtc.h"
#include "e-lagrange.h"
#include "element-families.h"
#include "maps.h"
#include "moments.h"
#include "polyset.h"
#include "quadrature.h"
#include "sobolev-spaces.h"
#include <cmath>
#include <vector>

using namespace basix;

//----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T> basix::element::create_rtc(cell::type celltype, int degree,
                                            element::lagrange_variant lvariant,
                                            bool discontinuous)
{
  if (celltype != cell::type::quadrilateral
      and celltype != cell::type::hexahedron)
  {
    throw std::runtime_error("Unsupported cell type");
  }

  if (degree < 1)
    throw std::runtime_error("Degree must be at least 1");

  const std::size_t tdim = cell::topological_dimension(celltype);

  const cell::type facettype
      = (tdim == 2) ? cell::type::interval : cell::type::quadrilateral;

  // Evaluate the expansion polynomials at the quadrature points
  const auto [_pts, qwts] = quadrature::make_quadrature<T>(
      quadrature::type::Default, celltype, polyset::type::standard, 2 * degree);
  impl::mdspan_t<const T, 2> pts(_pts.data(), qwts.size(),
                                 _pts.size() / qwts.size());
  const auto [_phi, shape]
      = polyset::tabulate(celltype, polyset::type::standard, degree, 0, pts);
  impl::mdspan_t<const T, 3> phi(_phi.data(), shape);

  // The number of order (degree) polynomials
  const std::size_t psize = phi.extent(1);

  const int facet_count = tdim == 2 ? 4 : 6;
  const int facet_dofs
      = polyset::dim(facettype, polyset::type::standard, degree - 1);
  const int internal_dofs = tdim == 2 ? 2 * degree * (degree - 1)
                                      : 3 * degree * degree * (degree - 1);
  const std::size_t ndofs = facet_count * facet_dofs + internal_dofs;

  // Create coefficients for order (degree-1) vector polynomials
  impl::mdarray_t<T, 2> wcoeffs(ndofs, psize * tdim);
  const int nv
      = polyset::dim(cell::type::interval, polyset::type::standard, degree);
  const int ns
      = polyset::dim(cell::type::interval, polyset::type::standard, degree - 1);
  int dof = 0;
  if (tdim == 2)
  {
    for (int i = 0; i < ns; ++i)
    {
      for (int j = 0; j < nv; ++j)
      {
        wcoeffs[dof++, j * nv + i] = 1;
        wcoeffs[dof++, psize + i * nv + j] = 1;
      }
    }
  }
  else
  {
    for (int i = 0; i < ns; ++i)
    {
      for (int j = 0; j < ns; ++j)
      {
        for (int k = 0; k < nv; ++k)
        {
          wcoeffs[dof++, k * nv * nv + j * nv + i] = 1;
          wcoeffs[dof++, psize + i * nv * nv + k * nv + j] = 1;
          wcoeffs[dof++, psize * 2 + j * nv * nv + i * nv + k] = 1;
        }
      }
    }
  }

  assert((std::size_t)dof == ndofs);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;

  for (std::size_t i = 0; i < tdim - 1; ++i)
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, i);
    x[i] = std::vector(num_ent, impl::mdarray_t<T, 2>(0, tdim));
    M[i] = std::vector(num_ent, impl::mdarray_t<T, 4>(0, tdim, 0, 1));
  }

  {
    FiniteElement<T> moment_space
        = element::create_lagrange<T>(facettype, degree - 1, lvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_normal_integral_moments<T>(
        moment_space, celltype, polyset::type::standard, tdim, 2 * degree - 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[tdim - 1].emplace_back(std::array{xshape[0], xshape[1]}, _x[i]);
      M[tdim - 1].emplace_back(
          std::array{Mshape[0], Mshape[1], Mshape[2], Mshape[3]}, _M[i]);
    }
  }

  // Add integral moments on interior
  if (degree > 1)
  {
    auto [_x, xshape, _M, Mshape] = moments::make_dot_integral_moments<T>(
        element::create_nce<T>(celltype, degree - 1, lvariant, true), celltype,
        polyset::type::standard, tdim, 2 * degree - 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[tdim].emplace_back(std::array{xshape[0], xshape[1]}, _x[i]);
      M[tdim].emplace_back(
          std::array{Mshape[0], Mshape[1], Mshape[2], Mshape[3]}, _M[i]);
    }
  }
  else
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, tdim);
    x[tdim] = std::vector(num_ent, impl::mdarray_t<T, 2>(0, tdim));
    M[tdim] = std::vector(num_ent, impl::mdarray_t<T, 4>(0, tdim, 0, 1));
  }

  std::array<std::vector<mdspan_t<const T, 2>>, 4> xview = impl::to_mdspan(x);
  std::array<std::vector<mdspan_t<const T, 4>>, 4> Mview = impl::to_mdspan(M);
  std::array<std::vector<std::vector<T>>, 4> xbuffer;
  std::array<std::vector<std::vector<T>>, 4> Mbuffer;
  if (discontinuous)
  {
    std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
    std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
    std::tie(xbuffer, xshape, Mbuffer, Mshape)
        = element::make_discontinuous(xview, Mview, tdim, tdim);
    xview = impl::to_mdspan(xbuffer, xshape);
    Mview = impl::to_mdspan(Mbuffer, Mshape);
  }

  sobolev::space space
      = discontinuous ? sobolev::space::L2 : sobolev::space::HCurl;
  return FiniteElement<T>(
      element::family::RT, celltype, polyset::type::standard, degree, {tdim},
      impl::mdspan_t<T, 2>(wcoeffs.data(), wcoeffs.extents()), xview, Mview, 0,
      maps::type::contravariantPiola, space, discontinuous, degree - 1, degree,
      lvariant, element::dpc_variant::unset);
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T> basix::element::create_nce(cell::type celltype, int degree,
                                            element::lagrange_variant lvariant,
                                            bool discontinuous)
{
  if (celltype != cell::type::quadrilateral
      and celltype != cell::type::hexahedron)
  {
    throw std::runtime_error("Unsupported cell type");
  }

  if (degree < 1)
    throw std::runtime_error("Degree must be at least 1");

  const std::size_t tdim = cell::topological_dimension(celltype);

  // Evaluate the expansion polynomials at the quadrature points
  const auto [_pts, wts] = quadrature::make_quadrature<T>(
      quadrature::type::Default, celltype, polyset::type::standard, 2 * degree);
  impl::mdspan_t<const T, 2> pts(_pts.data(), wts.size(),
                                 _pts.size() / wts.size());
  const auto [_phi, shape]
      = polyset::tabulate(celltype, polyset::type::standard, degree, 0, pts);
  impl::mdspan_t<const T, 3> phi(_phi.data(), shape);

  // The number of order (degree) polynomials
  const int psize = phi.extent(1);

  const int edge_count = tdim == 2 ? 4 : 12;
  const int edge_dofs
      = polyset::dim(cell::type::interval, polyset::type::standard, degree - 1);
  const int face_count = tdim == 2 ? 1 : 6;
  const int face_dofs = 2 * degree * (degree - 1);
  const int volume_count = tdim == 2 ? 0 : 1;
  const int volume_dofs = 3 * degree * (degree - 1) * (degree - 1);
  const std::size_t ndofs = edge_count * edge_dofs + face_count * face_dofs
                            + volume_count * volume_dofs;

  // Create coefficients for order (degree-1) vector polynomials
  impl::mdarray_t<T, 2> wcoeffs(ndofs, psize * tdim);

  const int nv
      = polyset::dim(cell::type::interval, polyset::type::standard, degree);
  const int ns
      = polyset::dim(cell::type::interval, polyset::type::standard, degree - 1);

  int dof = 0;
  if (tdim == 2)
  {
    for (int i = 0; i < ns; ++i)
    {
      for (int j = 0; j < nv; ++j)
      {
        wcoeffs[dof++, i * nv + j] = 1;
        wcoeffs[dof++, psize + j * nv + i] = 1;
      }
    }
  }
  else
  {
    for (int i = 0; i < ns; ++i)
    {
      for (int j = 0; j < nv; ++j)
      {
        for (int k = 0; k < nv; ++k)
        {
          wcoeffs[dof++, i * nv * nv + j * nv + k] = 1;
          wcoeffs[dof++, psize + k * nv * nv + i * nv + j] = 1;
          wcoeffs[dof++, psize * 2 + j * nv * nv + k * nv + i] = 1;
        }
      }
    }
  }

  assert((std::size_t)dof == ndofs);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;

  x[0] = std::vector(cell::num_sub_entities(celltype, 0),
                     impl::mdarray_t<T, 2>(0, tdim));
  M[0] = std::vector(cell::num_sub_entities(celltype, 0),
                     impl::mdarray_t<T, 4>(0, tdim, 0, 1));

  {
    FiniteElement<T> edge_moment_space = element::create_lagrange<T>(
        cell::type::interval, degree - 1, lvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_tangent_integral_moments<T>(
        edge_moment_space, celltype, polyset::type::standard, tdim,
        2 * degree - 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[1].emplace_back(std::array{xshape[0], xshape[1]}, _x[i]);
      M[1].emplace_back(std::array{Mshape[0], Mshape[1], Mshape[2], Mshape[3]},
                        _M[i]);
    }
  }

  // Add integral moments on interior
  if (degree > 1)
  {
    // Face integral moment
    FiniteElement<T> moment_space = element::create_rtc<T>(
        cell::type::quadrilateral, degree - 1, lvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_dot_integral_moments<T>(
        moment_space, celltype, polyset::type::standard, tdim, 2 * degree - 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[2].emplace_back(std::array{xshape[0], xshape[1]}, _x[i]);
      M[2].emplace_back(std::array{Mshape[0], Mshape[1], Mshape[2], Mshape[3]},
                        _M[i]);
    }
  }
  else
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, 2);
    x[2] = std::vector(num_ent, impl::mdarray_t<T, 2>(0, tdim));
    M[2] = std::vector(num_ent, impl::mdarray_t<T, 4>(0, tdim, 0, 1));
  }
  if (tdim == 3)
  {
    if (degree > 1)
    {
      FiniteElement<T> moment_space = element::create_rtc<T>(
          cell::type::hexahedron, degree - 1, lvariant, true);
      auto [_x, xshape, _M, Mshape] = moments::make_dot_integral_moments<T>(
          moment_space, celltype, polyset::type::standard, tdim,
          2 * degree - 1);
      assert(_x.size() == _M.size());
      for (std::size_t i = 0; i < _x.size(); ++i)
      {
        x[3].emplace_back(std::array{xshape[0], xshape[1]}, _x[i]);
        M[3].emplace_back(
            std::array{Mshape[0], Mshape[1], Mshape[2], Mshape[3]}, _M[i]);
      }
    }
    else
    {
      const std::size_t num_ent = cell::num_sub_entities(celltype, 3);
      x[3] = std::vector(num_ent, impl::mdarray_t<T, 2>(0, tdim));
      M[3] = std::vector(num_ent, impl::mdarray_t<T, 4>(0, tdim, 0, 1));
    }
  }

  std::array<std::vector<mdspan_t<const T, 2>>, 4> xview = impl::to_mdspan(x);
  std::array<std::vector<mdspan_t<const T, 4>>, 4> Mview = impl::to_mdspan(M);
  std::array<std::vector<std::vector<T>>, 4> xbuffer;
  std::array<std::vector<std::vector<T>>, 4> Mbuffer;
  if (discontinuous)
  {
    std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
    std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
    std::tie(xbuffer, xshape, Mbuffer, Mshape)
        = element::make_discontinuous(xview, Mview, tdim, tdim);
    xview = impl::to_mdspan(xbuffer, xshape);
    Mview = impl::to_mdspan(Mbuffer, Mshape);
  }

  sobolev::space space
      = discontinuous ? sobolev::space::L2 : sobolev::space::HCurl;
  return FiniteElement<T>(
      element::family::N1E, celltype, polyset::type::standard, degree, {tdim},
      impl::mdspan_t<T, 2>(wcoeffs.data(), wcoeffs.extents()), xview, Mview, 0,
      maps::type::covariantPiola, space, discontinuous, degree - 1, degree,
      lvariant, element::dpc_variant::unset);
}
//-----------------------------------------------------------------------------
template FiniteElement<float>
basix::element::create_rtc(cell::type, int, element::lagrange_variant, bool);
template FiniteElement<double>
basix::element::create_rtc(cell::type, int, element::lagrange_variant, bool);

template FiniteElement<float>
basix::element::create_nce(cell::type, int, element::lagrange_variant, bool);
template FiniteElement<double>
basix::element::create_nce(cell::type, int, element::lagrange_variant, bool);
//-----------------------------------------------------------------------------
