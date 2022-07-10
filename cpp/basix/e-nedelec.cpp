// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-nedelec.h"
#include "e-lagrange.h"
#include "e-raviart-thomas.h"
#include "element-families.h"
#include "maps.h"
#include "math.h"
#include "mdspan.hpp"
#include "moments.h"
#include "polyset.h"
#include "quadrature.h"
#include <array>
#include <vector>

using namespace basix;

namespace
{
//-----------------------------------------------------------------------------
impl::mdarray2_t create_nedelec_2d_space(int degree)
{
  // Number of order (degree) vector polynomials
  const std::size_t nv = degree * (degree + 1) / 2;

  // Number of order (degree-1) vector polynomials
  const std::size_t ns0 = (degree - 1) * degree / 2;

  // Number of additional polynomials in Nedelec set
  const std::size_t ns = degree;

  // Tabulate polynomial set at quadrature points
  const auto [_pts, wts] = quadrature::make_quadrature(
      quadrature::type::Default, cell::type::triangle, 2 * degree);
  impl::cmdspan2_t pts(_pts.data(), wts.size(), _pts.size() / wts.size());
  const auto [_phi, shape]
      = polyset::tabulate(cell::type::triangle, degree, 0, pts);
  impl::cmdspan3_t phi(_phi.data(), shape);

  const std::size_t psize = phi.extent(1);

  // Create coefficients for order (degree-1) vector polynomials
  impl::mdarray2_t wcoeffs(nv * 2 + ns, psize * 2);
  for (std::size_t i = 0; i < nv; ++i)
  {
    wcoeffs(i, i) = 1.0;
    wcoeffs(nv + i, psize + i) = 1.0;
  }

  // Create coefficients for the additional Nedelec polynomials
  for (std::size_t i = 0; i < ns; ++i)
  {
    for (std::size_t j = 0; j < psize; ++j)
    {
      wcoeffs(2 * nv + i, j) = 0.0;
      wcoeffs(2 * nv + i, j + psize) = 0.0;
      for (std::size_t k = 0; k < wts.size(); ++k)
      {
        double p = phi(0, ns0 + i, k);
        wcoeffs(2 * nv + i, j) += wts[k] * p * pts(k, 1) * phi(0, j, k);
        wcoeffs(2 * nv + i, j + psize) -= wts[k] * p * pts(k, 0) * phi(0, j, k);
      }
    }
  }

  return wcoeffs;
}
//-----------------------------------------------------------------------------
impl::mdarray2_t create_nedelec_3d_space(int degree)
{
  // Reference tetrahedron
  const std::size_t tdim = 3;

  // Number of order (degree) vector polynomials
  const std::size_t nv = degree * (degree + 1) * (degree + 2) / 6;

  // Number of order (degree-1) vector polynomials
  const std::size_t ns0 = (degree - 1) * degree * (degree + 1) / 6;

  // Number of additional Nedelec polynomials that could be added
  const std::size_t ns = degree * (degree + 1) / 2;

  // Number of polynomials that would be included that are not
  // independent so are removed
  const std::size_t ns_remove = degree * (degree - 1) / 2;

  // Number of dofs in the space, ie size of polynomial set
  const std::size_t ndofs = 6 * degree + 4 * degree * (degree - 1)
                            + (degree - 2) * (degree - 1) * degree / 2;

  // Tabulate polynomial basis at quadrature points
  const auto [_pts, wts] = quadrature::make_quadrature(
      quadrature::type::Default, cell::type::tetrahedron, 2 * degree);
  impl::cmdspan2_t pts(_pts.data(), wts.size(), _pts.size() / wts.size());
  const auto [_phi, shape]
      = polyset::tabulate(cell::type::tetrahedron, degree, 0, pts);
  impl::cmdspan3_t phi(_phi.data(), shape);
  const std::size_t psize = phi.extent(1);

  // Create coefficients for order (degree-1) polynomials
  impl::mdarray2_t wcoeffs(ndofs, psize * tdim);
  for (std::size_t i = 0; i < tdim; ++i)
    for (std::size_t j = 0; j < nv; ++j)
      wcoeffs(i * nv + j, i * psize + j) = 1.0;

  // Create coefficients for additional Nedelec polynomials
  for (std::size_t i = 0; i < ns; ++i)
  {
    for (std::size_t j = 0; j < psize; ++j)
    {
      double w = 0.0;
      for (std::size_t k = 0; k < wts.size(); ++k)
        w += wts[k] * phi(0, ns0 + i, k) * pts(k, 2) * phi(0, j, k);

      // Don't include polynomials (*, *, 0) that are dependant
      if (i >= ns_remove)
        wcoeffs(tdim * nv + i - ns_remove, psize + j) = -w;
      wcoeffs(tdim * nv + i + ns - ns_remove, j) = w;
    }
  }

  for (std::size_t i = 0; i < ns; ++i)
  {
    for (std::size_t j = 0; j < psize; ++j)
    {
      double w = 0.0;
      for (std::size_t k = 0; k < wts.size(); ++k)
        w += wts[k] * phi(0, ns0 + i, k) * pts(k, 1) * phi(0, j, k);
      wcoeffs(tdim * nv + i + ns * 2 - ns_remove, j) = -w;

      // Don't include polynomials (*, *, 0) that are dependant
      if (i >= ns_remove)
        wcoeffs(tdim * nv + i - ns_remove, psize * 2 + j) = w;
    }
  }

  for (std::size_t i = 0; i < ns; ++i)
  {
    for (std::size_t j = 0; j < psize; ++j)
    {
      double w = 0.0;
      for (std::size_t k = 0; k < wts.size(); ++k)
        w += wts[k] * phi(0, ns0 + i, k) * pts(k, 0) * phi(0, j, k);

      wcoeffs(tdim * nv + i + ns - ns_remove, psize * 2 + j) = -w;
      wcoeffs(tdim * nv + i + ns * 2 - ns_remove, psize + j) = w;
    }
  }

  return wcoeffs;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
FiniteElement element::create_nedelec(cell::type celltype, int degree,
                                      lagrange_variant lvariant,
                                      bool discontinuous)
{
  if (degree < 1)
    throw std::runtime_error("Degree must be at least 1");

  const std::size_t tdim = cell::topological_dimension(celltype);

  std::array<std::vector<impl::mdarray2_t>, 4> x;
  std::array<std::vector<impl::mdarray4_t>, 4> M;

  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, 0);
    x[0] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
    M[0] = std::vector(num_ent, impl::mdarray4_t(0, tdim, 0, 1));
  }

  std::vector<double> wcoeffs;
  std::array<std::size_t, 2> wshape;
  switch (celltype)
  {
  case cell::type::triangle:
  {
    impl::mdarray2_t w = create_nedelec_2d_space(degree);
    wshape = {w.extent(0), w.extent(1)};
    wcoeffs.resize(wshape[0] * wshape[1]);
    std::copy_n(w.data(), w.size(), wcoeffs.data());
    break;
  }
  case cell::type::tetrahedron:
  {
    impl::mdarray2_t w = create_nedelec_3d_space(degree);
    wshape = {w.extent(0), w.extent(1)};
    wcoeffs.resize(wshape[0] * wshape[1]);
    std::copy_n(w.data(), w.size(), wcoeffs.data());
    break;
  }
  default:
    throw std::runtime_error("Invalid celltype in Nedelec");
  }

  // Integral representation for the boundary (edge) dofs
  {
    FiniteElement edge_space = element::create_lagrange(
        cell::type::interval, degree - 1, lvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_tangent_integral_moments(
        edge_space, celltype, tdim, 2 * degree - 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[1].emplace_back(_x[i], xshape[0], xshape[1]);
      M[1].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2], Mshape[3]);
    }
  }

  // Face dofs
  if (degree > 1)
  {
    FiniteElement face_space = element::create_lagrange(
        cell::type::triangle, degree - 2, lvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_integral_moments(
        face_space, celltype, tdim, 2 * degree - 2);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[2].emplace_back(_x[i], xshape[0], xshape[1]);
      M[2].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2], Mshape[3]);
    }
  }
  else
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, 2);
    x[2] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
    M[2] = std::vector(num_ent, impl::mdarray4_t(0, tdim, 0, 1));
  }

  // Volume dofs
  if (tdim == 3)
  {
    if (degree > 2 and tdim == 3)
    {
      auto [_x, xshape, _M, Mshape] = moments::make_integral_moments(
          element::create_lagrange(cell::type::tetrahedron, degree - 3,
                                   lvariant, true),
          cell::type::tetrahedron, 3, 2 * degree - 3);
      assert(_x.size() == _M.size());
      for (std::size_t i = 0; i < _x.size(); ++i)
      {
        x[3].emplace_back(_x[i], xshape[0], xshape[1]);
        M[3].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2], Mshape[3]);
      }
    }
    else
    {
      const std::size_t num_ent = cell::num_sub_entities(celltype, 3);
      x[3] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
      M[3] = std::vector(num_ent, impl::mdarray4_t(0, tdim, 0, 1));
    }
  }

  std::array<std::vector<cmdspan2_t>, 4> xview = impl::to_mdspan(x);
  std::array<std::vector<cmdspan4_t>, 4> Mview = impl::to_mdspan(M);
  std::array<std::vector<std::vector<double>>, 4> xbuffer;
  std::array<std::vector<std::vector<double>>, 4> Mbuffer;
  if (discontinuous)
  {
    // std::tie(x, M) = element::make_discontinuous(x, M, tdim, tdim);
    std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
    std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
    std::tie(xbuffer, xshape, Mbuffer, Mshape)
        = element::make_discontinuous(xview, Mview, tdim, tdim);
    xview = impl::to_mdspan(xbuffer, xshape);
    Mview = impl::to_mdspan(Mbuffer, Mshape);
  }

  return FiniteElement(element::family::N1E, celltype, degree, {tdim},
                       impl::mdspan2_t(wcoeffs.data(), wshape), xview, Mview, 0,
                       maps::type::covariantPiola, discontinuous, degree - 1,
                       degree, lvariant);
}
//-----------------------------------------------------------------------------
FiniteElement element::create_nedelec2(cell::type celltype, int degree,
                                       lagrange_variant lvariant,
                                       bool discontinuous)
{
  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron)
    throw std::runtime_error("Invalid celltype in Nedelec");

  if (degree < 1)
    throw std::runtime_error("Degree must be at least 1");

  std::array<std::vector<impl::mdarray2_t>, 4> x;
  std::array<std::vector<impl::mdarray4_t>, 4> M;

  const std::size_t tdim = cell::topological_dimension(celltype);
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, 0);
    x[0] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
    M[0] = std::vector(num_ent, impl::mdarray4_t(0, tdim, 0, 1));
  }

  // Integral representation for the edge dofs

  {
    FiniteElement edge_space = element::create_lagrange(cell::type::interval,
                                                        degree, lvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_tangent_integral_moments(
        edge_space, celltype, tdim, 2 * degree);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[1].emplace_back(_x[i], xshape[0], xshape[1]);
      M[1].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2], Mshape[3]);
    }
  }

  if (degree > 1)
  {
    // Integral moments on faces
    FiniteElement face_space
        = element::create_rt(cell::type::triangle, degree - 1, lvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_dot_integral_moments(
        face_space, celltype, tdim, 2 * degree - 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[2].emplace_back(_x[i], xshape[0], xshape[1]);
      M[2].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2], Mshape[3]);
    }
  }
  else
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, 2);
    x[2] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
    M[2] = std::vector(num_ent, impl::mdarray4_t(0, tdim, 0, 1));
  }

  if (tdim == 3)
  {
    if (degree > 2)
    {
      auto [_x, xshape, _M, Mshape] = moments::make_dot_integral_moments(
          element::create_rt(cell::type::tetrahedron, degree - 2, lvariant,
                             true),
          celltype, tdim, 2 * degree - 2);
      assert(_x.size() == _M.size());
      for (std::size_t i = 0; i < _x.size(); ++i)
      {
        x[3].emplace_back(_x[i], xshape[0], xshape[1]);
        M[3].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2], Mshape[3]);
      }
    }
    else
    {
      const std::size_t num_ent = cell::num_sub_entities(celltype, 2);
      x[3] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
      M[3] = std::vector(num_ent, impl::mdarray4_t(0, tdim, 0, 1));
    }
  }

  std::array<std::vector<cmdspan2_t>, 4> xview = impl::to_mdspan(x);
  std::array<std::vector<cmdspan4_t>, 4> Mview = impl::to_mdspan(M);
  std::array<std::vector<std::vector<double>>, 4> xbuffer;
  std::array<std::vector<std::vector<double>>, 4> Mbuffer;
  if (discontinuous)
  {
    std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
    std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
    std::tie(xbuffer, xshape, Mbuffer, Mshape)
        = element::make_discontinuous(xview, Mview, tdim, tdim);
    xview = impl::to_mdspan(xbuffer, xshape);
    Mview = impl::to_mdspan(Mbuffer, Mshape);
  }

  const std::size_t psize = polyset::dim(celltype, degree);
  return FiniteElement(element::family::N2E, celltype, degree, {tdim},
                       impl::mdspan2_t(math::eye(tdim * psize).data(),
                                       tdim * psize, tdim * psize),
                       xview, Mview, 0, maps::type::covariantPiola,
                       discontinuous, degree, degree, lvariant);
}
//-----------------------------------------------------------------------------
