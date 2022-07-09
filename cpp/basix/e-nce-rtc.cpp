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
#include <vector>

using namespace basix;

//----------------------------------------------------------------------------
FiniteElement basix::element::create_rtc(cell::type celltype, int degree,
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
  const auto [_pts, qwts] = quadrature::make_quadrature(
      quadrature::type::Default, celltype, 2 * degree);
  impl::cmdspan2_t pts(_pts.data(), qwts.size(), _pts.size() / qwts.size());
  const auto [_phi, shape] = polyset::tabulate(celltype, degree, 0, pts);
  impl::cmdspan3_t phi(_phi.data(), shape);

  // The number of order (degree) polynomials
  const std::size_t psize = phi.extent(1);

  const int facet_count = tdim == 2 ? 4 : 6;
  const int facet_dofs = polyset::dim(facettype, degree - 1);
  const int internal_dofs = tdim == 2 ? 2 * degree * (degree - 1)
                                      : 3 * degree * degree * (degree - 1);
  const std::size_t ndofs = facet_count * facet_dofs + internal_dofs;

  // Create coefficients for order (degree-1) vector polynomials
  impl::mdarray2_t wcoeffs(ndofs, psize * tdim);
  const int nv_interval = polyset::dim(cell::type::interval, degree);
  const int ns_interval = polyset::dim(cell::type::interval, degree - 1);
  int dof = 0;
  if (tdim == 2)
  {
    for (std::size_t d = 0; d < tdim; ++d)
      for (int i = 0; i < ns_interval; ++i)
        for (int j = 0; j < ns_interval; ++j)
          wcoeffs(dof++, psize * d + i * nv_interval + j) = 1;
  }
  else
  {
    for (std::size_t d = 0; d < tdim; ++d)
      for (int i = 0; i < ns_interval; ++i)
        for (int j = 0; j < ns_interval; ++j)
          for (int k = 0; k < ns_interval; ++k)
            wcoeffs(dof++, psize * d + i * nv_interval * nv_interval
                               + j * nv_interval + k)
                = 1;
  }

  // Create coefficients for additional polynomials in the div space
  for (int i = 0; i < pow(degree, tdim - 1); ++i)
  {
    std::vector<int> indices(tdim - 1);
    if (tdim == 2)
      indices[0] = i;
    else
    {
      indices[0] = i / degree;
      indices[1] = i % degree;
    }

    for (std::size_t d = 0; d < tdim; ++d)
    {
      int n = 0;
      std::vector<double> integrand(pts.extent(0));
      for (std::size_t j = 0; j < integrand.size(); ++j)
        integrand[j] = std::pow(pts(j, d), degree);
      for (std::size_t c = 0; c < tdim; ++c)
      {
        if (c != d)
        {
          for (std::size_t j = 0; j < integrand.size(); ++j)
            integrand[j] *= std::pow(pts(j, c), indices[n]);
          ++n;
        }
      }

      for (std::size_t k = 0; k < psize; ++k)
      {
        double w_sum = 0.0;
        for (std::size_t j = 0; j < qwts.size(); ++j)
          w_sum += qwts[j] * integrand[j] * phi(0, k, j);

        wcoeffs(dof, k + psize * d) = w_sum;
      }
      ++dof;
    }
  }

  std::array<std::vector<impl::mdarray2_t>, 4> x;
  std::array<std::vector<impl::mdarray4_t>, 4> M;

  for (std::size_t i = 0; i < tdim - 1; ++i)
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, i);
    x[i] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
    M[i] = std::vector(num_ent, impl::mdarray4_t(0, tdim, 0, 1));
  }

  {
    FiniteElement moment_space
        = element::create_lagrange(facettype, degree - 1, lvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_normal_integral_moments(
        moment_space, celltype, tdim, 2 * degree - 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[tdim - 1].emplace_back(_x[i], xshape[0], xshape[1]);
      M[tdim - 1].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2],
                               Mshape[3]);
    }
  }

  // Add integral moments on interior
  if (degree > 1)
  {
    auto [_x, xshape, _M, Mshape] = moments::make_dot_integral_moments(
        element::create_nce(celltype, degree - 1, lvariant, true), celltype,
        tdim, 2 * degree - 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[tdim].emplace_back(_x[i], xshape[0], xshape[1]);
      M[tdim].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2], Mshape[3]);
    }
  }
  else
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, tdim);
    x[tdim] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
    M[tdim] = std::vector(num_ent, impl::mdarray4_t(0, tdim, 0, 1));
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

  return FiniteElement(element::family::RT, celltype, degree, {tdim},
                       impl::mdspan2_t(wcoeffs.data(), wcoeffs.extents()),
                       xview, Mview, 0, maps::type::contravariantPiola,
                       discontinuous, degree - 1, degree, lvariant);
}
//-----------------------------------------------------------------------------
FiniteElement basix::element::create_nce(cell::type celltype, int degree,
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
  const auto [_pts, wts] = quadrature::make_quadrature(
      quadrature::type::Default, celltype, 2 * degree);
  impl::cmdspan2_t pts(_pts.data(), wts.size(), _pts.size() / wts.size());
  const auto [_phi, shape] = polyset::tabulate(celltype, degree, 0, pts);
  impl::cmdspan3_t phi(_phi.data(), shape);

  // The number of order (degree) polynomials
  const int psize = phi.extent(1);

  const int edge_count = tdim == 2 ? 4 : 12;
  const int edge_dofs = polyset::dim(cell::type::interval, degree - 1);
  const int face_count = tdim == 2 ? 1 : 6;
  const int face_dofs = 2 * degree * (degree - 1);
  const int volume_count = tdim == 2 ? 0 : 1;
  const int volume_dofs = 3 * degree * (degree - 1) * (degree - 1);
  const std::size_t ndofs = edge_count * edge_dofs + face_count * face_dofs
                            + volume_count * volume_dofs;

  // Create coefficients for order (degree-1) vector polynomials
  impl::mdarray2_t wcoeffs(ndofs, psize * tdim);

  const int nv_interval = polyset::dim(cell::type::interval, degree);
  const int ns_interval = polyset::dim(cell::type::interval, degree - 1);
  int dof = 0;
  if (tdim == 2)
  {
    for (std::size_t d = 0; d < tdim; ++d)
      for (int i = 0; i < ns_interval; ++i)
        for (int j = 0; j < ns_interval; ++j)
          wcoeffs(dof++, psize * d + i * nv_interval + j) = 1;
  }
  else
  {
    for (std::size_t d = 0; d < tdim; ++d)
      for (int i = 0; i < ns_interval; ++i)
        for (int j = 0; j < ns_interval; ++j)
          for (int k = 0; k < ns_interval; ++k)
            wcoeffs(dof++, psize * d + i * nv_interval * nv_interval
                               + j * nv_interval + k)
                = 1;
  }

  // Create coefficients for additional polynomials in the curl space
  std::vector<double> integrand(pts.extent(0));
  switch (tdim)
  {
  case 2:
  {
    for (int i = 0; i < degree; ++i)
    {
      for (std::size_t d = 0; d < tdim; ++d)
      {
        for (std::size_t k = 0; k < integrand.size(); ++k)
          integrand[k] = pts(k, 1 - d);

        for (int j = 1; j < degree; ++j)
          for (std::size_t k = 0; k < integrand.size(); ++k)
            integrand[k] *= pts(k, 1 - d);

        for (int j = 0; j < i; ++j)
          for (std::size_t k = 0; k < integrand.size(); ++k)
            integrand[k] *= pts(k, d);

        for (int k = 0; k < psize; ++k)
        {
          double w_sum = 0.0;
          for (std::size_t k1 = 0; k1 < wts.size(); ++k1)
            w_sum += wts[k1] * integrand[k1] * phi(0, k, k1);
          wcoeffs(dof, k + psize * d) = w_sum;
        }
        ++dof;
      }
    }
    break;
  }
  default:
    for (int i = 0; i < degree; ++i)
    {
      for (int j = 0; j < degree + 1; ++j)
      {
        for (std::size_t c = 0; c < tdim; ++c)
        {
          for (std::size_t d = 0; d < tdim; ++d)
          {
            if (d != c)
            {
              const std::size_t e = 3 - c - d;
              if (c < e and j == degree)
                continue;

              for (std::size_t k1 = 0; k1 < integrand.size(); ++k1)
                integrand[k1] = pts(k1, e);

              for (int k = 1; k < degree; ++k)
                for (std::size_t k1 = 0; k1 < integrand.size(); ++k1)
                  integrand[k1] *= pts(k1, e);

              for (int k = 0; k < i; ++k)
                for (std::size_t k1 = 0; k1 < integrand.size(); ++k1)
                  integrand[k1] *= pts(k1, d);

              for (int k = 0; k < j; ++k)
                for (std::size_t k1 = 0; k1 < integrand.size(); ++k1)
                  integrand[k1] *= pts(k1, c);

              for (int k = 0; k < psize; ++k)
              {
                double w_sum = 0.0;
                for (std::size_t k1 = 0; k1 < wts.size(); ++k1)
                  w_sum += wts[k1] * integrand[k1] * phi(0, k, k1);
                wcoeffs(dof, k + psize * d) = w_sum;
              }
              ++dof;
            }
          }
        }
      }
    }
  }

  std::array<std::vector<impl::mdarray2_t>, 4> x;
  std::array<std::vector<impl::mdarray4_t>, 4> M;

  x[0] = std::vector(cell::num_sub_entities(celltype, 0),
                     impl::mdarray2_t(0, tdim));
  M[0] = std::vector(cell::num_sub_entities(celltype, 0),
                     impl::mdarray4_t(0, tdim, 0, 1));

  {
    FiniteElement edge_moment_space = element::create_lagrange(
        cell::type::interval, degree - 1, lvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_tangent_integral_moments(
        edge_moment_space, celltype, tdim, 2 * degree - 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[1].emplace_back(_x[i], xshape[0], xshape[1]);
      M[1].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2], Mshape[3]);
    }
  }

  // Add integral moments on interior
  if (degree > 1)
  {
    // Face integral moment
    FiniteElement moment_space = element::create_rtc(
        cell::type::quadrilateral, degree - 1, lvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_dot_integral_moments(
        moment_space, celltype, tdim, 2 * degree - 1);
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
    if (degree > 1)
    {
      FiniteElement moment_space = element::create_rtc(
          cell::type::hexahedron, degree - 1, lvariant, true);
      auto [_x, xshape, _M, Mshape] = moments::make_dot_integral_moments(
          moment_space, celltype, tdim, 2 * degree - 1);
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
    std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
    std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
    std::tie(xbuffer, xshape, Mbuffer, Mshape)
        = element::make_discontinuous(xview, Mview, tdim, tdim);
    xview = impl::to_mdspan(xbuffer, xshape);
    Mview = impl::to_mdspan(Mbuffer, Mshape);
  }

  return FiniteElement(element::family::N1E, celltype, degree, {tdim},
                       impl::mdspan2_t(wcoeffs.data(), wcoeffs.extents()),
                       xview, Mview, 0, maps::type::covariantPiola,
                       discontinuous, degree - 1, degree, lvariant);
}
//-----------------------------------------------------------------------------
