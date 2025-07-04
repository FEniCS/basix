// Copyright (c) 2022 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-hhj.h"
#include "e-lagrange.h"
#include "element-families.h"
#include "maps.h"
#include "math.h"
#include "polyset.h"
#include "quadrature.h"
#include "sobolev-spaces.h"
#include <cmath>

using namespace basix;

//-----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T> basix::element::create_hhj(cell::type celltype, int degree,
                                            bool discontinuous)
{
  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron)
    throw std::runtime_error("Unsupported celltype");

  const std::size_t tdim = cell::topological_dimension(celltype);

  const int nc = tdim * (tdim + 1) / 2;
  const int basis_size
      = polyset::dim(celltype, polyset::type::standard, degree);

  impl::mdarray_t<T, 2> wcoeffs(basis_size * nc, basis_size * tdim * tdim);
  for (std::size_t i = 0; i < tdim; ++i)
  {
    for (std::size_t j = 0; j < tdim; ++j)
    {
      int xoff = i + tdim * j;
      int yoff = i + j;
      if (tdim == 3 and i > 0 and j > 0)
        ++yoff;

      const std::size_t s = basis_size;
      for (std::size_t k = 0; k < s; ++k)
        wcoeffs(yoff * s + k, xoff * s + k) = i == j ? 1.0 : std::sqrt(0.5);
    }
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const auto [gbuffer, gshape] = cell::geometry<T>(celltype);
  impl::mdspan_t<const T, 2> geometry(gbuffer.data(), gshape);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;

  const std::size_t facet_dim = tdim - 1;

  // No DOFs on entities of lower dimension than facets
  for (std::size_t d = 0; d < facet_dim; ++d)
  {
    for (std::size_t e = 0; e < topology[d].size(); ++e)
    {
      x[d].emplace_back(0, tdim);
      M[d].emplace_back(0, tdim * tdim, 0, 1);
    }
  }
  // DOFs on facets
  // These dofs are defined by l(F) = integral(n^t @ F @ n * p),
  // where n is normal to the facet, p is polynomial on the facet
  // of degree `degree`
  auto [_data, _shape] = cell::scaled_facet_normals<T>(celltype);
  impl::mdspan_t<const T, 2> normals(_data.data(), _shape);

  for (std::size_t e = 0; e < topology[facet_dim].size(); ++e)
  {
    cell::type ct = cell::sub_entity_type(celltype, facet_dim, e);

    const std::size_t ndofs = polyset::dim(ct, polyset::type::standard, degree);
    const auto [ptsbuffer, wts] = quadrature::make_quadrature<T>(
        quadrature::type::Default, ct, polyset::type::standard, 2 * degree);
    impl::mdspan_t<const T, 2> pts(ptsbuffer.data(), wts.size(), facet_dim);

    FiniteElement<T> moment_space = create_lagrange<T>(
        ct, degree, element::lagrange_variant::legendre, true);
    const auto [phib, phishape] = moment_space.tabulate(0, pts);
    impl::mdspan_t<const T, 4> moment_values(phib.data(), phishape);

    // Entity coordinates
    const auto [entity_x_buffer, eshape]
        = cell::sub_entity_geometry<T>(celltype, facet_dim, e);
    std::span<const T> x0(entity_x_buffer.data(), eshape[1]);
    impl::mdspan_t<const T, 2> entity_x(entity_x_buffer.data(), eshape);

    // Copy points
    auto& _x = x[facet_dim].emplace_back(pts.extent(0), tdim);
    for (std::size_t p = 0; p < pts.extent(0); ++p)
    {
      for (std::size_t k = 0; k < _x.extent(1); ++k)
        _x(p, k) = x0[k];
      for (std::size_t k0 = 0; k0 + 1 < entity_x.extent(0); ++k0)
        for (std::size_t k1 = 0; k1 < _x.extent(1); ++k1)
          _x(p, k1) += (entity_x(k0 + 1, k1) - x0[k1]) * pts(p, k0);
    }

    auto& _M = M[facet_dim].emplace_back(ndofs, tdim * tdim, pts.extent(0), 1);
    for (int n = 0; n < moment_space.dim(); ++n)
    {
      for (std::size_t q = 0; q < pts.extent(0); ++q)
      {
        for (std::size_t k0 = 0; k0 < tdim; ++k0)
        {
          for (std::size_t k1 = 0; k1 < tdim; ++k1)
          {
            _M(n, tdim * k0 + k1, q, 0) = normals(e, k0) * normals(e, k1)
                                          * wts[q] * moment_values(0, q, n, 0);
          }
        }
      }
    }
  }

  // Interior
  // These DOFs are defined as in section 5.2 of
  // https://doi.org/10.1007/s00211-017-0933-3 (Pechstein & Schoberl, 2017)
  if (tdim == 2)
  {
    if (degree == 0)
    {
      x[tdim].emplace_back(0, tdim);
      M[tdim].emplace_back(0, tdim * tdim, 0, 1);
    }
    else
    {
      const std::size_t ndofs
          = polyset::dim(celltype, polyset::type::standard, degree - 1);

      const auto [ptsbuffer, wts] = quadrature::make_quadrature<T>(
          quadrature::type::Default, celltype, polyset::type::standard,
          2 * degree - 1);
      impl::mdspan_t<const T, 2> pts(ptsbuffer.data(), wts.size(), tdim);

      // Copy points
      auto& _x = x[tdim].emplace_back(pts.extent(0), tdim);
      for (std::size_t p = 0; p < pts.extent(0); ++p)
      {
        for (std::size_t k = 0; k < pts.extent(1); ++k)
          _x(p, k) += pts(p, k);
      }

      auto& _M = M[tdim].emplace_back(ndofs * 3, tdim * tdim, pts.extent(0), 1);

      FiniteElement<T> moment_space = create_lagrange<T>(
          celltype, degree - 1, element::lagrange_variant::legendre, true);
      const auto [phib, phishape] = moment_space.tabulate(0, pts);
      impl::mdspan_t<const T, 4> moment_values(phib.data(), phishape);

      for (int n = 0; n < moment_space.dim(); ++n)
      {
        for (std::size_t q = 0; q < pts.extent(0); ++q)
        {
          // [0, 1], [1, 0]
          _M(n * 3, 1, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 3, 2, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          // [-2, 1], [1, 0]
          _M(n * 3 + 1, 0, q, 0) = -2.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 3 + 1, 1, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 3 + 1, 2, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          // [0, 1], [1, -2]
          _M(n * 3 + 2, 1, q, 0) = -1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 3 + 2, 2, q, 0) = -1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 3 + 2, 3, q, 0) = 2.0 * wts[q] * moment_values(0, q, n, 0);
        }
      }
    }
  }
  else
  {
    const std::size_t ndofs0
        = degree == 0
              ? 0
              : polyset::dim(celltype, polyset::type::standard, degree - 1);
    const std::size_t ndofs1
        = polyset::dim(celltype, polyset::type::standard, degree);

    const auto [ptsbuffer, wts]
        = quadrature::make_quadrature<T>(quadrature::type::Default, celltype,
                                         polyset::type::standard, 2 * degree);
    impl::mdspan_t<const T, 2> pts(ptsbuffer.data(), wts.size(), tdim);

    // Copy points
    auto& _x = x[tdim].emplace_back(pts.extent(0), tdim);
    for (std::size_t p = 0; p < pts.extent(0); ++p)
    {
      for (std::size_t k = 0; k < pts.extent(1); ++k)
        _x(p, k) += pts(p, k);
    }

    auto& _M = M[tdim].emplace_back(ndofs0 * 4 + ndofs1 * 2, tdim * tdim,
                                    pts.extent(0), 1);

    if (degree > 0)
    {
      FiniteElement<T> moment_space = create_lagrange<T>(
          celltype, degree - 1, element::lagrange_variant::legendre, true);
      const auto [phib, phishape] = moment_space.tabulate(0, pts);
      impl::mdspan_t<const T, 4> moment_values(phib.data(), phishape);

      for (int n = 0; n < moment_space.dim(); ++n)
      {
        for (std::size_t q = 0; q < pts.extent(0); ++q)
        {
          // [0, 1, 1], [1, 0, 1], [1, 1, 0]
          _M(n * 4, 1, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4, 2, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4, 3, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4, 5, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4, 6, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4, 7, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);

          // [-6, 1, 1], [1, 0, 1], [1, 1, 0]
          _M(n * 4 + 1, 0, q, 0) = -6.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4 + 1, 1, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4 + 1, 2, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4 + 1, 3, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4 + 1, 5, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4 + 1, 6, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4 + 1, 7, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);

          // [0, 1, 1], [1, -6, 1], [1, 1, 0]
          _M(n * 4 + 2, 1, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4 + 2, 2, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4 + 2, 3, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4 + 2, 4, q, 0) = -6.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4 + 2, 5, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4 + 2, 6, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4 + 2, 7, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);

          // [0, 1, 1], [1, 0, 1], [1, 1, -6]
          _M(n * 4 + 3, 1, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4 + 3, 2, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4 + 3, 3, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4 + 3, 5, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4 + 3, 6, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4 + 3, 7, q, 0) = 1.0 * wts[q] * moment_values(0, q, n, 0);
          _M(n * 4 + 3, 8, q, 0) = -6.0 * wts[q] * moment_values(0, q, n, 0);
        }
      }
    }
    FiniteElement<T> moment_space = create_lagrange<T>(
        celltype, degree, element::lagrange_variant::legendre, true);
    const auto [phib, phishape] = moment_space.tabulate(0, pts);
    impl::mdspan_t<const T, 4> moment_values(phib.data(), phishape);

    for (int n = 0; n < moment_space.dim(); ++n)
    {
      for (std::size_t q = 0; q < pts.extent(0); ++q)
      {
        // [0, 0, -1], [0, 0, 1], [-1, 1, 0]
        _M(ndofs0 * 4 + n * 2, 2, q, 0)
            = -1.0 * wts[q] * moment_values(0, q, n, 0);
        _M(ndofs0 * 4 + n * 2, 5, q, 0)
            = 1.0 * wts[q] * moment_values(0, q, n, 0);
        _M(ndofs0 * 4 + n * 2, 6, q, 0)
            = -1.0 * wts[q] * moment_values(0, q, n, 0);
        _M(ndofs0 * 4 + n * 2, 7, q, 0)
            = 1.0 * wts[q] * moment_values(0, q, n, 0);
        // [0, -1, 0], [-1, 0, 1], [0, 1, 0]
        _M(ndofs0 * 4 + n * 2 + 1, 1, q, 0)
            = -1.0 * wts[q] * moment_values(0, q, n, 0);
        _M(ndofs0 * 4 + n * 2 + 1, 3, q, 0)
            = -1.0 * wts[q] * moment_values(0, q, n, 0);
        _M(ndofs0 * 4 + n * 2 + 1, 5, q, 0)
            = 1.0 * wts[q] * moment_values(0, q, n, 0);
        _M(ndofs0 * 4 + n * 2 + 1, 7, q, 0)
            = 1.0 * wts[q] * moment_values(0, q, n, 0);
      }
    }
  }

  std::array<std::vector<impl::mdspan_t<const T, 2>>, 4> xview
      = impl::to_mdspan(x);
  std::array<std::vector<impl::mdspan_t<const T, 4>>, 4> Mview
      = impl::to_mdspan(M);

  std::array<std::vector<std::vector<T>>, 4> xbuffer;
  std::array<std::vector<std::vector<T>>, 4> Mbuffer;
  if (discontinuous)
  {
    std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
    std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
    std::tie(xbuffer, xshape, Mbuffer, Mshape)
        = element::make_discontinuous(xview, Mview, tdim, tdim * tdim);
    xview = impl::to_mdspan(xbuffer, xshape);
    Mview = impl::to_mdspan(Mbuffer, Mshape);
  }

  sobolev::space space
      = discontinuous ? sobolev::space::L2 : sobolev::space::HDivDiv;

  return FiniteElement<T>(
      element::family::HHJ, celltype, polyset::type::standard, degree,
      {tdim, tdim}, impl::mdspan_t<T, 2>(wcoeffs.data(), wcoeffs.extents()),
      xview, Mview, 0, maps::type::doubleContravariantPiola, space,
      discontinuous, -1, degree, element::lagrange_variant::unset,
      element::dpc_variant::unset);
}
//-----------------------------------------------------------------------------
template FiniteElement<float> element::create_hhj(cell::type, int, bool);
template FiniteElement<double> element::create_hhj(cell::type, int, bool);
//-----------------------------------------------------------------------------
