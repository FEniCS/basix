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

using namespace basix;
namespace stdex = std::experimental;
using mdarray3_t = stdex::mdarray<double, stdex::dextents<std::size_t, 3>>;
using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
using cmdspan4_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;

//-----------------------------------------------------------------------------
FiniteElement basix::element::create_hhj(cell::type celltype, int degree,
                                         bool discontinuous)
{
  if (celltype != cell::type::triangle)
    throw std::runtime_error("Unsupported celltype");

  const std::size_t tdim = cell::topological_dimension(celltype);

  const int nc = tdim * (tdim + 1) / 2;
  const int basis_size = polyset::dim(celltype, degree);
  const std::size_t ndofs = basis_size * nc;
  const std::size_t psize = basis_size * tdim * tdim;

  impl::mdarray2_t wcoeffs(ndofs, psize);
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
        wcoeffs(yoff * s + k, xoff * s + k) = 1.0;
    }
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const auto [gbuffer, gshape] = cell::geometry(celltype);
  impl::cmdspan2_t geometry(gbuffer.data(), gshape);

  std::array<std::vector<impl::mdarray2_t>, 4> x;
  std::array<std::vector<impl::mdarray4_t>, 4> M;

  for (std::size_t e = 0; e < topology[0].size(); ++e)
  {
    x[0].emplace_back(0, tdim);
    M[0].emplace_back(0, tdim * tdim, 0, 1);
  }

  // Loop over edge and higher dimension entities
  for (std::size_t d = 1; d < topology.size(); ++d)
  {
    if (static_cast<std::size_t>(degree) + 1 < d)
    {
      for (std::size_t e = 0; e < topology[d].size(); ++e)
      {
        x[d].emplace_back(0, tdim);
        M[d].emplace_back(0, tdim * tdim, 0, 1);
      }
    }
    else
    {

      // Loop over entities of dimension dim
      for (std::size_t e = 0; e < topology[d].size(); ++e)
      {
        // Entity coordinates
        const auto [entity_x_buffer, eshape]
            = cell::sub_entity_geometry(celltype, d, e);
        std::span<const double> x0(entity_x_buffer.data(), eshape[1]);
        impl::cmdspan2_t entity_x(entity_x_buffer.data(), eshape);

        // Tabulate points in lattice
        cell::type ct = cell::sub_entity_type(celltype, d, e);

        const std::size_t ndofs = polyset::dim(ct, degree + 1 - d);
        const auto [ptsbuffer, wts]
            = quadrature::make_quadrature(ct, degree + (degree + 1 - d));
        impl::cmdspan2_t pts(ptsbuffer.data(), wts.size(),
                             ptsbuffer.size() / wts.size());

        FiniteElement moment_space = create_lagrange(
            ct, degree + 1 - d, element::lagrange_variant::legendre, true);
        const auto [phib, phishape] = moment_space.tabulate(0, pts);
        cmdspan4_t moment_values(phib.data(), phishape);

        auto& _x = x[d].emplace_back(pts.extent(0), tdim);

        // Copy points
        for (std::size_t p = 0; p < pts.extent(0); ++p)
        {
          for (std::size_t k = 0; k < _x.extent(1); ++k)
            _x(p, k) = x0[k];

          for (std::size_t k0 = 0; k0 < entity_x.extent(0) - 1; ++k0)
            for (std::size_t k1 = 0; k1 < _x.extent(1); ++k1)
              _x(p, k1) += (entity_x(k0 + 1, k1) - x0[k1]) * pts(p, k0);
        }

        // Store up outer(t, t) for all tangents
        const std::vector<int>& vert_ids = topology[d][e];
        const std::size_t ntangents = d * (d + 1) / 2;
        mdarray3_t vvt(ntangents, geometry.extent(1), geometry.extent(1));
        std::vector<double> edge_t(geometry.extent(1));
        int c = 0;
        for (std::size_t s = 0; s < d; ++s)
        {
          for (std::size_t r = s + 1; r < d + 1; ++r)
          {
            if (geometry.extent(1) != 2)
              throw std::runtime_error("Not implemented");
            edge_t[0] = geometry(vert_ids[s], 1) - geometry(vert_ids[r], 1);
            edge_t[1] = geometry(vert_ids[r], 0) - geometry(vert_ids[s], 0);

            // outer product v.v^T
            const auto [result, shape] = basix::math::outer(edge_t, edge_t);
            for (std::size_t k0 = 0; k0 < shape[0]; ++k0)
              for (std::size_t k1 = 0; k1 < shape[1]; ++k1)
                vvt(c, k0, k1) = result[k0 * shape[1] + k1];
            ++c;
          }
        }

        auto& _M = M[d].emplace_back(ndofs * ntangents, tdim * tdim,
                                     pts.extent(0), 1);
        for (int n = 0; n < moment_space.dim(); ++n)
        {
          for (std::size_t j = 0; j < ntangents; ++j)
          {
            std::vector<double> vvt_flat;
            for (std::size_t k0 = 0; k0 < vvt.extent(1); ++k0)
              for (std::size_t k1 = 0; k1 < vvt.extent(2); ++k1)
                vvt_flat.push_back(vvt(j, k0, k1));
            for (std::size_t q = 0; q < pts.extent(0); ++q)
            {
              for (std::size_t i = 0; i < tdim * tdim; ++i)
              {
                _M(n * ntangents + j, i, q, 0)
                    = vvt_flat[i] * wts[q] * moment_values(0, q, n, 0);
              }
            }
          }
        }
      }
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
        = element::make_discontinuous(xview, Mview, tdim, tdim * tdim);
    xview = impl::to_mdspan(xbuffer, xshape);
    Mview = impl::to_mdspan(Mbuffer, Mshape);
  }

  return FiniteElement(element::family::HHJ, celltype, degree, {tdim, tdim},
                       impl::mdspan2_t(wcoeffs.data(), wcoeffs.extents()),
                       xview, Mview, 0, maps::type::doubleContravariantPiola,
                       discontinuous, -1, degree);
}
//-----------------------------------------------------------------------------
