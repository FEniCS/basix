// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-regge.h"
#include "e-lagrange.h"
#include "element-families.h"
#include "maps.h"
#include "math.h"
#include "polyset.h"
#include "quadrature.h"
#include <xtensor/xbuilder.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

//-----------------------------------------------------------------------------
FiniteElement basix::element::create_regge(cell::type celltype, int degree,
                                           bool discontinuous)
{
  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron)
    throw std::runtime_error("Unsupported celltype");

  const std::size_t tdim = cell::topological_dimension(celltype);

  const int nc = tdim * (tdim + 1) / 2;
  const int basis_size = polyset::dim(celltype, degree);
  const std::size_t ndofs = basis_size * nc;
  const std::size_t psize = basis_size * tdim * tdim;

  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize});
  int s = basis_size;
  for (std::size_t i = 0; i < tdim; ++i)
  {
    for (std::size_t j = 0; j < tdim; ++j)
    {
      int xoff = i + tdim * j;
      int yoff = i + j;
      if (tdim == 3 and i > 0 and j > 0)
        ++yoff;

      for (std::size_t k = 0; k < s; ++k)
        wcoeffs(yoff * s + k, xoff * s + k) = 1.0;
    }
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const xt::xtensor<double, 2> geometry = cell::geometry(celltype);

  std::array<std::vector<xt::xtensor<double, 4>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  x[0] = std::vector(cell::num_sub_entities(celltype, 0),
                     xt::xtensor<double, 2>({0, tdim}));
  M[0] = std::vector(cell::num_sub_entities(celltype, 0),
                     xt::xtensor<double, 4>({0, tdim * tdim, 0, 1}));

  // Loop over edge and higher dimension entities
  for (std::size_t d = 1; d < topology.size(); ++d)
  {
    x[d].resize(topology[d].size());
    M[d].resize(topology[d].size());

    if (static_cast<std::size_t>(degree) + 1 < d)
    {
      for (std::size_t e = 0; e < topology[d].size(); ++e)
      {
        x[d][e] = xt::xtensor<double, 2>({0, tdim});
        M[d][e] = xt::xtensor<double, 4>({0, tdim * tdim, 0, 1});
      }
    }
    else
    {

      // Loop over entities of dimension dim
      for (std::size_t e = 0; e < topology[d].size(); ++e)
      {
        // Entity coordinates
        const xt::xtensor<double, 2> entity_x
            = cell::sub_entity_geometry(celltype, d, e);

        // Tabulate points in lattice
        cell::type ct = cell::sub_entity_type(celltype, d, e);

        const std::size_t ndofs = polyset::dim(ct, degree + 1 - d);
        const auto [pts, wts]
            = quadrature::make_quadrature(ct, degree + (degree + 1 - d));

        FiniteElement moment_space = create_lagrange(
            ct, degree + 1 - d, element::lagrange_variant::legendre, true);
        const auto moment_values = moment_space.tabulate(0, pts);
        x[d][e] = xt::xtensor<double, 2>({pts.shape(0), tdim});

        // Copy points
        for (std::size_t p = 0; p < pts.shape(0); ++p)
        {
          for (std::size_t j = 0; j < entity_x.shape(1); ++j)
            x[d][e](p, j) = entity_x(0, j);

          for (std::size_t i = 0; i < entity_x.shape(0) - 1; ++i)
            for (std::size_t j = 0; j < entity_x.shape(1); ++j)
              x[d][e](p, j)
                  += (entity_x(i + 1, j) - entity_x(0, j)) * pts(p, i);
        }

        // Store up outer(t, t) for all tangents
        const std::vector<int>& vert_ids = topology[d][e];
        const std::size_t ntangents = d * (d + 1) / 2;
        std::experimental::mdarray<double,
                                   std::experimental::dextents<std::size_t, 3>>
            vvt(ntangents, geometry.shape(1), geometry.shape(1));
        std::vector<double> edge(geometry.shape(1));

        int c = 0;
        for (std::size_t s = 0; s < d; ++s)
        {
          for (std::size_t r = s + 1; r < d + 1; ++r)
          {
            for (std::size_t p = 0; p < geometry.shape(1); ++p)
              edge[p] = geometry(vert_ids[r], p) - geometry(vert_ids[s], p);

            // outer product v.v^T
            auto [buffer, shape] = math::outer_new(edge, edge);
            impl::cmdspan2_t result(buffer.data(), shape);
            for (std::size_t i = 0; i < vvt.extent(1); ++i)
              for (std::size_t j = 0; j < vvt.extent(2); ++j)
                vvt(c, i, j) = result(i, j);

            ++c;
          }
        }

        M[d][e]
            = xt::zeros<double>({ndofs * ntangents, tdim * tdim, pts.shape(0),
                                 static_cast<std::size_t>(1)});
        for (int n = 0; n < moment_space.dim(); ++n)
        {
          for (std::size_t j = 0; j < ntangents; ++j)
          {
            std::vector<double> vvt_flat;
            for (std::size_t i = 0; i < vvt.extent(1); ++i)
              for (std::size_t k = 0; k < vvt.extent(2); ++k)
                vvt_flat.push_back(vvt(j, i, k));
            for (std::size_t q = 0; q < pts.shape(0); ++q)
            {
              for (std::size_t i = 0; i < tdim * tdim; ++i)
              {
                M[d][e](n * ntangents + j, i, q, 0)
                    = vvt_flat[i] * wts[q] * moment_values(0, q, n, 0);
              }
            }
          }
        }
      }
    }
  }

  // Regge has (d+1) dofs on each edge, 3d(d+1)/2 on each face and
  // d(d-1)(d+1) on the interior in 3D

  if (discontinuous)
    std::tie(x, M) = element::make_discontinuous(x, M, tdim, tdim * tdim);

  return FiniteElement(element::family::Regge, celltype, degree, {tdim, tdim},
                       wcoeffs, x, M, 0, maps::type::doubleCovariantPiola,
                       discontinuous, -1, degree);
}
//-----------------------------------------------------------------------------
