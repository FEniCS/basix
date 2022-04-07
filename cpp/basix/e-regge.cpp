// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-regge.h"
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

      xt::view(wcoeffs, xt::range(yoff * s, yoff * s + s),
               xt::range(xoff * s, xoff * s + s))
          = xt::eye<double>(s);
    }
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const xt::xtensor<double, 2> geometry = cell::geometry(celltype);

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  // Loop over edge and higher dimension entities
  for (std::size_t d = 1; d < topology.size(); ++d)
  {
    x[d].resize(topology[d].size());
    M[d].resize(topology[d].size());

    if (static_cast<std::size_t>(degree) + 1 >= d)
    {

      // Loop over entities of dimension dim
      for (std::size_t e = 0; e < topology[d].size(); ++e)
      {
        // Entity coordinates
        const xt::xtensor<double, 2> entity_x
            = cell::sub_entity_geometry(celltype, d, e);

        // Tabulate points in lattice
        cell::type ct = cell::sub_entity_type(celltype, d, e);
        const auto [pts, wts] = quadrature::make_quadrature(ct, degree + 1 - d);
        std::cout << wts[0] << "\n";
        const auto x0 = xt::row(entity_x, 0);
        x[d][e] = xt::xtensor<double, 2>({pts.shape(0), tdim});

        // Copy points
        for (std::size_t p = 0; p < pts.shape(0); ++p)
        {
          xt::row(x[d][e], p) = x0;
          for (std::size_t k = 0; k < entity_x.shape(0) - 1; ++k)
          {
            xt::row(x[d][e], p) += (xt::row(entity_x, k + 1) - x0) * pts(p, k);
          }
        }

        // Store up outer(t, t) for all tangents
        const std::vector<int>& vert_ids = topology[d][e];
        const std::size_t ntangents = d * (d + 1) / 2;
        xt::xtensor<double, 3> vvt(
            {ntangents, geometry.shape(1), geometry.shape(1)});
        std::vector<double> _edge(geometry.shape(1));
        auto edge_t = xt::adapt(_edge);

        int c = 0;
        for (std::size_t s = 0; s < d; ++s)
        {
          for (std::size_t r = s + 1; r < d + 1; ++r)
          {
            for (std::size_t p = 0; p < geometry.shape(1); ++p)
              edge_t[p] = geometry(vert_ids[r], p) - geometry(vert_ids[s], p);

            // outer product v.v^T
            auto result = basix::math::outer(edge_t, edge_t);
            xt::view(vvt, c, xt::all(), xt::all()).assign(result);
            ++c;
          }
        }

        M[d][e] = xt::zeros<double>(
            {pts.shape(0) * ntangents, tdim * tdim, pts.shape(0)});
        for (std::size_t p = 0; p < pts.shape(0); ++p)
        {
          for (std::size_t j = 0; j < ntangents; ++j)
          {
            auto vvt_flat = xt::ravel(xt::view(vvt, j, xt::all(), xt::all()));
            for (std::size_t i = 0; i < tdim * tdim; ++i)
              M[d][e](p * ntangents + j, i, p) = vvt_flat(i);
          }
        }
      }
    }
  }

  // Regge has (d+1) dofs on each edge, 3d(d+1)/2 on each face
  // and d(d-1)(d+1) on the interior in 3D

  if (discontinuous)
  {
    std::tie(x, M) = element::make_discontinuous(x, M, tdim, tdim * tdim);
  }

  return FiniteElement(element::family::Regge, celltype, degree, {tdim, tdim},
                       wcoeffs, x, M, maps::type::doubleCovariantPiola,
                       discontinuous, -1);
}
//-----------------------------------------------------------------------------
