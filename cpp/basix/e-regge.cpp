// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-regge.h"
#include "dof-transformations.h"
#include "element-families.h"
#include "lattice.h"
#include "maps.h"
#include "math.h"
#include "polyset.h"
#include <xtensor/xbuilder.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

namespace
{
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_regge_space(cell::type celltype, int degree)
{
  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron)
    throw std::runtime_error("Unsupported celltype");

  const int tdim = cell::topological_dimension(celltype);
  const int nc = tdim * (tdim + 1) / 2;
  const int basis_size = polyset::dim(celltype, degree);
  const std::size_t ndofs = basis_size * nc;
  const std::size_t psize = basis_size * tdim * tdim;

  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize});
  int s = basis_size;
  for (int i = 0; i < tdim; ++i)
  {
    for (int j = 0; j < tdim; ++j)
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

  return wcoeffs;
}
//-----------------------------------------------------------------------------
std::pair<std::array<std::vector<xt::xtensor<double, 2>>, 4>,
          std::array<std::vector<xt::xtensor<double, 3>>, 4>>
create_regge_interpolation(cell::type celltype, int degree)
{
  const std::size_t tdim = cell::topological_dimension(celltype);
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

    // Loop over entities of dimension dim
    for (std::size_t e = 0; e < topology[d].size(); ++e)
    {
      // Entity coordinates
      const xt::xtensor<double, 2> entity_x
          = cell::sub_entity_geometry(celltype, d, e);

      // Tabulate points in lattice
      cell::type ct = cell::sub_entity_type(celltype, d, e);
      const auto lattice
          = lattice::create(ct, degree + 2, lattice::type::equispaced, false);
      const auto x0 = xt::row(entity_x, 0);
      x[d][e] = xt::xtensor<double, 2>({lattice.shape(0), tdim});

      // Copy points
      for (std::size_t p = 0; p < lattice.shape(0); ++p)
      {
        xt::row(x[d][e], p) = x0;
        for (std::size_t k = 0; k < entity_x.shape(0) - 1; ++k)
        {
          xt::row(x[d][e], p)
              += (xt::row(entity_x, k + 1) - x0) * lattice(p, k);
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
          {lattice.shape(0) * ntangents, tdim * tdim, lattice.shape(0)});
      for (std::size_t p = 0; p < lattice.shape(0); ++p)
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

  return {x, M};
}
//-----------------------------------------------------------------------------
} // namespace
//-----------------------------------------------------------------------------
FiniteElement basix::element::create_regge(cell::type celltype, int degree,
                                           bool discontinuous)
{
  const std::size_t tdim = cell::topological_dimension(celltype);

  const xt::xtensor<double, 2> wcoeffs = create_regge_space(celltype, degree);
  auto [x, M] = create_regge_interpolation(celltype, degree);

  // Regge has (d+1) dofs on each edge, 3d(d+1)/2 on each face
  // and d(d-1)(d+1) on the interior in 3D

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::map<cell::type, xt::xtensor<double, 3>> entity_transformations;

  const std::vector<int> edge_ref
      = doftransforms::interval_reflection(degree + 1);
  const std::array<std::size_t, 3> e_shape
      = {1, edge_ref.size(), edge_ref.size()};
  xt::xtensor<double, 3> et = xt::zeros<double>(e_shape);
  for (std::size_t i = 0; i < edge_ref.size(); ++i)
    et(0, i, edge_ref[i]) = 1;
  entity_transformations[cell::type::interval] = et;

  if (tdim > 2)
  {
    const std::vector<int> face_rot_perm
        = doftransforms::triangle_rotation(degree);
    const std::vector<int> face_ref_perm
        = doftransforms::triangle_reflection(degree);

    const xt::xtensor_fixed<double, xt::xshape<3, 3>> sub_rot
        = {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}};
    const xt::xtensor_fixed<double, xt::xshape<3, 3>> sub_ref
        = {{0, 1, 0}, {1, 0, 0}, {0, 0, 1}};

    const std::array<std::size_t, 3> f_shape
        = {2, face_ref_perm.size() * 3, face_ref_perm.size() * 3};
    xt::xtensor<double, 3> face_trans = xt::zeros<double>(f_shape);

    for (std::size_t i = 0; i < face_ref_perm.size(); ++i)
    {
      xt::view(face_trans, 0, xt::range(3 * i, 3 * i + 3),
               xt::range(3 * face_rot_perm[i], 3 * face_rot_perm[i] + 3))
          = sub_rot;
      xt::view(face_trans, 1, xt::range(3 * i, 3 * i + 3),
               xt::range(3 * face_ref_perm[i], 3 * face_ref_perm[i] + 3))
          = sub_ref;
    }

    entity_transformations[cell::type::triangle] = face_trans;
  }

  if (discontinuous)
  {
    std::tie(x, M, entity_transformations) = element::make_discontinuous(
        x, M, entity_transformations, tdim, tdim * tdim);
  }

  return FiniteElement(element::family::Regge, celltype, degree, {tdim, tdim},
                       wcoeffs, entity_transformations, x, M,
                       maps::type::doubleCovariantPiola, discontinuous, degree,
                       -1);
}
//-----------------------------------------------------------------------------
