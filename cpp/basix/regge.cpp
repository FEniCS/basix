// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "regge.h"
#include "dof-transformations.h"
#include "element-families.h"
#include "lattice.h"
#include "maps.h"
#include "polyset.h"
#include <xtensor-blas/xlinalg.hpp>
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
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
create_regge_interpolation(cell::type celltype, int degree)
{
  const std::size_t tdim = cell::topological_dimension(celltype);

  const int basis_size = polyset::dim(celltype, degree);

  const std::size_t ndofs = basis_size * (tdim + 1) * tdim / 2;
  const std::size_t space_size = basis_size * tdim * tdim;

  const std::size_t npoints
      = tdim == 2 ? 3 * (degree + 1) + degree * (degree + 1) / 2
                  : 6 * (degree + 1) + 4 * degree * (degree + 1) / 2
                        + degree * (degree + 1) * (degree - 1) / 6;

  xt::xtensor<double, 2> points({npoints, tdim});
  xt::xtensor<double, 2> matrix
      = xt::zeros<double>({ndofs, npoints * tdim * tdim});

  xt::xtensor<double, 2> dualmat({ndofs, space_size});
  std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const xt::xtensor<double, 2> geometry = cell::geometry(celltype);

  // point and dof counters
  int point_n = 0;
  int dof = 0;
  for (std::size_t dim = 1; dim < topology.size(); ++dim)
  {
    for (std::size_t i = 0; i < topology[dim].size(); ++i)
    {
      const xt::xtensor<double, 2> entity_geom
          = cell::sub_entity_geometry(celltype, dim, i);

      cell::type ct = cell::sub_entity_type(celltype, dim, i);
      auto lattice
          = lattice::create(ct, degree + 2, lattice::type::equispaced, false);
      for (std::size_t j = 0; j < lattice.shape(0); ++j)
      {
        xt::row(points, point_n + j) = xt::row(entity_geom, 0);
        for (std::size_t k = 0; k < entity_geom.shape(0) - 1; ++k)
        {
          xt::row(points, point_n + j)
              += (xt::row(entity_geom, k + 1) - xt::row(entity_geom, 0))
                 * lattice(j, k);
        }
      }

      auto pt_view = xt::view(
          points, xt::range(point_n, point_n + lattice.shape(0)), xt::all());
      xt::xtensor<double, 2> basis
          = xt::view(polyset::tabulate(celltype, degree, 0, pt_view), 0,
                     xt::all(), xt::all());

      // Store up outer(t, t) for all tangents
      std::vector<int>& vert_ids = topology[dim][i];
      std::size_t ntangents = dim * (dim + 1) / 2;
      xt::xtensor<double, 3> vvt(
          {ntangents, geometry.shape(1), geometry.shape(1)});
      std::vector<double> _edge(geometry.shape(1));
      auto edge_t = xt::adapt(_edge);
      int c = 0;
      for (std::size_t s = 0; s < dim; ++s)
      {
        for (std::size_t d = s + 1; d < dim + 1; ++d)
        {
          for (std::size_t p = 0; p < geometry.shape(1); ++p)
            edge_t[p] = geometry(vert_ids[d], p) - geometry(vert_ids[s], p);
          // outer product v.v^T
          xt::view(vvt, c, xt::all(), xt::all())
              = xt::linalg::outer(edge_t, edge_t);
          ++c;
        }
      }

      for (std::size_t k = 0; k < lattice.shape(0); ++k)
      {
        for (std::size_t j = 0; j < ntangents; ++j)
        {
          auto vvt_flat = xt::ravel(xt::view(vvt, j, xt::all(), xt::all()));
          for (std::size_t i = 0; i < tdim * tdim; ++i)
            matrix(dof, point_n + i * npoints) = vvt_flat(i);
          ++dof;
        }
        ++point_n;
      }
    }
  }

  return std::make_pair(points, matrix);
}
//-----------------------------------------------------------------------------
std::pair<std::vector<xt::xtensor<double, 3>>,
          std::vector<xt::xtensor<double, 4>>>
create_regge_interpolation_new(cell::type celltype, int degree)
{
  const std::size_t tdim = cell::topological_dimension(celltype);

  const int basis_size = polyset::dim(celltype, degree);

  const std::size_t ndofs = basis_size * (tdim + 1) * tdim / 2;
  const std::size_t space_size = basis_size * tdim * tdim;

  const std::size_t npoints
      = tdim == 2 ? 3 * (degree + 1) + degree * (degree + 1) / 2
                  : 6 * (degree + 1) + 4 * degree * (degree + 1) / 2
                        + degree * (degree + 1) * (degree - 1) / 6;

  xt::xtensor<double, 2> points({npoints, tdim});
  xt::xtensor<double, 2> matrix
      = xt::zeros<double>({ndofs, npoints * tdim * tdim});

  xt::xtensor<double, 2> dualmat({ndofs, space_size});
  std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const xt::xtensor<double, 2> geometry = cell::geometry(celltype);

  std::vector<xt::xtensor<double, 3>> x(topology.size() - 1);
  std::vector<xt::xtensor<double, 4>> M(topology.size() - 1);

  // point and dof counters
  // int point_n = 0;
  // int dof = 0;
  for (std::size_t d = 1; d < topology.size(); ++d)
  {
    // Loop over entities of dimension dim
    for (std::size_t e = 0; e < topology[d].size(); ++e)
    {
      // Entity coordinates
      const xt::xtensor<double, 2> entity_x
          = cell::sub_entity_geometry(celltype, d, e);

      // Tabulate points in lattice
      cell::type ct = cell::sub_entity_type(celltype, d, e);
      auto lattice
          = lattice::create(ct, degree + 2, lattice::type::equispaced, false);
      auto x0 = xt::row(entity_x, 0);

      if (x[d - 1].size() == 0)
        x[d - 1].resize({topology[d].size(), lattice.shape(0), tdim});

      // Copy points
      for (std::size_t p = 0; p < lattice.shape(0); ++p)
      {
        xt::view(x[d - 1], e, p, xt::all()) = x0;
        for (std::size_t k = 0; k < entity_x.shape(0) - 1; ++k)
        {
          xt::view(x[d - 1], e, p, xt::all())
              += (xt::row(entity_x, k + 1) - x0) * lattice(p, k);
        }
      }

      // auto pt_view = xt::view(
      //     points, xt::range(point_n, point_n + lattice.shape(0)), xt::all());
      // xt::xtensor<double, 2> basis
      //     = xt::view(polyset::tabulate(celltype, degree, 0, pt_view), 0,
      //                xt::all(), xt::all());

      // Store up outer(t, t) for all tangents
      std::vector<int>& vert_ids = topology[d][e];
      std::size_t ntangents = d * (d + 1) / 2;
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
          xt::view(vvt, c, xt::all(), xt::all())
              = xt::linalg::outer(edge_t, edge_t);
          ++c;
        }
      }

      if (M[d - 1].size() == 0)
      {
        M[d - 1] = xt::zeros<double>(
            {lattice.shape(0) * topology[d].size() * ntangents, tdim * tdim,
             topology[d].size(), lattice.shape(0)});
      }

      // std::size_t dofs_per_entity = lattice.shape(0) * topology[d].size();
      for (std::size_t p = 0; p < lattice.shape(0); ++p)
      {
        for (std::size_t j = 0; j < ntangents; ++j)
        {
          auto vvt_flat = xt::ravel(xt::view(vvt, j, xt::all(), xt::all()));
          for (std::size_t i = 0; i < tdim * tdim; ++i)
          {
            // matrix(dof, point_n + i * npoints) = vvt_flat(i);
            M[d - 1](e * lattice.shape(0) * ntangents + p * ntangents + j, i, e,
                     p)
                = vvt_flat(i);
          }
          // ++dof;
        }
        // ++point_n;
      }
    }
  }

  return {x, M};
}
//-----------------------------------------------------------------------------
} // namespace
//-----------------------------------------------------------------------------
FiniteElement basix::create_regge(cell::type celltype, int degree)
{
  const std::size_t tdim = cell::topological_dimension(celltype);
  const int basis_size = polyset::dim(celltype, degree);
  const std::size_t ndofs = basis_size * (tdim + 1) * tdim / 2;

  xt::xtensor<double, 2> wcoeffs = create_regge_space(celltype, degree);
  xt::xtensor<double, 2> points, matrix;
  std::tie(points, matrix) = create_regge_interpolation(celltype, degree);

  // xt::xtensor<double, 2> coeffs = compute_expansion_coefficients(
  //     celltype, wcoeffs, matrix, points, degree);

  auto [x, M] = create_regge_interpolation_new(celltype, degree);
  // for (auto _x : x)
  //   std::cout << _x << std::endl;
  // std::cout << "------" << std::endl;
  // std::cout << points << std::endl;
  // std::cout << "EEEEE" << std::endl;

  // std::cout << "M shape" << std::endl;
  // for (auto _M : M)
  // {
  //   std::cout << "   block" << std::endl;
  //   for (auto s : _M.shape())
  //     std::cout << "    " << s << std::endl;
  // }

  // for (auto _M : M)
  //   std::cout << _M << std::endl;
  // std::cout << "------" << std::endl;
  // std::cout << matrix << std::endl;
  // std::cout << "EEEEE" << std::endl;

  xt::xtensor<double, 3> coeffs
      = compute_expansion_coefficients_new(celltype, wcoeffs, M, x, degree);

  // std::cout << coeffs << std::endl;
  // std::cout << "------" << std::endl;
  // std::cout << coeffs_new << std::endl;
  // std::cout << "EEEEE: "
  //           << xt::sum(xt::square(
  //                  coeffs
  //                  - xt::reshape_view(
  //                      coeffs_new,
  //                      {coeffs_new.shape(0),
  //                       coeffs_new.shape(1) * coeffs_new.shape(2)})))()
  //           << std::endl;

  // Regge has (d+1) dofs on each edge, 3d(d+1)/2 on each face
  // and d(d-1)(d+1) on the interior in 3D
  const int edge_dofs = degree + 1;
  const int face_dofs = 3 * (degree + 1) * degree / 2;
  const int volume_dofs = tdim > 2 ? (degree + 1) * degree * (degree - 1) : 0;

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const int num_vertices = topology[0].size();
  const int num_edges = topology[1].size();
  const int num_faces = topology[2].size();

  std::size_t transform_count = tdim == 2 ? 3 : 14;
  xt::xtensor<double, 3> base_transformations
      = xt::zeros<double>({transform_count, ndofs, ndofs});
  for (std::size_t i = 0; i < base_transformations.shape(0); ++i)
  {
    xt::view(base_transformations, i, xt::all(), xt::all())
        = xt::eye<double>(ndofs);
  }

  const std::vector<int> edge_ref
      = doftransforms::interval_reflection(degree + 1);
  for (int edge = 0; edge < num_edges; ++edge)
  {
    const int start = edge_ref.size() * edge;
    for (std::size_t i = 0; i < edge_ref.size(); ++i)
    {
      base_transformations(edge, start + i, start + i) = 0;
      base_transformations(edge, start + i, start + edge_ref[i]) = 1;
    }
  }
  if (tdim > 2)
  {
    const std::vector<int> face_ref_perm
        = doftransforms::triangle_reflection(degree);
    const std::vector<int> face_rot_perm
        = doftransforms::triangle_rotation(degree);

    xt::xtensor<double, 2> sub_ref({3, 3});
    sub_ref = {{0, 1, 0}, {1, 0, 0}, {0, 0, 1}};
    xt::xtensor<double, 2> sub_rot({3, 3});
    sub_rot = {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}};

    std::array<std::size_t, 2> shape
        = {face_ref_perm.size() * 3, face_ref_perm.size() * 3};
    xt::xtensor<double, 2> face_ref = xt::zeros<double>(shape);
    xt::xtensor<double, 2> face_rot = xt::zeros<double>(shape);

    for (std::size_t i = 0; i < face_ref_perm.size(); ++i)
    {
      xt::view(face_rot, xt::range(3 * i, 3 * i + 3),
               xt::range(3 * face_rot_perm[i], 3 * face_rot_perm[i] + 3))
          = sub_rot;
      xt::view(face_ref, xt::range(3 * i, 3 * i + 3),
               xt::range(3 * face_ref_perm[i], 3 * face_ref_perm[i] + 3))
          = sub_ref;
    }

    for (int face = 0; face < num_faces; ++face)
    {
      const int start = edge_dofs * num_edges + face_dofs * face;
      xt::view(base_transformations, num_edges + 2 * face,
               xt::range(start, start + face_rot.shape(0)),
               xt::range(start, start + face_rot.shape(1)))
          = face_rot;
      xt::view(base_transformations, num_edges + 2 * face + 1,
               xt::range(start, start + face_ref.shape(0)),
               xt::range(start, start + face_ref.shape(1)))
          = face_ref;
    }
  }

  std::vector<std::vector<int>> entity_dofs(topology.size());
  entity_dofs[0].resize(num_vertices, 0);
  entity_dofs[1].resize(num_edges, edge_dofs);
  entity_dofs[2].resize(num_faces, face_dofs);
  if (tdim > 2)
    entity_dofs[3] = {volume_dofs};

  return FiniteElement(element::family::Regge, celltype, degree, {tdim, tdim},
                       coeffs, entity_dofs, base_transformations, points,
                       matrix, maps::type::doubleCovariantPiola);
}
//-----------------------------------------------------------------------------
