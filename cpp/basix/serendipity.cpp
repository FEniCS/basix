// Copyright (c) 2021 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "serendipity.h"
#include "element-families.h"
#include "lagrange.h"
#include "lattice.h"
#include "log.h"
#include "maps.h"
#include "moments.h"
#include "polyset.h"
#include "quadrature.h"
#include <numeric>
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

namespace
{
//----------------------------------------------------------------------------
xt::xtensor<double, 2> make_serendipity_space_2d(int degree)
{
  const std::size_t ndofs = degree == 1 ? 4 : degree * (degree + 3) / 2 + 3;

  // Evaluate the expansion polynomials at the quadrature points
  auto [Qpts, _Qwts] = quadrature::make_quadrature(
      "default", cell::type::quadrilateral, 2 * degree);
  auto Qwts = xt::adapt(_Qwts);

  xt::xtensor<double, 2> polyset_at_Qpts
      = xt::view(polyset::tabulate(cell::type::quadrilateral, degree, 0, Qpts),
                 0, xt::all(), xt::all());
  xt::xtensor<double, 2> smaller_polyset_at_Qpts
      = xt::view(polyset::tabulate(cell::type::triangle, degree, 0, Qpts), 0,
                 xt::all(), xt::all());

  const std::size_t psize = polyset_at_Qpts.shape(1);
  const std::size_t nv = smaller_polyset_at_Qpts.shape(1);

  // Create coefficients for order (degree-1) vector polynomials
  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize});
  for (std::size_t i = 0; i < nv; ++i)
  {
    auto p_i = xt::col(smaller_polyset_at_Qpts, i);
    for (std::size_t k = 0; k < psize; ++k)
      wcoeffs(i, k) = xt::sum(Qwts * p_i * xt::col(polyset_at_Qpts, k))();
  }

  auto q0 = xt::col(Qpts, 0);
  auto q1 = xt::col(Qpts, 1);
  if (degree == 1)
  {
    for (std::size_t k = 0; k < psize; ++k)
      wcoeffs(nv, k) = xt::sum(Qwts * q0 * q1 * xt::col(polyset_at_Qpts, k))();
    return wcoeffs;
  }

  xt::xtensor<double, 1> integrand;
  for (std::size_t k = 0; k < psize; ++k)
  {
    auto pk = xt::col(polyset_at_Qpts, k);
    for (std::size_t a = 0; a < 2; ++a)
    {
      auto q_a = xt::col(Qpts, a);
      integrand = Qwts * q0 * q1 * pk;
      for (int i = 1; i < degree; ++i)
        integrand *= q_a;
      wcoeffs(nv + a, k) = xt::sum(integrand)();
    }
  }

  return wcoeffs;
}
//----------------------------------------------------------------------------
std::vector<std::array<int, 3>>
serendipity_3d_indices(int total, int linear, std::vector<int> done = {})
{
  if (done.size() == 3)
  {
    int count = 0;
    for (int i = 0; i < 3; ++i)
      if (done[i] == 1)
        ++count;

    if (count >= linear)
      return {{done[0], done[1], done[2]}};
    return {};
  }
  if (done.size() == 2)
  {
    return serendipity_3d_indices(
        total, linear, {done[0], done[1], total - done[0] - done[1]});
  }

  std::vector<int> new_done(done.size() + 1);
  int sum_done = 0;
  for (std::size_t i = 0; i < done.size(); ++i)
  {
    new_done[i] = done[i];
    sum_done += done[i];
  }

  std::vector<std::array<int, 3>> out;
  for (int i = 0; i <= total - sum_done; ++i)
  {
    new_done[done.size()] = i;
    for (std::array<int, 3> j : serendipity_3d_indices(total, linear, new_done))
      out.push_back(j);
  }
  return out;
}
//----------------------------------------------------------------------------
xt::xtensor<double, 2> make_serendipity_space_3d(int degree)
{
  const std::size_t ndofs
      = degree < 4 ? 12 * degree - 4
                   : (degree < 6 ? 3 * degree * degree - 3 * degree + 14
                                 : degree * (degree - 1) * (degree + 1) / 6
                                       + degree * degree + 5 * degree + 4);
  // Number of order (degree) polynomials

  // Evaluate the expansion polynomials at the quadrature points
  auto [Qpts, _Qwts] = quadrature::make_quadrature(
      "default", cell::type::hexahedron, 2 * degree);
  auto Qwts = xt::adapt(_Qwts);
  xt::xtensor<double, 2> polyset_at_Qpts
      = xt::view(polyset::tabulate(cell::type::hexahedron, degree, 0, Qpts), 0,
                 xt::all(), xt::all());
  xt::xtensor<double, 2> smaller_polyset_at_Qpts
      = xt::view(polyset::tabulate(cell::type::tetrahedron, degree, 0, Qpts), 0,
                 xt::all(), xt::all());

  const std::size_t psize = polyset_at_Qpts.shape(1);
  const std::size_t nv = smaller_polyset_at_Qpts.shape(1);

  // Create coefficients for order (degree-1) vector polynomials
  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize});
  for (std::size_t i = 0; i < nv; ++i)
  {
    auto p_i = xt::col(smaller_polyset_at_Qpts, i);
    for (std::size_t k = 0; k < psize; ++k)
      wcoeffs(i, k) = xt::sum(Qwts * p_i * xt::col(polyset_at_Qpts, k))();
  }

  std::size_t c = nv;
  xt::xtensor<double, 1> integrand;
  std::vector<std::array<int, 3>> indices;
  for (std::size_t s = 1; s <= 3; ++s)
  {
    indices = serendipity_3d_indices(s + degree, s);
    for (std::array<int, 3> i : indices)
    {
      for (std::size_t k = 0; k < psize; ++k)
      {
        integrand = Qwts * xt::col(polyset_at_Qpts, k);
        for (int d = 0; d < 3; ++d)
        {
          auto q_d = xt::col(Qpts, d);
          for (int j = 0; j < i[d]; ++j)
            integrand *= q_d;
        }

        wcoeffs(c, k) = xt::sum(integrand)();
      }
      ++c;
    }
  }

  return wcoeffs;
}
//----------------------------------------------------------------------------
} // namespace

//----------------------------------------------------------------------------
FiniteElement basix::create_serendipity(cell::type celltype, int degree)
{
  if (celltype != cell::type::interval and celltype != cell::type::quadrilateral
      and celltype != cell::type::hexahedron)
  {
    throw std::runtime_error("Invalid celltype");
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const std::size_t tdim = cell::topological_dimension(celltype);

  // Number of dofs and interpolation points
  int quad_deg = 5 * degree;

  std::vector<xt::xtensor<double, 3>> x(4);
  std::vector<xt::xtensor<double, 4>> M(4);

  const std::size_t vertex_count = cell::sub_entity_count(celltype, 0);
  M[0].resize({vertex_count, 1, vertex_count, 1});
  xt::view(M[0], xt::all(), 0, xt::all(), 0) = xt::eye<double>(vertex_count);

  xt::xtensor<double, 2> points_1d, matrix_1d;
  xt::xtensor<double, 3> edge_transforms, face_transforms;
  if (degree >= 2)
  {
    FiniteElement moment_space = create_dpc(cell::type::interval, degree - 2);
    std::tie(points_1d, matrix_1d)
        = moments::make_integral_moments(moment_space, celltype, 1, quad_deg);
    std::tie(x[1], M[1]) = moments::make_integral_moments_new(
        moment_space, celltype, 1, quad_deg);
    if (tdim > 1)
    {
      edge_transforms
          = moments::create_dot_moment_dof_transformations(moment_space);
    }
  }

  xt::xtensor<double, 2> points_2d, matrix_2d;
  if (tdim >= 2 and degree >= 4)
  {
    FiniteElement moment_space
        = create_dpc(cell::type::quadrilateral, degree - 4);
    std::tie(points_2d, matrix_2d)
        = moments::make_integral_moments(moment_space, celltype, 1, quad_deg);
    std::tie(x[2], M[2]) = moments::make_integral_moments_new(
        moment_space, celltype, 1, quad_deg);
    if (tdim > 2)
    {
      face_transforms
          = moments::create_dot_moment_dof_transformations(moment_space);
    }
  }

  xt::xtensor<double, 2> points_3d, matrix_3d;
  if (tdim == 3 and degree >= 6)
  {
    std::tie(points_3d, matrix_3d) = moments::make_integral_moments(
        create_dpc(cell::type::hexahedron, degree - 6), celltype, 1, quad_deg);
    std::tie(x[3], M[3]) = moments::make_integral_moments_new(
        create_dpc(cell::type::hexahedron, degree - 6), celltype, 1, quad_deg);
  }

  const std::array<std::size_t, 3> num_pts_dim
      = {points_1d.shape(0), points_2d.shape(0), points_3d.shape(0)};
  std::size_t num_pts
      = std::accumulate(num_pts_dim.begin(), num_pts_dim.end(), 0);

  const std::array<std::size_t, 3> num_mat_dim0
      = {matrix_1d.shape(0), matrix_2d.shape(0), matrix_3d.shape(0)};
  const std::array<std::size_t, 3> num_mat_dim1
      = {matrix_1d.shape(1), matrix_2d.shape(1), matrix_3d.shape(1)};
  std::size_t num_mat0
      = std::accumulate(num_mat_dim0.begin(), num_mat_dim0.end(), 0);
  std::size_t num_mat1
      = std::accumulate(num_mat_dim1.begin(), num_mat_dim1.end(), 0);

  xt::xtensor<double, 2> interpolation_points({vertex_count + num_pts, tdim});
  xt::xtensor<double, 2> interpolation_matrix
      = xt::zeros<double>({vertex_count + num_mat0, vertex_count + num_mat1});

  const xt::xtensor<double, 2> geometry = cell::geometry(celltype);
  xt::view(interpolation_points, xt::range(0, vertex_count), xt::all())
      = geometry;

  x[0].resize({vertex_count, 1, tdim});
  xt::view(x[0], xt::all(), 0, xt::all()) = geometry;

  if (points_1d.size() > 0)
  {
    xt::view(interpolation_points,
             xt::range(vertex_count, vertex_count + num_pts_dim[0]), xt::all())
        = points_1d;
  }

  if (points_2d.size() > 0)
  {
    xt::view(interpolation_points,
             xt::range(vertex_count + num_pts_dim[0],
                       vertex_count + num_pts_dim[0] + num_pts_dim[1]),
             xt::all())
        = points_2d;
  }

  if (points_3d.size() > 0)
  {
    xt::view(interpolation_points,
             xt::range(vertex_count + num_pts_dim[0] + num_pts_dim[1],
                       vertex_count + num_pts_dim[0] + num_pts_dim[1]
                           + +num_pts_dim[2]),
             xt::all())
        = points_3d;
  }

  auto r0 = xt::range(0, vertex_count);
  xt::view(interpolation_matrix, r0, r0) = xt::eye<double>(vertex_count);

  xt::view(interpolation_matrix,
           xt::range(vertex_count, vertex_count + num_mat_dim0[0]),
           xt::range(vertex_count, vertex_count + num_mat_dim1[0]))
      = matrix_1d;
  xt::view(interpolation_matrix,
           xt::range(vertex_count + num_mat_dim0[0],
                     vertex_count + num_mat_dim0[0] + num_mat_dim0[1]),
           xt::range(vertex_count + num_mat_dim1[0],
                     vertex_count + num_mat_dim1[0] + +num_mat_dim1[1]))
      = matrix_2d;
  xt::view(interpolation_matrix,
           xt::range(vertex_count + num_mat_dim0[0] + num_mat_dim0[1],
                     vertex_count + num_mat_dim0[0] + num_mat_dim0[1]
                         + num_mat_dim0[2]),
           xt::range(vertex_count + num_mat_dim1[0] + num_mat_dim1[1],
                     vertex_count + num_mat_dim1[0] + +num_mat_dim1[1]
                         + num_mat_dim1[2]))
      = matrix_3d;

  xt::xtensor<double, 2> wcoeffs;
  if (tdim == 1)
    wcoeffs = xt::eye<double>(degree + 1);
  else if (tdim == 2)
    wcoeffs = make_serendipity_space_2d(degree);
  else if (tdim == 3)
    wcoeffs = make_serendipity_space_3d(degree);

  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t j = 0; j < topology[0].size(); ++j)
    entity_dofs[0].push_back(1);
  for (std::size_t j = 0; j < topology[1].size(); ++j)
    entity_dofs[1].push_back(num_mat_dim0[0] / topology[1].size());
  if (tdim >= 2)
    for (std::size_t j = 0; j < topology[2].size(); ++j)
      entity_dofs[2].push_back(num_mat_dim0[1] / topology[2].size());
  if (tdim == 3)
    for (std::size_t j = 0; j < topology[3].size(); ++j)
      entity_dofs[3].push_back(num_mat_dim0[2] / topology[3].size());

  const std::size_t ndofs = interpolation_matrix.shape(0);
  std::size_t transform_count = 0;
  for (std::size_t i = 1; i < topology.size() - 1; ++i)
    transform_count += topology[i].size() * i;

  xt::xtensor<double, 3> base_transformations
      = xt::zeros<double>({transform_count, ndofs, ndofs});
  for (std::size_t i = 0; i < base_transformations.shape(0); ++i)
  {
    xt::view(base_transformations, i, xt::all(), xt::all())
        = xt::eye<double>(ndofs);
  }

  if (tdim >= 2 and degree >= 2)
  {
    const int edge_dofs = degree - 1;
    const int num_vertices = topology[0].size();
    const std::size_t num_edges = topology[1].size();
    for (std::size_t edge = 0; edge < num_edges; ++edge)
    {
      const std::size_t start = num_vertices + edge_dofs * edge;
      auto range = xt::range(start, start + edge_dofs);
      xt::view(base_transformations, edge, range, range)
          = xt::view(edge_transforms, 0, xt::all(), xt::all());
    }
    if (tdim == 3 and degree >= 4)
    {
      const std::size_t face_dofs = face_transforms.shape(1);
      const std::size_t num_faces = topology[2].size();
      for (std::size_t face = 0; face < num_faces; ++face)
      {
        const std::size_t start
            = num_vertices + num_edges * edge_dofs + face * face_dofs;
        auto range = xt::range(start, start + face_dofs);
        xt::view(base_transformations, num_edges + 2 * face, range, range)
            = xt::view(face_transforms, 0, xt::all(), xt::all());
        xt::view(base_transformations, num_edges + 2 * face + 1, range, range)
            = xt::view(face_transforms, 1, xt::all(), xt::all());
      }
    }
  }

  xt::xtensor<double, 3> coeffs
      = compute_expansion_coefficients_new(celltype, wcoeffs, M, x, degree);
  return FiniteElement(element::family::Serendipity, celltype, degree, {1},
                       coeffs, entity_dofs, base_transformations,
                       interpolation_points, interpolation_matrix,
                       maps::type::identity);
}
//-----------------------------------------------------------------------------
