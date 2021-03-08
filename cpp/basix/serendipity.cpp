// Copyright (c) 2021 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "serendipity.h"
#include "element-families.h"
#include "lagrange.h"
#include "lattice.h"
#include "log.h"
#include "mappings.h"
#include "moments.h"
#include "polyset.h"
#include "quadrature.h"
#include <Eigen/Dense>
#include <numeric>

using namespace basix;

namespace
{
//----------------------------------------------------------------------------
Eigen::MatrixXd make_serendipity_space_2d(const int degree)
{
  const int ndofs = degree == 1 ? 4 : degree * (degree + 3) / 2 + 3;

  // Evaluate the expansion polynomials at the quadrature points
  auto [Qpts, Qwts] = quadrature::make_quadrature(
      "default", cell::type::quadrilateral, 2 * degree);
  std::cout << "S0 call" << std::endl;
  Eigen::ArrayXXd polyset_at_Qpts
      = polyset::tabulate(cell::type::quadrilateral, degree, 0, Qpts)[0];
  std::cout << "S1 call" << std::endl;
  Eigen::ArrayXXd smaller_polyset_at_Qpts
      = polyset::tabulate(cell::type::triangle, degree, 0, Qpts)[0];
  std::cout << "S2 call" << std::endl;

  const int psize = polyset_at_Qpts.cols();
  const int nv = smaller_polyset_at_Qpts.cols();

  // Create coefficients for order (degree-1) vector polynomials
  Eigen::MatrixXd wcoeffs = Eigen::MatrixXd::Zero(ndofs, psize);
  for (int i = 0; i < nv; ++i)
    for (int k = 0; k < psize; ++k)
      wcoeffs(i, k)
          = (Qwts * smaller_polyset_at_Qpts.col(i) * polyset_at_Qpts.col(k))
                .sum();
  if (degree == 1)
  {
    for (int k = 0; k < psize; ++k)
      wcoeffs(nv, k)
          = (Qwts * Qpts.col(0) * Qpts.col(1) * polyset_at_Qpts.col(k)).sum();
    return wcoeffs;
  }

  for (int k = 0; k < psize; ++k)
  {
    for (int a = 0; a < 2; ++a)
    {
      Eigen::ArrayXd integrand
          = Qwts * Qpts.col(0) * Qpts.col(1) * polyset_at_Qpts.col(k);
      for (int i = 1; i < degree; ++i)
        integrand *= Qpts.col(a);
      wcoeffs(nv + a, k) = integrand.sum();
    }
  }
  return wcoeffs;
}
//----------------------------------------------------------------------------
std::vector<std::array<int, 3>>
serendipity_3d_indices(const int total, const int linear,
                       std::vector<int> done = {})
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
    return serendipity_3d_indices(
        total, linear, {done[0], done[1], total - done[0] - done[1]});

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
Eigen::MatrixXd make_serendipity_space_3d(const int degree)
{
  const int ndofs = degree < 4
                        ? 12 * degree - 4
                        : (degree < 6 ? 3 * degree * degree - 3 * degree + 14
                                      : degree * (degree - 1) * (degree + 1) / 6
                                            + degree * degree + 5 * degree + 4);
  // Number of order (degree) polynomials

  // Evaluate the expansion polynomials at the quadrature points
  auto [Qpts, Qwts] = quadrature::make_quadrature(
      "default", cell::type::hexahedron, 2 * degree);
  Eigen::ArrayXXd polyset_at_Qpts
      = polyset::tabulate(cell::type::hexahedron, degree, 0, Qpts)[0];
  Eigen::ArrayXXd smaller_polyset_at_Qpts
      = polyset::tabulate(cell::type::tetrahedron, degree, 0, Qpts)[0];

  const int psize = polyset_at_Qpts.cols();
  const int nv = smaller_polyset_at_Qpts.cols();

  // Create coefficients for order (degree-1) vector polynomials
  Eigen::MatrixXd wcoeffs = Eigen::MatrixXd::Zero(ndofs, psize);
  for (int i = 0; i < nv; ++i)
    for (int k = 0; k < psize; ++k)
      wcoeffs(i, k)
          = (Qwts * smaller_polyset_at_Qpts.col(i) * polyset_at_Qpts.col(k))
                .sum();

  int c = nv;
  for (int s = 1; s <= 3; ++s)
  {
    std::vector<std::array<int, 3>> indices
        = serendipity_3d_indices(s + degree, s);
    for (std::array<int, 3> i : indices)
    {
      for (int k = 0; k < psize; ++k)
      {
        Eigen::ArrayXd integrand = Qwts * polyset_at_Qpts.col(k);
        for (int d = 0; d < 3; ++d)
          for (int j = 0; j < i[d]; ++j)
            integrand *= Qpts.col(d);
        wcoeffs(c, k) = integrand.sum();
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
    throw std::runtime_error("Invalid celltype");

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  const int tdim = cell::topological_dimension(celltype);

  // Number of dofs and interpolation points
  int quad_deg = 5 * degree;

  Eigen::ArrayXXd points_1d(0, tdim);
  Eigen::MatrixXd matrix_1d(0, 0);

  std::vector<Eigen::MatrixXd> edge_transforms;
  std::vector<Eigen::MatrixXd> face_transforms;

  if (degree >= 2)
  {
    FiniteElement moment_space = create_dpc(cell::type::interval, degree - 2);
    std::tie(points_1d, matrix_1d) = moments::make_integral_moments(
        moment_space, celltype, 1, degree, quad_deg);
    if (tdim > 1)
      edge_transforms
          = moments::create_dot_moment_dof_transformations(moment_space);
  }

  Eigen::ArrayXXd points_2d(0, tdim);
  Eigen::MatrixXd matrix_2d(0, 0);
  if (tdim >= 2 and degree >= 4)
  {
    FiniteElement moment_space
        = create_dpc(cell::type::quadrilateral, degree - 4);
    std::tie(points_2d, matrix_2d) = moments::make_integral_moments(
        moment_space, celltype, 1, degree, quad_deg);
    if (tdim > 2)
      face_transforms
          = moments::create_dot_moment_dof_transformations(moment_space);
  }

  Eigen::ArrayXXd points_3d(0, tdim);
  Eigen::MatrixXd matrix_3d(0, 0);
  if (tdim == 3 and degree >= 6)
  {
    std::tie(points_3d, matrix_3d) = moments::make_integral_moments(
        create_dpc(cell::type::hexahedron, degree - 6), celltype, 1, degree,
        quad_deg);
  }

  const int vertex_count = cell::sub_entity_count(celltype, 0);

  Eigen::ArrayXXd interpolation_points(
      vertex_count + points_1d.rows() + points_2d.rows() + points_3d.rows(),
      tdim);
  Eigen::MatrixXd interpolation_matrix = Eigen::MatrixXd::Zero(
      vertex_count + matrix_1d.rows() + matrix_2d.rows() + matrix_3d.rows(),
      vertex_count + matrix_1d.cols() + matrix_2d.cols() + matrix_3d.cols());

  const xt::xtensor<double, 2> geometry = cell::geometry(celltype);
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      _geometry(geometry.data(), geometry.shape()[0], geometry.shape()[1]);
  interpolation_points.block(0, 0, vertex_count, tdim) = _geometry;
  interpolation_points.block(vertex_count, 0, points_1d.rows(), tdim)
      = points_1d;
  interpolation_points.block(vertex_count + points_1d.rows(), 0,
                             points_2d.rows(), tdim)
      = points_2d;
  interpolation_points.block(vertex_count + points_1d.rows() + points_2d.rows(),
                             0, points_3d.rows(), tdim)
      = points_3d;

  for (int i = 0; i < vertex_count; ++i)
    interpolation_matrix(i, i) = 1;
  interpolation_matrix.block(vertex_count, vertex_count, matrix_1d.rows(),
                             matrix_1d.cols())
      = matrix_1d;
  interpolation_matrix.block(vertex_count + matrix_1d.rows(),
                             vertex_count + matrix_1d.cols(), matrix_2d.rows(),
                             matrix_2d.cols())
      = matrix_2d;
  interpolation_matrix.block(vertex_count + matrix_1d.rows() + matrix_2d.rows(),
                             vertex_count + matrix_1d.cols() + matrix_2d.cols(),
                             matrix_3d.rows(), matrix_3d.cols())
      = matrix_3d;

  Eigen::MatrixXd wcoeffs;
  if (tdim == 1)
    wcoeffs = Eigen::MatrixXd::Identity(degree + 1, degree + 1);
  else if (tdim == 2)
    wcoeffs = make_serendipity_space_2d(degree);
  else if (tdim == 3)
    wcoeffs = make_serendipity_space_3d(degree);

  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t j = 0; j < topology[0].size(); ++j)
    entity_dofs[0].push_back(1);
  for (std::size_t j = 0; j < topology[1].size(); ++j)
    entity_dofs[1].push_back(matrix_1d.rows() / topology[1].size());
  if (tdim >= 2)
    for (std::size_t j = 0; j < topology[2].size(); ++j)
      entity_dofs[2].push_back(matrix_2d.rows() / topology[2].size());
  if (tdim == 3)
    for (std::size_t j = 0; j < topology[3].size(); ++j)
      entity_dofs[3].push_back(matrix_3d.rows() / topology[3].size());

  const int ndofs = interpolation_matrix.rows();

  int transform_count = 0;
  for (std::size_t i = 1; i < topology.size() - 1; ++i)
    transform_count += topology[i].size() * i;

  std::vector<Eigen::MatrixXd> base_transformations(
      transform_count, Eigen::MatrixXd::Identity(ndofs, ndofs));

  if (tdim >= 2 and degree >= 2)
  {
    const int edge_dofs = degree - 1;
    const int num_vertices = topology[0].size();
    const int num_edges = topology[1].size();
    for (int edge = 0; edge < num_edges; ++edge)
    {
      const int start = num_vertices + edge_dofs * edge;
      base_transformations[edge].block(start, start, edge_dofs, edge_dofs)
          = edge_transforms[0];
    }
    if (tdim == 3 and degree >= 4)
    {
      const int face_dofs = face_transforms[0].rows();
      const int num_faces = topology[2].size();
      for (int face = 0; face < num_faces; ++face)
      {
        const int start
            = num_vertices + num_edges * edge_dofs + face * face_dofs;
        base_transformations[num_edges + 2 * face].block(start, start,
                                                         face_dofs, face_dofs)
            = face_transforms[0];
        base_transformations[num_edges + 2 * face + 1].block(
            start, start, face_dofs, face_dofs)
            = face_transforms[1];
      }
    }
  }

  Eigen::MatrixXd coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, interpolation_matrix, interpolation_points, degree);

  return FiniteElement(element::family::Serendipity, celltype, degree, {1},
                       coeffs, entity_dofs, base_transformations,
                       interpolation_points, interpolation_matrix,
                       mapping::type::identity);
}
//-----------------------------------------------------------------------------
