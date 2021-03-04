// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "nedelec.h"
#include "core/dof-permutations.h"
#include "core/element-families.h"
#include "core/mappings.h"
#include "core/moments.h"
#include "core/polyset.h"
#include "core/quadrature.h"
#include "lagrange.h"
#include "raviart-thomas.h"
#include <Eigen/Dense>
#include <numeric>
#include <vector>

using namespace basix;

namespace
{
//-----------------------------------------------------------------------------
Eigen::MatrixXd create_nedelec_2d_space(int degree)
{
  // Number of order (degree) vector polynomials
  const int nv = degree * (degree + 1) / 2;

  // Number of order (degree-1) vector polynomials
  const int ns0 = (degree - 1) * degree / 2;

  // Number of additional polynomials in Nedelec set
  const int ns = degree;

  // Tabulate polynomial set at quadrature points
  auto [Qpts, Qwts] = quadrature::make_quadrature(
      "default", cell::type::triangle, 2 * degree);
  Eigen::ArrayXXd Pkp1_at_Qpts
      = polyset::tabulate(cell::type::triangle, degree, 0, Qpts)[0];

  const int psize = Pkp1_at_Qpts.cols();

  // Create coefficients for order (degree-1) vector polynomials
  Eigen::MatrixXd wcoeffs = Eigen::MatrixXd::Zero(nv * 2 + ns, psize * 2);
  wcoeffs.block(0, 0, nv, nv) = Eigen::MatrixXd::Identity(nv, nv);
  wcoeffs.block(nv, psize, nv, nv) = Eigen::MatrixXd::Identity(nv, nv);

  // Create coefficients for the additional Nedelec polynomials
  for (int i = 0; i < ns; ++i)
  {
    for (int k = 0; k < psize; ++k)
    {
      wcoeffs(2 * nv + i, k) = (Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(1)
                                * Pkp1_at_Qpts.col(k))
                                   .sum();
      wcoeffs(2 * nv + i, k + psize) = (-Qwts * Pkp1_at_Qpts.col(ns0 + i)
                                        * Qpts.col(0) * Pkp1_at_Qpts.col(k))
                                           .sum();
    }
  }

  return wcoeffs;
}
//-----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
create_nedelec_2d_interpolation(int degree)
{
  // dof counter
  const int quad_deg = 5 * degree;

  // Integral representation for the boundary (edge) dofs
  Eigen::ArrayXXd points_1d;
  Eigen::MatrixXd matrix_1d;
  std::tie(points_1d, matrix_1d) = moments::make_tangent_integral_moments(
      create_dlagrange(cell::type::interval, degree - 1), cell::type::triangle,
      2, degree, quad_deg);

  Eigen::ArrayXXd points_2d(0, 2);
  Eigen::MatrixXd matrix_2d(0, 0);
  if (degree > 1)
  {
    // Interior integral moment
    std::tie(points_2d, matrix_2d) = moments::make_integral_moments(
        create_dlagrange(cell::type::triangle, degree - 2),
        cell::type::triangle, 2, degree, quad_deg);
  }

  return combine_interpolation_data(points_1d, points_2d, {}, matrix_1d,
                                    matrix_2d, {}, 2, 2);
}
//-----------------------------------------------------------------------------
std::vector<Eigen::MatrixXd> create_nedelec_2d_base_perms(int degree)
{
  const int ndofs = degree * (degree + 2);
  std::vector<Eigen::MatrixXd> base_permutations(
      3, Eigen::MatrixXd::Identity(ndofs, ndofs));

  std::vector<int> edge_ref = dofperms::interval_reflection(degree);
  ndarray<double, 2> edge_dir
      = dofperms::interval_reflection_tangent_directions(degree);
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      _edge_dir(edge_dir.data(), edge_dir.shape[0], edge_dir.shape[1]);
  for (int edge = 0; edge < 3; ++edge)
  {
    const int start = edge_ref.size() * edge;
    for (std::size_t i = 0; i < edge_ref.size(); ++i)
    {
      base_permutations[edge](start + i, start + i) = 0;
      base_permutations[edge](start + i, start + edge_ref[i]) = 1;
    }
    Eigen::MatrixXd directions = Eigen::MatrixXd::Identity(ndofs, ndofs);
    directions.block(_edge_dir.rows() * edge, _edge_dir.cols() * edge,
                     _edge_dir.rows(), _edge_dir.cols())
        = _edge_dir;
    base_permutations[edge] *= directions;
  }

  return base_permutations;
}
//-----------------------------------------------------------------------------
Eigen::MatrixXd create_nedelec_3d_space(int degree)
{
  // Reference tetrahedron
  const int tdim = 3;

  // Number of order (degree) vector polynomials
  const int nv = degree * (degree + 1) * (degree + 2) / 6;

  // Number of order (degree-1) vector polynomials
  const int ns0 = (degree - 1) * degree * (degree + 1) / 6;
  // Number of additional Nedelec polynomials that could be added
  const int ns = degree * (degree + 1) / 2;
  // Number of polynomials that would be included that are not independent so
  // are removed
  const int ns_remove = degree * (degree - 1) / 2;

  // Number of dofs in the space, ie size of polynomial set
  const int ndofs = 6 * degree + 4 * degree * (degree - 1)
                    + (degree - 2) * (degree - 1) * degree / 2;

  // Tabulate polynomial basis at quadrature points
  auto [Qpts, Qwts] = quadrature::make_quadrature(
      "default", cell::type::tetrahedron, 2 * degree);
  Eigen::ArrayXXd Pkp1_at_Qpts
      = polyset::tabulate(cell::type::tetrahedron, degree, 0, Qpts)[0];
  const int psize = Pkp1_at_Qpts.cols();

  // Create coefficients for order (degree-1) polynomials
  Eigen::MatrixXd wcoeffs = Eigen::MatrixXd::Zero(ndofs, psize * tdim);
  for (int i = 0; i < tdim; ++i)
  {
    wcoeffs.block(nv * i, psize * i, nv, nv)
        = Eigen::MatrixXd::Identity(nv, nv);
  }

  // Create coefficients for additional Nedelec polynomials
  for (int i = 0; i < ns; ++i)
  {
    for (int k = 0; k < psize; ++k)
    {
      const double w = (Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(2)
                        * Pkp1_at_Qpts.col(k))
                           .sum();
      // Don't include polynomials (*, *, 0) that are dependant
      if (i >= ns_remove)
        wcoeffs(tdim * nv + i - ns_remove, psize + k) = -w;
      wcoeffs(tdim * nv + i + ns - ns_remove, k) = w;
    }
  }

  for (int i = 0; i < ns; ++i)
  {
    for (int k = 0; k < psize; ++k)
    {
      const double w = (Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(1)
                        * Pkp1_at_Qpts.col(k))
                           .sum();
      wcoeffs(tdim * nv + i + ns * 2 - ns_remove, k) = -w;
      // Don't include polynomials (*, *, 0) that are dependant
      if (i >= ns_remove)
        wcoeffs(tdim * nv + i - ns_remove, psize * 2 + k) = w;
    }
  }

  for (int i = 0; i < ns; ++i)
  {
    for (int k = 0; k < psize; ++k)
    {
      const double w = (Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(0)
                        * Pkp1_at_Qpts.col(k))
                           .sum();
      wcoeffs(tdim * nv + i + ns - ns_remove, psize * 2 + k) = -w;
      wcoeffs(tdim * nv + i + ns * 2 - ns_remove, psize + k) = w;
    }
  }

  return wcoeffs;
}
//-----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
create_nedelec_3d_interpolation(int degree)
{
  // Number of dofs and interpolation points
  int quad_deg = 5 * degree;

  Eigen::ArrayXXd points_1d;
  Eigen::MatrixXd matrix_1d;
  std::tie(points_1d, matrix_1d) = moments::make_tangent_integral_moments(
      create_dlagrange(cell::type::interval, degree - 1),
      cell::type::tetrahedron, 3, degree, quad_deg);

  Eigen::ArrayXXd points_2d(0, 3);
  Eigen::MatrixXd matrix_2d(0, 0);
  if (degree > 1)
  {
    std::tie(points_2d, matrix_2d) = moments::make_integral_moments(
        create_dlagrange(cell::type::triangle, degree - 2),
        cell::type::tetrahedron, 3, degree, quad_deg);
  }

  Eigen::ArrayXXd points_3d(0, 3);
  Eigen::MatrixXd matrix_3d(0, 0);
  if (degree > 2)
  {
    std::tie(points_3d, matrix_3d) = moments::make_integral_moments(
        create_dlagrange(cell::type::tetrahedron, degree - 3),
        cell::type::tetrahedron, 3, degree, quad_deg);
  }

  return combine_interpolation_data(points_1d, points_2d, points_3d, matrix_1d,
                                    matrix_2d, matrix_3d, 3, 3);
}
//-----------------------------------------------------------------------------
std::vector<Eigen::MatrixXd> create_nedelec_3d_base_perms(int degree)
{
  const int ndofs = 6 * degree + 4 * degree * (degree - 1)
                    + (degree - 2) * (degree - 1) * degree / 2;
  std::vector<Eigen::MatrixXd> base_permutations(
      14, Eigen::MatrixXd::Identity(ndofs, ndofs));

  std::vector<int> edge_ref = dofperms::interval_reflection(degree);
  ndarray<double, 2> edge_dir
      = dofperms::interval_reflection_tangent_directions(degree);
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      _edge_dir(edge_dir.data(), edge_dir.shape[0], edge_dir.shape[1]);
  for (int edge = 0; edge < 6; ++edge)
  {
    const int start = edge_ref.size() * edge;
    for (std::size_t i = 0; i < edge_ref.size(); ++i)
    {
      base_permutations[edge](start + i, start + i) = 0;
      base_permutations[edge](start + i, start + edge_ref[i]) = 1;
    }
    Eigen::MatrixXd directions = Eigen::MatrixXd::Identity(ndofs, ndofs);
    directions.block(_edge_dir.rows() * edge, _edge_dir.cols() * edge,
                     _edge_dir.rows(), _edge_dir.cols())
        = _edge_dir;
    base_permutations[edge] *= directions;
  }

  // Faces
  const std::vector<int> face_rot = dofperms::triangle_rotation(degree - 1);
  const std::vector<int> face_ref = dofperms::triangle_reflection(degree - 1);
  ndarray<double, 2> face_dir_ref
      = dofperms::triangle_reflection_tangent_directions(degree - 1);
  ndarray<double, 2> face_dir_rot
      = dofperms::triangle_rotation_tangent_directions(degree - 1);
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      _face_dir_ref(face_dir_ref.data(), face_dir_ref.shape[0],
                    face_dir_ref.shape[1]);
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      _face_dir_rot(face_dir_rot.data(), face_dir_rot.shape[0],
                    face_dir_rot.shape[1]);
  for (int face = 0; face < 4; ++face)
  {
    const int start = edge_ref.size() * 6 + face_ref.size() * 2 * face;
    const int p = 6 + 2 * face;
    for (std::size_t i = 0; i < face_rot.size(); ++i)
    {
      for (int b = 0; b < 2; ++b)
      {
        const int p1 = start + 2 * i + b;
        base_permutations[p](p1, start + i * 2 + b) = 0;
        base_permutations[p](p1, start + face_rot[i] * 2 + b) = 1;
        base_permutations[p + 1](p1, start + i * 2 + b) = 0;
        base_permutations[p + 1](p1, start + face_ref[i] * 2 + b) = 1;
      }
    }
    // Rotate face
    Eigen::MatrixXd rotation = Eigen::MatrixXd::Identity(ndofs, ndofs);
    rotation.block(_edge_dir.rows() * 6 + _face_dir_rot.rows() * face,
                   _edge_dir.cols() * 6 + _face_dir_rot.rows() * face,
                   _face_dir_rot.rows(), _face_dir_rot.cols())
        = _face_dir_rot;
    base_permutations[p] *= rotation;

    // Reflect face
    Eigen::MatrixXd reflection = Eigen::MatrixXd::Identity(ndofs, ndofs);
    reflection.block(_edge_dir.rows() * 6 + _face_dir_ref.rows() * face,
                     _edge_dir.cols() * 6 + _face_dir_ref.rows() * face,
                     _face_dir_ref.rows(), _face_dir_ref.cols())
        = _face_dir_ref;
    base_permutations[p + 1] *= reflection;
  }

  return base_permutations;
}

//-----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
create_nedelec2_2d_interpolation(int degree)
{
  // Number of dofs and interpolation points
  int quad_deg = 5 * degree;

  Eigen::ArrayXXd points_1d;
  Eigen::MatrixXd matrix_1d;
  std::tie(points_1d, matrix_1d) = moments::make_tangent_integral_moments(
      create_dlagrange(cell::type::interval, degree), cell::type::triangle, 2,
      degree, quad_deg);

  Eigen::ArrayXXd points_2d(0, 2);
  Eigen::MatrixXd matrix_2d(0, 0);
  if (degree > 1)
  {
    std::tie(points_2d, matrix_2d) = moments::make_dot_integral_moments(
        create_rt(cell::type::triangle, degree - 1), cell::type::triangle, 2,
        degree, quad_deg);
  }

  return combine_interpolation_data(points_1d, points_2d, {}, matrix_1d,
                                    matrix_2d, {}, 2, 2);
}
//-----------------------------------------------------------------------------
std::vector<Eigen::MatrixXd> create_nedelec2_2d_base_permutations(int degree)
{
  const int ndofs = (degree + 1) * (degree + 2);
  std::vector<Eigen::MatrixXd> base_permutations(
      3, Eigen::MatrixXd::Identity(ndofs, ndofs));

  const std::vector<int> edge_ref = dofperms::interval_reflection(degree + 1);
  ndarray<double, 2> edge_dir
      = dofperms::interval_reflection_tangent_directions(degree + 1);
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      _edge_dir(edge_dir.data(), edge_dir.shape[0], edge_dir.shape[1]);
  for (int edge = 0; edge < 3; ++edge)
  {
    const int start = edge_ref.size() * edge;
    for (std::size_t i = 0; i < edge_ref.size(); ++i)
    {
      base_permutations[edge](start + i, start + i) = 0;
      base_permutations[edge](start + i, start + edge_ref[i]) = 1;
    }
    Eigen::MatrixXd directions = Eigen::MatrixXd::Identity(ndofs, ndofs);
    directions.block(_edge_dir.rows() * edge, _edge_dir.cols() * edge,
                     _edge_dir.rows(), _edge_dir.cols())
        = _edge_dir;
    base_permutations[edge] *= directions;
  }

  return base_permutations;
}
//-----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
create_nedelec2_3d_interpolation(int degree)
{
  // Create quadrature scheme on the edge
  int quad_deg = 5 * degree;

  // Integral representation for the boundary (edge) dofs
  Eigen::ArrayXXd points_1d;
  Eigen::MatrixXd matrix_1d;
  std::tie(points_1d, matrix_1d) = moments::make_tangent_integral_moments(
      create_dlagrange(cell::type::interval, degree), cell::type::tetrahedron,
      3, degree, quad_deg);

  Eigen::ArrayXXd points_2d(0, 3);
  Eigen::MatrixXd matrix_2d(0, 0);

  if (degree > 1)
  {
    // Integral moments on faces
    std::tie(points_2d, matrix_2d) = moments::make_dot_integral_moments(
        create_rt(cell::type::triangle, degree - 1), cell::type::tetrahedron, 3,
        degree, quad_deg);
  }

  Eigen::ArrayXXd points_3d(0, 3);
  Eigen::MatrixXd matrix_3d(0, 0);
  if (degree > 2)
  {
    // Interior integral moment
    std::tie(points_3d, matrix_3d) = moments::make_dot_integral_moments(
        create_rt(cell::type::tetrahedron, degree - 2), cell::type::tetrahedron,
        3, degree, quad_deg);
  }

  return combine_interpolation_data(points_1d, points_2d, points_3d, matrix_1d,
                                    matrix_2d, matrix_3d, 3, 3);
}
//-----------------------------------------------------------------------------
std::vector<Eigen::MatrixXd> create_nedelec2_3d_base_permutations(int degree)
{
  const int ndofs = (degree + 1) * (degree + 2) * (degree + 3) / 2;
  std::vector<Eigen::MatrixXd> base_permutations(
      14, Eigen::MatrixXd::Identity(ndofs, ndofs));

  const std::vector<int> edge_ref = dofperms::interval_reflection(degree + 1);
  ndarray<double, 2> edge_dir
      = dofperms::interval_reflection_tangent_directions(degree + 1);
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      _edge_dir(edge_dir.data(), edge_dir.shape[0], edge_dir.shape[1]);
  for (int edge = 0; edge < 6; ++edge)
  {
    const int start = edge_ref.size() * edge;
    for (std::size_t i = 0; i < edge_ref.size(); ++i)
    {
      base_permutations[edge](start + i, start + i) = 0;
      base_permutations[edge](start + i, start + edge_ref[i]) = 1;
    }
    Eigen::MatrixXd directions = Eigen::MatrixXd::Identity(ndofs, ndofs);
    directions.block(_edge_dir.rows() * edge, _edge_dir.cols() * edge,
                     _edge_dir.rows(), _edge_dir.cols())
        = _edge_dir;
    base_permutations[edge] *= directions;
  }

  // Faces
  Eigen::MatrixXd face_rot = dofperms::triangle_rt_rotation(degree - 1);
  Eigen::MatrixXd face_ref = dofperms::triangle_rt_reflection(degree - 1);
  for (int face = 0; face < 4; ++face)
  {
    const int start = edge_ref.size() * 6 + face_ref.rows() * face;
    const int p = 6 + 2 * face;

    base_permutations[p].block(start, start, face_rot.rows(), face_rot.cols())
        = face_rot;
    base_permutations[p + 1].block(start, start, face_ref.rows(),
                                   face_ref.cols())
        = face_ref;
  }

  return base_permutations;
}

} // namespace

//-----------------------------------------------------------------------------
FiniteElement basix::create_nedelec(cell::type celltype, int degree)
{
  Eigen::MatrixXd wcoeffs;
  Eigen::ArrayXXd points;
  Eigen::MatrixXd interp_matrix;
  std::vector<Eigen::MatrixXd> perms;
  std::vector<Eigen::MatrixXd> directions;
  if (celltype == cell::type::triangle)
  {
    wcoeffs = create_nedelec_2d_space(degree);
    std::tie(points, interp_matrix) = create_nedelec_2d_interpolation(degree);
    perms = create_nedelec_2d_base_perms(degree);
  }
  else if (celltype == cell::type::tetrahedron)
  {
    wcoeffs = create_nedelec_3d_space(degree);
    std::tie(points, interp_matrix) = create_nedelec_3d_interpolation(degree);
    perms = create_nedelec_3d_base_perms(degree);
  }
  else
    throw std::runtime_error("Invalid celltype in Nedelec");

  // Nedelec has d dofs on each edge, d(d-1) on each face
  // and d(d-1)(d-2)/2 on the interior in 3D
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  std::vector<std::vector<int>> entity_dofs(topology.size());
  entity_dofs[0].resize(topology[0].size(), 0);
  entity_dofs[1].resize(topology[1].size(), degree);
  entity_dofs[2].resize(topology[2].size(), degree * (degree - 1));
  const int tdim = cell::topological_dimension(celltype);
  if (tdim > 2)
    entity_dofs[3] = {degree * (degree - 1) * (degree - 2) / 2};

  const Eigen::MatrixXd coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, interp_matrix, points, degree);
  return FiniteElement(element::family::N1E, celltype, degree, {tdim}, coeffs,
                       entity_dofs, perms, points, interp_matrix,
                       mapping::type::covariantPiola);
}
//-----------------------------------------------------------------------------
FiniteElement basix::create_nedelec2(cell::type celltype, int degree)
{
  const int tdim = cell::topological_dimension(celltype);
  const int psize = polyset::dim(celltype, degree);
  Eigen::MatrixXd wcoeffs
      = Eigen::MatrixXd::Identity(tdim * psize, tdim * psize);

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  Eigen::ArrayXXd points;
  Eigen::MatrixXd interp_matrix;
  std::vector<Eigen::MatrixXd> base_permutations;

  if (celltype == cell::type::triangle)
  {
    std::tie(points, interp_matrix) = create_nedelec2_2d_interpolation(degree);
    base_permutations = create_nedelec2_2d_base_permutations(degree);
  }
  else if (celltype == cell::type::tetrahedron)
  {
    std::tie(points, interp_matrix) = create_nedelec2_3d_interpolation(degree);
    base_permutations = create_nedelec2_3d_base_permutations(degree);
  }
  else
    throw std::runtime_error("Invalid celltype in Nedelec");

  // Nedelec(2nd kind) has (d+1) dofs on each edge, (d+1)(d-1) on each face
  // and (d-2)(d-1)(d+1)/2 on the interior in 3D
  std::vector<std::vector<int>> entity_dofs(topology.size());
  entity_dofs[0].resize(topology[0].size(), 0);
  entity_dofs[1].resize(topology[1].size(), degree + 1);
  entity_dofs[2].resize(topology[2].size(), (degree + 1) * (degree - 1));
  if (tdim > 2)
    entity_dofs[3] = {(degree - 2) * (degree - 1) * (degree + 1) / 2};

  const Eigen::MatrixXd coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, interp_matrix, points, degree);
  return FiniteElement(element::family::N2E, celltype, degree, {tdim}, coeffs,
                       entity_dofs, base_permutations, points, interp_matrix,
                       mapping::type::covariantPiola);
}
//-----------------------------------------------------------------------------
