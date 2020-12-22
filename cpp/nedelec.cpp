// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "nedelec.h"
#include "dof-permutations.h"
#include "lagrange.h"
#include "moments.h"
#include "polyset.h"
#include "quadrature.h"
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
  auto [Qpts, Qwts]
      = quadrature::make_quadrature(cell::type::triangle, 2 * degree);
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
Eigen::MatrixXd create_nedelec_2d_dual(int degree)
{
  // Number of dofs and size of polynomial set P(k+1)
  const int ndofs = 3 * degree + degree * (degree - 1);
  const int psize = (degree + 1) * (degree + 2) / 2;

  // Dual space
  Eigen::MatrixXd dual = Eigen::MatrixXd::Zero(ndofs, psize * 2);

  // dof counter
  const int quad_deg = 5 * degree;

  // Integral representation for the boundary (edge) dofs
  dual.block(0, 0, 3 * degree, psize * 2)
      = moments::make_tangent_integral_moments(
          create_dlagrange(cell::type::interval, degree - 1),
          cell::type::triangle, 2, degree, quad_deg);

  if (degree > 1)
  {
    // Interior integral moment
    dual.block(3 * degree, 0, degree * (degree - 1), psize * 2)
        = moments::make_integral_moments(
            create_dlagrange(cell::type::triangle, degree - 2),
            cell::type::triangle, 2, degree, quad_deg);
  }

  return dual;
}
//-----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
create_nedelec_2d_interpolation(int degree)
{
  // TODO: fix interpolation for higher order elements
  if (degree > 2)
    return {{}, {}};

  // Number of dofs and interpolation points
  int quad_deg = 5 * degree;

  Eigen::ArrayXXd points_1d;
  Eigen::MatrixXd matrix_1d;
  std::tie(points_1d, matrix_1d)
      = moments::make_tangent_integral_moments_interpolation(
          create_dlagrange(cell::type::interval, degree - 1),
          cell::type::triangle, 2, degree, quad_deg);

  if (degree == 1)
    return {points_1d, matrix_1d};

  Eigen::ArrayXXd points_2d;
  Eigen::MatrixXd matrix_2d;
  std::tie(points_2d, matrix_2d) = moments::make_integral_moments_interpolation(
      create_dlagrange(cell::type::triangle, degree - 2), cell::type::triangle,
      2, degree, quad_deg);

  Eigen::ArrayXXd points(points_1d.rows() + points_2d.rows(), 2);
  Eigen::MatrixXd matrix(matrix_1d.rows() + matrix_2d.rows(),
                         matrix_1d.cols() + matrix_2d.cols());
  matrix.setZero();
  assert(points_1d.rows() + points_2d.rows() * 2
         == matrix_1d.cols() + matrix_2d.cols());
  assert(matrix_1d.rows() + matrix_2d.rows() == (degree + 1) * (degree + 2));

  points.block(0, 0, points_1d.rows(), 2) = points_1d;
  points.block(points_1d.rows(), 0, points_2d.rows(), 2) = points_2d;

  for (int i = 0; i < 2; ++i)
  {
    const int r1d = matrix_1d.rows();
    const int r2d = matrix_2d.rows();
    const int c1d = matrix_1d.cols() / 2;
    const int c2d = matrix_2d.cols() / 2;
    matrix.block(0, i * (c1d + c2d), r1d, c1d)
        = matrix_1d.block(0, i * c1d, r1d, c1d);
    matrix.block(r1d, i * (c1d + c2d) + c1d, r2d, c2d)
        = matrix_2d.block(0, i * c2d, r2d, c2d);
  }

  return {points, matrix};
}
//-----------------------------------------------------------------------------
std::vector<Eigen::MatrixXd> create_nedelec_2d_base_perms(int degree)
{
  const int ndofs = degree * (degree + 2);
  std::vector<Eigen::MatrixXd> base_permutations(
      3, Eigen::MatrixXd::Identity(ndofs, ndofs));

  Eigen::ArrayXi edge_ref = dofperms::interval_reflection(degree);
  Eigen::ArrayXXd edge_dir
      = dofperms::interval_reflection_tangent_directions(degree);
  for (int edge = 0; edge < 3; ++edge)
  {
    const int start = edge_ref.size() * edge;
    for (int i = 0; i < edge_ref.size(); ++i)
    {
      base_permutations[edge](start + i, start + i) = 0;
      base_permutations[edge](start + i, start + edge_ref[i]) = 1;
    }
    Eigen::MatrixXd directions = Eigen::MatrixXd::Identity(ndofs, ndofs);
    directions.block(edge_dir.rows() * edge, edge_dir.cols() * edge,
                     edge_dir.rows(), edge_dir.cols())
        = edge_dir;
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
  auto [Qpts, Qwts]
      = quadrature::make_quadrature(cell::type::tetrahedron, 2 * degree);
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
Eigen::MatrixXd create_nedelec_3d_dual(int degree)
{
  const int tdim = 3;

  // Size of polynomial set P(k+1)
  const int psize = (degree + 1) * (degree + 2) * (degree + 3) / 6;

  // Work out number of dofs
  const int ndofs = 6 * degree + 4 * degree * (degree - 1)
                    + (degree - 2) * (degree - 1) * degree / 2;
  Eigen::MatrixXd dual = Eigen::MatrixXd::Zero(ndofs, psize * tdim);

  // Create quadrature scheme on the edge
  const int quad_deg = 5 * degree;

  // Integral representation for the boundary (edge) dofs
  dual.block(0, 0, 6 * degree, psize * 3)
      = moments::make_tangent_integral_moments(
          create_dlagrange(cell::type::interval, degree - 1),
          cell::type::tetrahedron, 3, degree, quad_deg);

  if (degree > 1)
  {
    // Integral moments on faces
    dual.block(6 * degree, 0, 4 * (degree - 1) * degree, psize * 3)
        = moments::make_integral_moments(
            create_dlagrange(cell::type::triangle, degree - 2),
            cell::type::tetrahedron, 3, degree, quad_deg);
  }

  if (degree > 2)
  {
    // Interior integral moment
    dual.block(6 * degree + 4 * degree * (degree - 1), 0,
               (degree - 2) * (degree - 1) * degree / 2, psize * 3)
        = moments::make_integral_moments(
            create_dlagrange(cell::type::tetrahedron, degree - 3),
            cell::type::tetrahedron, 3, degree, quad_deg);
  }

  return dual;
}
//-----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
create_nedelec_3d_interpolation(int degree)
{
  // TODO: fix interpolation for higher order elements
  if (degree > 1)
    return {{}, {}};

  // Number of dofs and interpolation points
  int quad_deg = 5 * degree;

  Eigen::ArrayXXd points_1d;
  Eigen::MatrixXd matrix_1d;
  std::tie(points_1d, matrix_1d)
      = moments::make_tangent_integral_moments_interpolation(
          create_dlagrange(cell::type::interval, degree - 1),
          cell::type::tetrahedron, 3, degree, quad_deg);

  if (degree == 1)
    return {points_1d, matrix_1d};

  Eigen::ArrayXXd points_2d;
  Eigen::MatrixXd matrix_2d;
  std::tie(points_2d, matrix_2d) = moments::make_integral_moments_interpolation(
      create_dlagrange(cell::type::triangle, degree - 2),
      cell::type::tetrahedron, 3, degree, quad_deg);

  if (degree == 2)
  {
    Eigen::ArrayXXd points(points_1d.rows() + points_2d.rows(), 3);
    Eigen::MatrixXd matrix(matrix_1d.rows() + matrix_2d.rows(),
                           matrix_1d.cols() + matrix_2d.cols());
    matrix.setZero();

    points.block(0, 0, points_1d.rows(), 2) = points_1d;
    points.block(points_1d.rows(), 0, points_2d.rows(), 2) = points_2d;

    for (int i = 0; i < 3; ++i)
    {
      const int r1d = matrix_1d.rows();
      const int r2d = matrix_2d.rows();
      const int c1d = matrix_1d.cols() / 3;
      const int c2d = matrix_2d.cols() / 3;
      matrix.block(0, i * (c1d + c2d), r1d, c1d)
          = matrix_1d.block(0, i * c1d, r1d, c1d);
      matrix.block(r1d, i * (c1d + c2d) + c1d, r2d, c2d)
          = matrix_2d.block(0, i * c2d, r2d, c2d);
    }

    return {points, matrix};
  }

  Eigen::ArrayXXd points_3d;
  Eigen::MatrixXd matrix_3d;
  std::tie(points_3d, matrix_3d) = moments::make_integral_moments_interpolation(
      create_dlagrange(cell::type::tetrahedron, degree - 3),
      cell::type::tetrahedron, 3, degree, quad_deg);

  Eigen::ArrayXXd points(points_1d.rows() + points_2d.rows() + points_3d.rows(),
                         3);
  Eigen::MatrixXd matrix(matrix_1d.rows() + matrix_2d.rows() + matrix_3d.rows(),
                         matrix_1d.cols() + matrix_2d.cols()
                             + matrix_3d.cols());
  matrix.setZero();
  points.block(0, 0, points_1d.rows(), 2) = points_1d;
  points.block(points_1d.rows(), 0, points_2d.rows(), 2) = points_2d;
  points.block(points_1d.rows() + points_2d.rows(), 0, points_3d.rows(), 2)
      = points_3d;

  for (int i = 0; i < 3; ++i)
  {
    const int r1d = matrix_1d.rows();
    const int r2d = matrix_2d.rows();
    const int r3d = matrix_3d.rows();
    const int c1d = matrix_1d.cols() / 3;
    const int c2d = matrix_2d.cols() / 3;
    const int c3d = matrix_3d.cols() / 3;
    matrix.block(0, i * (c1d + c2d + c3d), r1d, c1d)
        = matrix_1d.block(0, i * c1d, r1d, c1d);
    matrix.block(r1d, i * (c1d + c2d + c3d) + c1d, r2d, c2d)
        = matrix_2d.block(0, i * c2d, r2d, c2d);
    matrix.block(r1d + r2d, i * (c1d + c2d + c3d) + c1d + c2d, r2d, c2d)
        = matrix_3d.block(0, i * c3d, r3d, c3d);
  }

  return {points, matrix};
}
//-----------------------------------------------------------------------------
std::vector<Eigen::MatrixXd> create_nedelec_3d_base_perms(int degree)
{
  const int ndofs = 6 * degree + 4 * degree * (degree - 1)
                    + (degree - 2) * (degree - 1) * degree / 2;
  std::vector<Eigen::MatrixXd> base_permutations(
      14, Eigen::MatrixXd::Identity(ndofs, ndofs));

  Eigen::ArrayXi edge_ref = dofperms::interval_reflection(degree);
  Eigen::ArrayXXd edge_dir
      = dofperms::interval_reflection_tangent_directions(degree);
  for (int edge = 0; edge < 6; ++edge)
  {
    const int start = edge_ref.size() * edge;
    for (int i = 0; i < edge_ref.size(); ++i)
    {
      base_permutations[edge](start + i, start + i) = 0;
      base_permutations[edge](start + i, start + edge_ref[i]) = 1;
    }
    Eigen::MatrixXd directions = Eigen::MatrixXd::Identity(ndofs, ndofs);
    directions.block(edge_dir.rows() * edge, edge_dir.cols() * edge,
                     edge_dir.rows(), edge_dir.cols())
        = edge_dir;
    base_permutations[edge] *= directions;
  }

  // Faces
  Eigen::ArrayXi face_rot = dofperms::triangle_rotation(degree - 1);
  Eigen::ArrayXi face_ref = dofperms::triangle_reflection(degree - 1);
  Eigen::ArrayXXd face_dir_ref
      = dofperms::triangle_reflection_tangent_directions(degree - 1);
  Eigen::ArrayXXd face_dir_rot
      = dofperms::triangle_rotation_tangent_directions(degree - 1);
  for (int face = 0; face < 4; ++face)
  {
    const int start = edge_ref.size() * 6 + face_ref.size() * 2 * face;
    for (int i = 0; i < face_rot.size(); ++i)
    {
      for (int b = 0; b < 2; ++b)
      {
        const int p = 6 + 2 * face;
        const int p1 = start + 2 * i + b;
        base_permutations[p](p1, start + i * 2 + b) = 0;
        base_permutations[p](p1, start + face_rot[i] * 2 + b) = 1;
        base_permutations[p + 1](p1, start + i * 2 + b) = 0;
        base_permutations[p + 1](p1, start + face_ref[i] * 2 + b) = 1;
      }
    }
    // Rotate face
    Eigen::MatrixXd rotation = Eigen::MatrixXd::Identity(ndofs, ndofs);
    rotation.block(edge_dir.rows() * 6 + face_dir_rot.rows() * face,
                   edge_dir.cols() * 6 + face_dir_rot.rows() * face,
                   face_dir_rot.rows(), face_dir_rot.cols())
        = face_dir_rot;
    base_permutations[6 + 2 * face] *= rotation;

    // Reflect face
    Eigen::MatrixXd reflection = Eigen::MatrixXd::Identity(ndofs, ndofs);
    reflection.block(edge_dir.rows() * 6 + face_dir_ref.rows() * face,
                     edge_dir.cols() * 6 + face_dir_ref.rows() * face,
                     face_dir_ref.rows(), face_dir_ref.cols())
        = face_dir_ref;
    base_permutations[6 + 2 * face + 1] *= reflection;
  }

  return base_permutations;
}

//-----------------------------------------------------------------------------
Eigen::MatrixXd create_nedelec2_2d_dual(int degree)
{
  // Number of dofs and size of polynomial set P(k+1)
  const int ndofs = (degree + 1) * (degree + 2);
  const int psize = (degree + 1) * (degree + 2) / 2;

  // Dual space
  Eigen::MatrixXd dual = Eigen::MatrixXd::Zero(ndofs, psize * 2);

  // dof counter
  int quad_deg = 5 * degree;

  // Integral representation for the boundary (edge) dofs
  dual.block(0, 0, 3 * (degree + 1), psize * 2)
      = moments::make_tangent_integral_moments(
          create_dlagrange(cell::type::interval, degree), cell::type::triangle,
          2, degree, quad_deg);

  if (degree > 1)
  {
    // Interior integral moment
    dual.block(3 * (degree + 1), 0, (degree - 1) * (degree + 1), psize * 2)
        = moments::make_dot_integral_moments(
            create_rt(cell::type::triangle, degree - 1), cell::type::triangle,
            2, degree, quad_deg);
  }

  return dual;
}
//-----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
create_nedelec2_2d_interpolation(int degree)
{
  // TODO: fix interpolation for higher order elements
  if (degree > 1)
    return {{}, {}};

  // Number of dofs and interpolation points
  int quad_deg = 5 * degree;

  Eigen::ArrayXXd points_1d;
  Eigen::MatrixXd matrix_1d;
  std::tie(points_1d, matrix_1d)
      = moments::make_tangent_integral_moments_interpolation(
          create_dlagrange(cell::type::interval, degree), cell::type::triangle,
          2, degree, quad_deg);

  if (degree == 1)
    return {points_1d, matrix_1d};

  Eigen::ArrayXXd points_2d;
  Eigen::MatrixXd matrix_2d;
  std::tie(points_2d, matrix_2d)
      = moments::make_dot_integral_moments_interpolation(
          create_rt(cell::type::triangle, degree - 1), cell::type::triangle, 2,
          degree, quad_deg);

  Eigen::ArrayXXd points(points_1d.rows() + points_2d.rows(), 2);
  Eigen::MatrixXd matrix(matrix_1d.rows() + matrix_2d.rows(),
                         matrix_1d.cols() + matrix_2d.cols());
  matrix.setZero();
  assert(points_1d.rows() + points_2d.rows() * 2
         == matrix_1d.cols() + matrix_2d.cols());
  assert(matrix_1d.rows() + matrix_2d.rows() == (degree + 1) * (degree + 2));

  points.block(0, 0, points_1d.rows(), 2) = points_1d;
  points.block(points_1d.rows(), 0, points_2d.rows(), 2) = points_2d;

  for (int i = 0; i < 2; ++i)
  {
    const int r1d = matrix_1d.rows();
    const int r2d = matrix_2d.rows();
    const int c1d = matrix_1d.cols() / 2;
    const int c2d = matrix_2d.cols() / 2;
    matrix.block(0, i * (c1d + c2d), r1d, c1d)
        = matrix_1d.block(0, i * c1d, r1d, c1d);
    matrix.block(r1d, i * (c1d + c2d) + c1d, r2d, c2d)
        = matrix_2d.block(0, i * c2d, r2d, c2d);
  }

  return {points, matrix};
}
//-----------------------------------------------------------------------------
std::vector<Eigen::MatrixXd> create_nedelec2_2d_base_permutations(int degree)
{
  const int ndofs = (degree + 1) * (degree + 2);
  std::vector<Eigen::MatrixXd> base_permutations(
      3, Eigen::MatrixXd::Identity(ndofs, ndofs));

  Eigen::ArrayXi edge_ref = dofperms::interval_reflection(degree + 1);
  Eigen::ArrayXXd edge_dir
      = dofperms::interval_reflection_tangent_directions(degree + 1);
  for (int edge = 0; edge < 3; ++edge)
  {
    const int start = edge_ref.size() * edge;
    for (int i = 0; i < edge_ref.size(); ++i)
    {
      base_permutations[edge](start + i, start + i) = 0;
      base_permutations[edge](start + i, start + edge_ref[i]) = 1;
    }
    Eigen::MatrixXd directions = Eigen::MatrixXd::Identity(ndofs, ndofs);
    directions.block(edge_dir.rows() * edge, edge_dir.cols() * edge,
                     edge_dir.rows(), edge_dir.cols())
        = edge_dir;
    base_permutations[edge] *= directions;
  }

  return base_permutations;
}
//-----------------------------------------------------------------------------
Eigen::MatrixXd create_nedelec2_3d_dual(int degree)
{
  const int tdim = 3;

  // Size of polynomial set P(k+1)
  const int psize = (degree + 1) * (degree + 2) * (degree + 3) / 6;

  // Work out number of dofs
  const int ndofs = (degree + 1) * (degree + 2) * (degree + 3) / 2;

  Eigen::MatrixXd dual = Eigen::MatrixXd::Zero(ndofs, psize * tdim);

  // Create quadrature scheme on the edge
  int quad_deg = 5 * degree;

  // Integral representation for the boundary (edge) dofs
  dual.block(0, 0, 6 * (degree + 1), psize * 3)
      = moments::make_tangent_integral_moments(
          create_dlagrange(cell::type::interval, degree),
          cell::type::tetrahedron, 3, degree, quad_deg);

  if (degree > 1)
  {
    // Integral moments on faces
    dual.block(6 * (degree + 1), 0, 4 * (degree - 1) * (degree + 1), psize * 3)
        = moments::make_dot_integral_moments(
            create_rt(cell::type::triangle, degree - 1),
            cell::type::tetrahedron, 3, degree, quad_deg);
  }

  if (degree > 2)
  {
    // Interior integral moment
    dual.block((6 + 4 * (degree - 1)) * (degree + 1), 0,
               (degree - 1) * (degree - 2) * (degree + 1) / 2, psize * 3)
        = moments::make_integral_moments(
            create_dlagrange(cell::type::tetrahedron, degree - 2),
            cell::type::tetrahedron, 3, degree, quad_deg);
  }

  return dual;
}
//-----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
create_nedelec2_3d_interpolation(int degree)
{
  // TODO
  Eigen::ArrayXXd points(0, 0);
  Eigen::MatrixXd matrix(0, 0);
  return {points, matrix};
}
//-----------------------------------------------------------------------------
std::vector<Eigen::MatrixXd> create_nedelec2_3d_base_permutations(int degree)
{
  // TODO
  const int ndofs = (degree + 1) * (degree + 2) * (degree + 3) / 2;
  std::vector<Eigen::MatrixXd> base_permutations(
      14, Eigen::MatrixXd::Identity(ndofs, ndofs));

  Eigen::ArrayXi edge_ref = dofperms::interval_reflection(degree + 1);
  Eigen::ArrayXXd edge_dir
      = dofperms::interval_reflection_tangent_directions(degree + 1);
  for (int edge = 0; edge < 6; ++edge)
  {
    const int start = edge_ref.size() * edge;
    for (int i = 0; i < edge_ref.size(); ++i)
    {
      base_permutations[edge](start + i, start + i) = 0;
      base_permutations[edge](start + i, start + edge_ref[i]) = 1;
    }
    Eigen::MatrixXd directions = Eigen::MatrixXd::Identity(ndofs, ndofs);
    directions.block(edge_dir.rows() * edge, edge_dir.cols() * edge,
                     edge_dir.rows(), edge_dir.cols())
        = edge_dir;
    base_permutations[edge] *= directions;
  }

  // Faces
  Eigen::MatrixXd face_rot = dofperms::triangle_rt_rotation(degree - 1);
  Eigen::MatrixXd face_ref = dofperms::triangle_rt_reflection(degree - 1);
  for (int face = 0; face < 4; ++face)
  {
    const int start = edge_ref.size() * 6 + face_ref.size() * 2 * face;
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
FiniteElement basix::create_nedelec(cell::type celltype, int degree,
                                     const std::string& name)
{
  Eigen::MatrixXd wcoeffs;
  Eigen::MatrixXd dual;
  Eigen::ArrayXXd points;
  Eigen::MatrixXd interp_matrix;
  std::vector<Eigen::MatrixXd> perms;
  std::vector<Eigen::MatrixXd> directions;
  if (celltype == cell::type::triangle)
  {
    wcoeffs = create_nedelec_2d_space(degree);
    std::tie(points, interp_matrix) = create_nedelec_2d_interpolation(degree);
    dual = create_nedelec_2d_dual(degree);
    perms = create_nedelec_2d_base_perms(degree);
  }
  else if (celltype == cell::type::tetrahedron)
  {
    wcoeffs = create_nedelec_3d_space(degree);
    std::tie(points, interp_matrix) = create_nedelec_3d_interpolation(degree);
    dual = create_nedelec_3d_dual(degree);
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

  const Eigen::MatrixXd coeffs = compute_expansion_coefficients(wcoeffs, dual);
  return FiniteElement(name, celltype, degree, {tdim}, coeffs, entity_dofs,
                       perms, points, interp_matrix, "covariant piola");
}
//-----------------------------------------------------------------------------
FiniteElement basix::create_nedelec2(cell::type celltype, int degree,
                                      const std::string& name)
{
  const int tdim = cell::topological_dimension(celltype);
  const int psize = polyset::dim(celltype, degree);
  Eigen::MatrixXd wcoeffs
      = Eigen::MatrixXd::Identity(tdim * psize, tdim * psize);

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  Eigen::MatrixXd dual;
  Eigen::ArrayXXd points;
  Eigen::MatrixXd interp_matrix;
  std::vector<Eigen::MatrixXd> base_permutations;

  if (celltype == cell::type::triangle)
  {
    dual = create_nedelec2_2d_dual(degree);
    std::tie(points, interp_matrix) = create_nedelec2_2d_interpolation(degree);
    base_permutations = create_nedelec2_2d_base_permutations(degree);
  }
  else if (celltype == cell::type::tetrahedron)
  {
    dual = create_nedelec2_3d_dual(degree);
    std::tie(points, interp_matrix) = create_nedelec2_3d_interpolation(degree);
    base_permutations = create_nedelec2_3d_base_permutations(degree);
  }
  else
    throw std::runtime_error("Invalid celltype in Nedelec");

  const Eigen::MatrixXd coeffs = compute_expansion_coefficients(wcoeffs, dual);

  // Nedelec(2nd kind) has (d+1) dofs on each edge, (d+1)(d-1) on each face
  // and (d-2)(d-1)(d+1)/2 on the interior in 3D
  std::vector<std::vector<int>> entity_dofs(topology.size());
  entity_dofs[0].resize(topology[0].size(), 0);
  entity_dofs[1].resize(topology[1].size(), degree + 1);
  entity_dofs[2].resize(topology[2].size(), (degree + 1) * (degree - 1));
  if (tdim > 2)
    entity_dofs[3] = {(degree - 2) * (degree - 1) * (degree + 1) / 2};

  return FiniteElement(name, celltype, degree, {tdim}, coeffs, entity_dofs,
                       base_permutations, points, interp_matrix,
                       "covariant piola");
}
//-----------------------------------------------------------------------------
