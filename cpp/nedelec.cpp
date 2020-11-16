// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "nedelec.h"
#include "dof-permutations.h"
#include "integral-moments.h"
#include "lagrange.h"
#include "polynomial-set.h"
#include "quadrature.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <numeric>
#include <vector>

using namespace libtab;

namespace
{

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_nedelec_2d_space(int degree)
{
  // Number of order (degree) vector polynomials
  const int nv = degree * (degree + 1) / 2;
  // Number of order (degree-1) vector polynomials
  const int ns0 = (degree - 1) * degree / 2;
  // Number of additional polynomials in Nedelec set
  const int ns = degree;

  // Tabulate polynomial set at quadrature points
  auto [Qpts, Qwts]
      = quadrature::make_quadrature(cell::Type::triangle, 2 * degree);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts
      = polyset::tabulate(cell::Type::triangle, degree, 0, Qpts)[0];

  const int psize = Pkp1_at_Qpts.cols();

  // Create coefficients for order (degree-1) vector polynomials
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      wcoeffs(nv * 2 + ns, psize * 2);
  wcoeffs.setZero();
  wcoeffs.block(0, 0, nv, nv) = Eigen::MatrixXd::Identity(nv, nv);
  wcoeffs.block(nv, psize, nv, nv) = Eigen::MatrixXd::Identity(nv, nv);

  // Create coefficients for the additional Nedelec polynomials
  for (int i = 0; i < ns; ++i)
  {
    for (int k = 0; k < psize; ++k)
    {
      auto w0 = Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(1)
                * Pkp1_at_Qpts.col(k);
      wcoeffs(2 * nv + i, k) = w0.sum();

      auto w1 = -Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(0)
                * Pkp1_at_Qpts.col(k);
      wcoeffs(2 * nv + i, k + psize) = w1.sum();
    }
  }
  return wcoeffs;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_nedelec_2d_dual(int degree)
{
  // Number of dofs and size of polynomial set P(k+1)
  const int ndofs = 3 * degree + degree * (degree - 1);
  const int psize = (degree + 1) * (degree + 2) / 2;

  // Dual space
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat(ndofs, psize * 2);
  dualmat.setZero();

  // dof counter
  int quad_deg = 5 * degree;

  // Integral representation for the boundary (edge) dofs
  FiniteElement moment_space_E
      = DiscontinuousLagrange::create(cell::Type::interval, degree - 1);
  dualmat.block(0, 0, 3 * degree, psize * 2)
      = moments::make_tangent_integral_moments(
          moment_space_E, cell::Type::triangle, 2, degree, quad_deg);

  if (degree > 1)
  {
    // Interior integral moment
    FiniteElement moment_space_I
        = DiscontinuousLagrange::create(cell::Type::triangle, degree - 2);
    dualmat.block(3 * degree, 0, degree * (degree - 1), psize * 2)
        = moments::make_integral_moments(moment_space_I, cell::Type::triangle,
                                         2, degree, quad_deg);
  }
  return dualmat;
}
//-----------------------------------------------------------------------------
std::vector<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
create_nedelec_2d_base_perms(int degree)
{
  const int ndofs = degree * (degree + 2);
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      base_permutations(3, Eigen::MatrixXd::Identity(ndofs, ndofs));

  Eigen::Array<int, Eigen::Dynamic, 1> edge_ref
      = dofperms::interval_reflection(degree);
  for (int edge = 0; edge < 3; ++edge)
  {
    const int start = edge_ref.size() * edge;
    for (int i = 0; i < edge_ref.size(); ++i)
    {
      base_permutations[edge](start + i, start + i) = 0;
      base_permutations[edge](start + i, start + edge_ref[i]) = 1;
    }
  }

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> edge_dir
      = dofperms::interval_reflection_tangent_directions(degree);
  for (int edge = 0; edge < 3; ++edge)
  {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        directions = Eigen::MatrixXd::Identity(ndofs, ndofs);
    directions.block(edge_dir.rows() * edge, edge_dir.cols() * edge,
                     edge_dir.rows(), edge_dir.cols())
        = edge_dir;
    base_permutations[edge] *= directions;
  }
  return base_permutations;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_nedelec_3d_space(int degree)
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
      = quadrature::make_quadrature(cell::Type::tetrahedron, 2 * degree);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts
      = polyset::tabulate(cell::Type::tetrahedron, degree, 0, Qpts)[0];
  const int psize = Pkp1_at_Qpts.cols();

  // Create coefficients for order (degree-1) polynomials
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      wcoeffs(ndofs, psize * tdim);
  wcoeffs.setZero();
  for (int i = 0; i < tdim; ++i)
    wcoeffs.block(nv * i, psize * i, nv, nv)
        = Eigen::MatrixXd::Identity(nv, nv);

  // Create coefficients for additional Nedelec polynomials
  for (int i = 0; i < ns; ++i)
  {
    for (int k = 0; k < psize; ++k)
    {
      auto w = Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(2)
               * Pkp1_at_Qpts.col(k);
      // Don't include polynomials (*, *, 0) that are dependant
      if (i >= ns_remove)
        wcoeffs(tdim * nv + i - ns_remove, psize + k) = -w.sum();
      wcoeffs(tdim * nv + i + ns - ns_remove, k) = w.sum();
    }
  }
  for (int i = 0; i < ns; ++i)
  {
    for (int k = 0; k < psize; ++k)
    {
      auto w = Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(1)
               * Pkp1_at_Qpts.col(k);
      wcoeffs(tdim * nv + i + ns * 2 - ns_remove, k) = -w.sum();
      // Don't include polynomials (*, *, 0) that are dependant
      if (i >= ns_remove)
        wcoeffs(tdim * nv + i - ns_remove, psize * 2 + k) = w.sum();
    }
  }
  for (int i = 0; i < ns; ++i)
  {
    for (int k = 0; k < psize; ++k)
    {
      auto w = Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(0)
               * Pkp1_at_Qpts.col(k);
      wcoeffs(tdim * nv + i + ns - ns_remove, psize * 2 + k) = -w.sum();
      wcoeffs(tdim * nv + i + ns * 2 - ns_remove, psize + k) = w.sum();
    }
  }

  return wcoeffs;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_nedelec_3d_dual(int degree)
{
  const int tdim = 3;

  // Size of polynomial set P(k+1)
  const int psize = (degree + 1) * (degree + 2) * (degree + 3) / 6;

  // Work out number of dofs
  const int ndofs = 6 * degree + 4 * degree * (degree - 1)
                    + (degree - 2) * (degree - 1) * degree / 2;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat(ndofs, psize * tdim);
  dualmat.setZero();

  // Create quadrature scheme on the edge
  const int quad_deg = 5 * degree;

  // Integral representation for the boundary (edge) dofs
  FiniteElement moment_space_E
      = DiscontinuousLagrange::create(cell::Type::interval, degree - 1);
  dualmat.block(0, 0, 6 * degree, psize * 3)
      = moments::make_tangent_integral_moments(
          moment_space_E, cell::Type::tetrahedron, 3, degree, quad_deg);

  if (degree > 1)
  {
    // Integral moments on faces
    FiniteElement moment_space_F
        = DiscontinuousLagrange::create(cell::Type::triangle, degree - 2);
    dualmat.block(6 * degree, 0, 4 * (degree - 1) * degree, psize * 3)
        = moments::make_integral_moments(
            moment_space_F, cell::Type::tetrahedron, 3, degree, quad_deg);
  }

  if (degree > 2)
  {
    // Interior integral moment
    FiniteElement moment_space_I
        = DiscontinuousLagrange::create(cell::Type::tetrahedron, degree - 3);
    dualmat.block(6 * degree + 4 * degree * (degree - 1), 0,
                  (degree - 2) * (degree - 1) * degree / 2, psize * 3)
        = moments::make_integral_moments(
            moment_space_I, cell::Type::tetrahedron, 3, degree, quad_deg);
  }

  return dualmat;
}
//-----------------------------------------------------------------------------
std::vector<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
create_nedelec_3d_base_perms(int degree)
{
  const int ndofs = 6 * degree + 4 * degree * (degree - 1)
                    + (degree - 2) * (degree - 1) * degree / 2;
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      base_permutations(14, Eigen::MatrixXd::Identity(ndofs, ndofs));

  Eigen::Array<int, Eigen::Dynamic, 1> edge_ref
      = dofperms::interval_reflection(degree);
  for (int edge = 0; edge < 6; ++edge)
  {
    const int start = edge_ref.size() * edge;
    for (int i = 0; i < edge_ref.size(); ++i)
    {
      base_permutations[edge](start + i, start + i) = 0;
      base_permutations[edge](start + i, start + edge_ref[i]) = 1;
    }
  }
  Eigen::Array<int, Eigen::Dynamic, 1> face_rot
      = dofperms::triangle_rotation(degree - 1);
  Eigen::Array<int, Eigen::Dynamic, 1> face_ref
      = dofperms::triangle_reflection(degree - 1);
  for (int face = 0; face < 4; ++face)
  {
    const int start = edge_ref.size() * 6 + face_ref.size() * 2 * face;
    for (int i = 0; i < face_rot.size(); ++i)
      for (int b = 0; b < 2; ++b)
      {
        base_permutations[6 + 2 * face](start + 2 * i + b, start + i * 2 + b)
            = 0;
        base_permutations[6 + 2 * face](start + 2 * i + b,
                                        start + face_rot[i] * 2 + b)
            = 1;
        base_permutations[6 + 2 * face + 1](start + 2 * i + b,
                                            start + i * 2 + b)
            = 0;
        base_permutations[6 + 2 * face + 1](start + 2 * i + b,
                                            start + face_ref[i] * 2 + b)
            = 1;
      }
  }

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> edge_dir
      = dofperms::interval_reflection_tangent_directions(degree);
  for (int edge = 0; edge < 6; ++edge)
  {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        directions = Eigen::MatrixXd::Identity(ndofs, ndofs);
    directions.block(edge_dir.rows() * edge, edge_dir.cols() * edge,
                     edge_dir.rows(), edge_dir.cols())
        = edge_dir;
    base_permutations[edge] *= directions;
  }

  // Faces
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      face_dir_ref
      = dofperms::triangle_reflection_tangent_directions(degree - 1);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      face_dir_rot = dofperms::triangle_rotation_tangent_directions(degree - 1);
  for (int face = 0; face < 4; ++face)
  {
    // Rotate face
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        rotation = Eigen::MatrixXd::Identity(ndofs, ndofs);
    rotation.block(edge_dir.rows() * 6 + face_dir_rot.rows() * face,
                   edge_dir.cols() * 6 + face_dir_rot.rows() * face,
                   face_dir_rot.rows(), face_dir_rot.cols())
        = face_dir_rot;
    base_permutations[6 + 2 * face] *= rotation;
    // Reflect face
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        reflection = Eigen::MatrixXd::Identity(ndofs, ndofs);
    reflection.block(edge_dir.rows() * 6 + face_dir_ref.rows() * face,
                     edge_dir.cols() * 6 + face_dir_ref.rows() * face,
                     face_dir_ref.rows(), face_dir_ref.cols())
        = face_dir_ref;
    base_permutations[6 + 2 * face + 1] *= reflection;
  }

  return base_permutations;
}
} // namespace

//-----------------------------------------------------------------------------
FiniteElement Nedelec::create(cell::Type celltype, int degree)
{
  const int tdim = cell::topological_dimension(celltype);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> wcoeffs;
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dualmat;
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      perms;
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      directions;

  if (celltype == cell::Type::triangle)
  {
    wcoeffs = create_nedelec_2d_space(degree);
    dualmat = create_nedelec_2d_dual(degree);
    perms = create_nedelec_2d_base_perms(degree);
  }
  else if (celltype == cell::Type::tetrahedron)
  {
    wcoeffs = create_nedelec_3d_space(degree);
    dualmat = create_nedelec_3d_dual(degree);
    perms = create_nedelec_3d_base_perms(degree);
  }
  else
    throw std::runtime_error("Invalid celltype in Nedelec");

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t i = 0; i < topology.size(); ++i)
    entity_dofs[i].resize(topology[i].size(), 0);
  for (int& q : entity_dofs[1])
    q = degree;
  for (int& q : entity_dofs[2])
    q = degree * (degree - 1);
  if (tdim > 2)
    entity_dofs[3] = {degree * (degree - 1) * (degree - 2) / 2};

  auto new_coeffs
      = FiniteElement::compute_expansion_coefficents(wcoeffs, dualmat);
  FiniteElement el(celltype, degree, {tdim}, new_coeffs, entity_dofs, perms);
  return el;
}
//-----------------------------------------------------------------------------
