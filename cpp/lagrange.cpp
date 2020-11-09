// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "lagrange.h"
#include "dof-permutations.h"
#include "polynomial-set.h"
#include <Eigen/Dense>
#include <iostream>

using namespace libtab;

FiniteElement Lagrange::create(cell::Type celltype, int degree)
{
  if (celltype != cell::Type::interval and celltype != cell::Type::triangle
      and celltype != cell::Type::tetrahedron)
    throw std::runtime_error("Invalid celltype");

  const int ndofs = polyset::size(celltype, degree);
  const int tdim = cell::topological_dimension(celltype);

  // Create points at nodes, ordered by topology (vertices first)
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pt(
      ndofs, tdim);

  std::array<int, 4> entity_dofs = {0, 0, 0, 0};

  std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  if (ndofs == 1)
  {
    if (tdim == 1)
      pt.row(0) << 0.5;
    else if (tdim == 2)
      pt.row(0) << 1.0 / 3, 1.0 / 3;
    else if (tdim == 3)
      pt.row(0) << 0.25, 0.25, 0.25;
    entity_dofs[tdim] = 1;
  }
  else
  {
    entity_dofs[0] = 1;
    entity_dofs[1] = degree - 1;
    if (tdim > 1 and degree > 1)
      entity_dofs[2] = degree - 2;
    if (tdim > 2 and degree > 2)
      entity_dofs[3] = degree - 3;

    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        geometry = cell::geometry(celltype);
    int c = 0;
    for (std::size_t dim = 0; dim < topology.size(); ++dim)
    {
      for (std::size_t i = 0; i < topology[dim].size(); ++i)
      {
        const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor>
            entity_geom = cell::sub_entity_geometry(celltype, dim, i);

        Eigen::ArrayXd point = entity_geom.row(0);
        cell::Type ct = cell::simplex_type(dim);

        const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor>
            lattice = cell::create_lattice(ct, degree, false);
        for (int j = 0; j < lattice.rows(); ++j)
        {
          pt.row(c) = entity_geom.row(0);
          for (int k = 0; k < entity_geom.rows() - 1; ++k)
          {
            pt.row(c) += (entity_geom.row(k + 1) - entity_geom.row(0))
                         * lattice(j, k);
          }
          ++c;
        }
      }
    }
  }

  // Initial coefficients are Identity Matrix
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(ndofs, ndofs);

  // Point evaluation of basis
  Eigen::MatrixXd dualmat = polyset::tabulate(celltype, degree, 0, pt)[0];

  auto new_coeffs
      = FiniteElement::compute_expansion_coefficents(coeffs, dualmat);

  int perm_count = 0;
  for (int i = 1; i < tdim; ++i)
    perm_count += topology[i].size() * i;

  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      base_permutations(perm_count, ndofs);
  for (int i = 0; i < ndofs; ++i)
    base_permutations.col(i) = i;

  if (celltype == cell::Type::triangle)
  {
    Eigen::Array<int, Eigen::Dynamic, 1> edge_ref
        = dofperms::interval_reflection(degree - 1);
    for (int edge = 0; edge < 3; ++edge)
    {
      const int start = 3 + edge_ref.size() * edge;
      for (int i = 0; i < edge_ref.size(); ++i)
        base_permutations(edge, start + i) = start + edge_ref[i];
    }
  }
  else if (celltype == cell::Type::tetrahedron)
  {
    Eigen::Array<int, Eigen::Dynamic, 1> edge_ref
        = dofperms::interval_reflection(degree - 1);
    for (int edge = 0; edge < 6; ++edge)
    {
      const int start = 4 + edge_ref.size() * edge;
      for (int i = 0; i < edge_ref.size(); ++i)
        base_permutations(edge, start + i) = start + edge_ref[i];
    }
    Eigen::Array<int, Eigen::Dynamic, 1> face_ref
        = dofperms::triangle_reflection(degree - 2);
    Eigen::Array<int, Eigen::Dynamic, 1> face_rot
        = dofperms::triangle_rotation(degree - 2);
    for (int face = 0; face < 4; ++face)
    {
      const int start = 4 + edge_ref.size() * 6 + face_ref.size() * face;
      for (int i = 0; i < face_rot.size(); ++i)
        base_permutations(6 + 2 * face, start + i) = start + face_rot[i];
      for (int i = 0; i < face_ref.size(); ++i)
        base_permutations(6 + 2 * face + 1, start + i) = start + face_ref[i];
    }
  }

  FiniteElement el(celltype, degree, 1, new_coeffs, base_permutations);
  return el;
}
//-----------------------------------------------------------------------------
FiniteElement DiscontinuousLagrange::create(cell::Type celltype, int degree)
{
  if (celltype != cell::Type::interval and celltype != cell::Type::triangle
      and celltype != cell::Type::tetrahedron)
    throw std::runtime_error("Invalid celltype");

  // Only tabulate for scalar. Vector spaces can easily be built from the
  // scalar space.

  const int ndofs = polyset::size(celltype, degree);
  const int tdim = cell::topological_dimension(celltype);

  // Create points at nodes, ordered by topology (vertices first)
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pt(
      ndofs, tdim);

  std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> geometry
      = cell::geometry(celltype);

  if (ndofs == 1)
  {
    if (tdim == 1)
      pt.row(0) << 0.5;
    else if (tdim == 2)
      pt.row(0) << 1.0 / 3, 1.0 / 3;
    else if (tdim == 3)
      pt.row(0) << 0.25, 0.25, 0.25;
  }
  else
  {
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        lattice = cell::create_lattice(celltype, degree, true);

    for (int j = 0; j < lattice.rows(); ++j)
    {
      pt.row(j) = geometry.row(0);
      for (int k = 0; k < geometry.rows() - 1; ++k)
        pt.row(j) += (geometry.row(k + 1) - geometry.row(0)) * lattice(j, k);
    }
  }
  // Initial coefficients are Identity Matrix
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(ndofs, ndofs);

  // Point evaluation of basis
  Eigen::MatrixXd dualmat = polyset::tabulate(celltype, degree, 0, pt)[0];

  auto new_coeffs
      = FiniteElement::compute_expansion_coefficents(coeffs, dualmat);

  int perm_count = 0;
  for (int i = 1; i < tdim; ++i)
    perm_count += topology[i].size();

  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      base_permutations(perm_count, ndofs);
  for (int i = 0; i < ndofs; ++i)
    base_permutations.col(i) = i;

  FiniteElement el(celltype, degree, 1, new_coeffs, base_permutations);
  return el;
}
//-----------------------------------------------------------------------------
