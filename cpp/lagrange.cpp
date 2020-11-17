// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "lagrange.h"
#include "dof-permutations.h"
#include "lattice.h"
#include "polynomial-set.h"
#include <Eigen/Dense>
#include <iostream>

using namespace libtab;

//----------------------------------------------------------------------------
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

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t i = 0; i < topology.size(); ++i)
    entity_dofs[i].resize(topology[i].size(), 0);

  if (ndofs == 1)
  {
    if (tdim == 1)
      pt.row(0) << 0.5;
    else if (tdim == 2)
      pt.row(0) << 1.0 / 3, 1.0 / 3;
    else if (tdim == 3)
      pt.row(0) << 0.25, 0.25, 0.25;
    entity_dofs[tdim] = {1};
  }
  else
  {
    // One dof on each vertex
    for (int& q : entity_dofs[0])
      q = 1;
    // Degree-1 dofs on each edge
    for (int& q : entity_dofs[1])
      q = degree - 1;
    // Triangle
    if (tdim > 1)
    {
      for (int& q : entity_dofs[2])
        q = (degree - 1) * (degree - 2) / 2;
    }
    // Tetrahedron
    if (tdim > 2)
      entity_dofs[3] = {(degree - 1) * (degree - 2) * (degree - 3) / 6};

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
        cell::Type ct = cell::sub_entity_type(celltype, dim, i);
        const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor>
            lattice
            = lattice::create(ct, degree, lattice::Type::equispaced, false);
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

  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      base_permutations(perm_count, Eigen::MatrixXd::Identity(ndofs, ndofs));

  if (celltype == cell::Type::triangle)
  {
    Eigen::Array<int, Eigen::Dynamic, 1> edge_ref
        = dofperms::interval_reflection(degree - 1);
    for (int edge = 0; edge < 3; ++edge)
    {
      const int start = 3 + edge_ref.size() * edge;
      for (int i = 0; i < edge_ref.size(); ++i)
      {
        base_permutations[edge](start + i, start + i) = 0;
        base_permutations[edge](start + i, start + edge_ref[i]) = 1;
      }
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
      {
        base_permutations[edge](start + i, start + i) = 0;
        base_permutations[edge](start + i, start + edge_ref[i]) = 1;
      }
    }
    Eigen::Array<int, Eigen::Dynamic, 1> face_ref
        = dofperms::triangle_reflection(degree - 2);
    Eigen::Array<int, Eigen::Dynamic, 1> face_rot
        = dofperms::triangle_rotation(degree - 2);
    for (int face = 0; face < 4; ++face)
    {
      const int start = 4 + edge_ref.size() * 6 + face_ref.size() * face;
      for (int i = 0; i < face_rot.size(); ++i)
      {
        base_permutations[6 + 2 * face](start + i, start + i) = 0;
        base_permutations[6 + 2 * face](start + i, start + face_rot[i]) = 1;
        base_permutations[6 + 2 * face + 1](start + i, start + i) = 0;
        base_permutations[6 + 2 * face + 1](start + i, start + face_ref[i]) = 1;
      }
    }
  }

  FiniteElement el(celltype, degree, {1}, new_coeffs, entity_dofs,
                   base_permutations);
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

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t i = 0; i < topology.size(); ++i)
    entity_dofs[i].resize(topology[i].size(), 0);
  entity_dofs[tdim][0] = ndofs;

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
        lattice
        = lattice::create(celltype, degree, lattice::Type::equispaced, true);

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
    perm_count += topology[i].size() * i;

  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      base_permutations(perm_count, Eigen::MatrixXd::Identity(ndofs, ndofs));

  return FiniteElement(celltype, degree, {1}, new_coeffs, entity_dofs,
                       base_permutations);
}
//-----------------------------------------------------------------------------
