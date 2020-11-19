// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "lagrange.h"
#include "dof-permutations.h"
#include "lattice.h"
#include "polynomial-set.h"
#include <Eigen/Dense>
#include <iostream>
#include <numeric>

using namespace libtab;

//----------------------------------------------------------------------------
FiniteElement Lagrange::create(cell::Type celltype, int degree)
{
  if (celltype == cell::Type::point)
    throw std::runtime_error("Invalid celltype");

  const int ndofs = polyset::size(celltype, degree);

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  std::vector<std::vector<int>> entity_dofs(topology.size());

  // Create points at nodes, ordered by topology (vertices first)
  Eigen::ArrayXXd pt(ndofs, topology.size() - 1);
  if (degree == 0)
  {
    pt = lattice::create(celltype, 0, lattice::Type::equispaced, true);
    for (std::size_t i = 0; i < entity_dofs.size(); ++i)
      entity_dofs[i].resize(topology[i].size(), 0);
    entity_dofs[topology.size() - 1][0] = 1;
  }
  else
  {
    int c = 0;
    for (std::size_t dim = 0; dim < topology.size(); ++dim)
    {
      for (std::size_t i = 0; i < topology[dim].size(); ++i)
      {
        const Eigen::ArrayXXd entity_geom
            = cell::sub_entity_geometry(celltype, dim, i);

        if (dim == 0)
        {
          pt.row(c++) = entity_geom.row(0);
          entity_dofs[0].push_back(1);
        }
        else if (dim == topology.size() - 1)
        {
          const Eigen::ArrayXXd lattice = lattice::create(
              celltype, degree, lattice::Type::equispaced, false);
          for (int j = 0; j < lattice.rows(); ++j)
            pt.row(c++) = lattice.row(j);
          entity_dofs[dim].push_back(lattice.rows());
        }
        else
        {
          cell::Type ct = cell::sub_entity_type(celltype, dim, i);
          const Eigen::ArrayXXd lattice
              = lattice::create(ct, degree, lattice::Type::equispaced, false);
          entity_dofs[dim].push_back(lattice.rows());
          for (int j = 0; j < lattice.rows(); ++j)
          {
            pt.row(c) = entity_geom.row(0);
            for (int k = 0; k < lattice.cols(); ++k)
            {
              pt.row(c) += (entity_geom.row(k + 1) - entity_geom.row(0))
                           * lattice(j, k);
            }
            ++c;
          }
        }
      }
    }
  }

  int perm_count = 0;
  for (std::size_t i = 1; i < topology.size() - 1; ++i)
    perm_count += topology[i].size() * i;

  std::vector<Eigen::MatrixXd> base_permutations(
      perm_count, Eigen::MatrixXd::Identity(ndofs, ndofs));
  if (celltype == cell::Type::triangle)
  {
    Eigen::ArrayXi edge_ref = dofperms::interval_reflection(degree - 1);
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
    Eigen::ArrayXi edge_ref = dofperms::interval_reflection(degree - 1);
    for (int edge = 0; edge < 6; ++edge)
    {
      const int start = 4 + edge_ref.size() * edge;
      for (int i = 0; i < edge_ref.size(); ++i)
      {
        base_permutations[edge](start + i, start + i) = 0;
        base_permutations[edge](start + i, start + edge_ref[i]) = 1;
      }
    }
    Eigen::ArrayXi face_ref = dofperms::triangle_reflection(degree - 2);
    Eigen::ArrayXi face_rot = dofperms::triangle_rotation(degree - 2);
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

  // Point evaluation of basis
  Eigen::MatrixXd dualmat = polyset::tabulate(celltype, degree, 0, pt)[0];
  Eigen::MatrixXd coeffs = FiniteElement::compute_expansion_coefficients(
      Eigen::MatrixXd::Identity(ndofs, ndofs), dualmat);

  return FiniteElement(Lagrange::family_name, celltype, degree, {1}, coeffs,
                       entity_dofs, base_permutations);
}
//-----------------------------------------------------------------------------
FiniteElement DiscontinuousLagrange::create(cell::Type celltype, int degree)
{
  if (celltype != cell::Type::interval and celltype != cell::Type::triangle
      and celltype != cell::Type::tetrahedron)
    throw std::runtime_error("Invalid celltype");

  // Only tabulate for scalar. Vector spaces can easily be built from
  // the scalar space.

  const int ndofs = polyset::size(celltype, degree);

  std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t i = 0; i < topology.size(); ++i)
    entity_dofs[i].resize(topology[i].size(), 0);
  entity_dofs[topology.size() - 1][0] = ndofs;

  Eigen::ArrayXXd geometry = cell::geometry(celltype);
  const Eigen::ArrayXXd lattice
      = lattice::create(celltype, degree, lattice::Type::equispaced, true);

  // Create points at nodes, ordered by topology (vertices first)
  Eigen::ArrayXXd pt(ndofs, topology.size() - 1);
  for (int j = 0; j < lattice.rows(); ++j)
  {
    pt.row(j) = geometry.row(0);
    for (int k = 0; k < geometry.rows() - 1; ++k)
      pt.row(j) += (geometry.row(k + 1) - geometry.row(0)) * lattice(j, k);
  }

  // Point evaluation of basis
  Eigen::MatrixXd dualmat = polyset::tabulate(celltype, degree, 0, pt)[0];

  Eigen::MatrixXd coeffs = FiniteElement::compute_expansion_coefficients(
      Eigen::MatrixXd::Identity(ndofs, ndofs), dualmat);

  int perm_count = 0;
  for (std::size_t i = 1; i < topology.size() - 1; ++i)
    perm_count += topology[i].size() * i;

  std::vector<Eigen::MatrixXd> base_permutations(
      perm_count, Eigen::MatrixXd::Identity(ndofs, ndofs));

  return FiniteElement(DiscontinuousLagrange::family_name, celltype, degree,
                       {1}, coeffs, entity_dofs, base_permutations);
}
//-----------------------------------------------------------------------------
