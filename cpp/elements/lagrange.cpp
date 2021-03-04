// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "lagrange.h"
#include "core/dof-permutations.h"
#include "core/element-families.h"
#include "core/lattice.h"
#include "core/mappings.h"
#include "core/polyset.h"
#include <Eigen/Dense>
#include <iostream>
#include <numeric>

using namespace basix;

//----------------------------------------------------------------------------
FiniteElement basix::create_lagrange(cell::type celltype, int degree)
{
  if (celltype == cell::type::point)
    throw std::runtime_error("Invalid celltype");

  const int ndofs = polyset::dim(celltype, degree);

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  std::vector<std::vector<int>> entity_dofs(topology.size());

  // Create points at nodes, ordered by topology (vertices first)
  Eigen::ArrayXXd pt(ndofs, topology.size() - 1);
  if (degree == 0)
  {
    pt = lattice::create(celltype, 0, lattice::type::equispaced, true);
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
        const ndarray<double, 2> entity_geom
            = cell::sub_entity_geometry(celltype, dim, i);
        Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::RowMajor>>
            _entity_geom(entity_geom.data(), entity_geom.shape[0],
                         entity_geom.shape[1]);

        if (dim == 0)
        {
          pt.row(c++) = _entity_geom.row(0);
          entity_dofs[0].push_back(1);
        }
        else if (dim == topology.size() - 1)
        {
          const Eigen::ArrayXXd lattice = lattice::create(
              celltype, degree, lattice::type::equispaced, false);
          for (int j = 0; j < lattice.rows(); ++j)
            pt.row(c++) = lattice.row(j);
          entity_dofs[dim].push_back(lattice.rows());
        }
        else
        {
          cell::type ct = cell::sub_entity_type(celltype, dim, i);
          const Eigen::ArrayXXd lattice
              = lattice::create(ct, degree, lattice::type::equispaced, false);
          entity_dofs[dim].push_back(lattice.rows());
          for (int j = 0; j < lattice.rows(); ++j)
          {
            pt.row(c) = _entity_geom.row(0);
            for (int k = 0; k < lattice.cols(); ++k)
            {
              pt.row(c) += (_entity_geom.row(k + 1) - _entity_geom.row(0))
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
  if (celltype == cell::type::interval)
  {
    assert(perm_count == 0);
  }
  else if (celltype == cell::type::triangle)
  {
    const std::vector<int> edge_ref = dofperms::interval_reflection(degree - 1);
    for (int edge = 0; edge < 3; ++edge)
    {
      const int start = 3 + edge_ref.size() * edge;
      for (std::size_t i = 0; i < edge_ref.size(); ++i)
      {
        base_permutations[edge](start + i, start + i) = 0;
        base_permutations[edge](start + i, start + edge_ref[i]) = 1;
      }
    }
  }
  else if (celltype == cell::type::quadrilateral)
  {
    const std::vector<int> edge_ref = dofperms::interval_reflection(degree - 1);
    for (int edge = 0; edge < 4; ++edge)
    {
      const int start = 4 + edge_ref.size() * edge;
      for (std::size_t i = 0; i < edge_ref.size(); ++i)
      {
        base_permutations[edge](start + i, start + i) = 0;
        base_permutations[edge](start + i, start + edge_ref[i]) = 1;
      }
    }
  }
  else if (celltype == cell::type::tetrahedron)
  {
    const std::vector<int> edge_ref = dofperms::interval_reflection(degree - 1);
    for (int edge = 0; edge < 6; ++edge)
    {
      const int start = 4 + edge_ref.size() * edge;
      for (std::size_t i = 0; i < edge_ref.size(); ++i)
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
  else if (celltype == cell::type::hexahedron)
  {
    const std::vector<int> edge_ref = dofperms::interval_reflection(degree - 1);
    for (int edge = 0; edge < 12; ++edge)
    {
      const int start = 8 + edge_ref.size() * edge;
      for (std::size_t i = 0; i < edge_ref.size(); ++i)
      {
        base_permutations[edge](start + i, start + i) = 0;
        base_permutations[edge](start + i, start + edge_ref[i]) = 1;
      }
    }
    Eigen::ArrayXi face_ref = dofperms::quadrilateral_reflection(degree - 1);
    Eigen::ArrayXi face_rot = dofperms::quadrilateral_rotation(degree - 1);
    for (int face = 0; face < 6; ++face)
    {
      const int start = 8 + edge_ref.size() * 12 + face_ref.size() * face;
      for (int i = 0; i < face_rot.size(); ++i)
      {
        base_permutations[12 + 2 * face](start + i, start + i) = 0;
        base_permutations[12 + 2 * face](start + i, start + face_rot[i]) = 1;
        base_permutations[12 + 2 * face + 1](start + i, start + i) = 0;
        base_permutations[12 + 2 * face + 1](start + i, start + face_ref[i])
            = 1;
      }
    }
  }
  else
  {
    std::cout << "Base permutations not implemented for this cell type."
              << std::endl;
  }

  Eigen::MatrixXd coeffs = compute_expansion_coefficients(
      celltype, Eigen::MatrixXd::Identity(ndofs, ndofs),
      Eigen::MatrixXd::Identity(ndofs, ndofs), pt, degree);

  return FiniteElement(element::family::P, celltype, degree, {1}, coeffs,
                       entity_dofs, base_permutations, pt,
                       Eigen::MatrixXd::Identity(ndofs, ndofs),
                       mapping::type::identity);
}
//-----------------------------------------------------------------------------
FiniteElement basix::create_dlagrange(cell::type celltype, int degree)
{
  // Only tabulate for scalar. Vector spaces can easily be built from
  // the scalar space.

  const int ndofs = polyset::dim(celltype, degree);

  std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t i = 0; i < topology.size(); ++i)
    entity_dofs[i].resize(topology[i].size(), 0);
  entity_dofs[topology.size() - 1][0] = ndofs;

  // Eigen::ArrayXXd geometry = cell::geometry(celltype);
  const Eigen::ArrayXXd pt
      = lattice::create(celltype, degree, lattice::type::equispaced, true);

  int perm_count = 0;
  for (std::size_t i = 1; i < topology.size() - 1; ++i)
    perm_count += topology[i].size() * i;

  std::vector<Eigen::MatrixXd> base_permutations(
      perm_count, Eigen::MatrixXd::Identity(ndofs, ndofs));

  Eigen::MatrixXd coeffs = compute_expansion_coefficients(
      celltype, Eigen::MatrixXd::Identity(ndofs, ndofs),
      Eigen::MatrixXd::Identity(ndofs, ndofs), pt, degree);

  return FiniteElement(element::family::DP, celltype, degree, {1}, coeffs,
                       entity_dofs, base_permutations, pt,
                       Eigen::MatrixXd::Identity(ndofs, ndofs),
                       mapping::type::identity);
}
//-----------------------------------------------------------------------------
