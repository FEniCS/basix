// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "brezzi-douglas-marini.h"
#include "core/dof-permutations.h"
#include "core/element-families.h"
#include "core/mappings.h"
#include "core/moments.h"
#include "core/polyset.h"
#include "core/quadrature.h"
#include "lagrange.h"
#include "nedelec.h"
#include <Eigen/Dense>
#include <numeric>
#include <vector>

using namespace basix;

//----------------------------------------------------------------------------
FiniteElement basix::create_bdm(cell::type celltype, int degree)
{
  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron)
    throw std::runtime_error("Unsupported cell type");

  const int tdim = cell::topological_dimension(celltype);

  const cell::type facettype
      = (tdim == 2) ? cell::type::interval : cell::type::triangle;

  // The number of order (degree) scalar polynomials
  const int npoly = polyset::dim(celltype, degree);
  const int ndofs = npoly * tdim;

  // Create coefficients for order (degree-1) vector polynomials
  Eigen::MatrixXd wcoeffs = Eigen::MatrixXd::Identity(ndofs, ndofs);

  // quadrature degree
  int quad_deg = 5 * degree;

  // Add integral moments on facets
  const int facet_count = tdim + 1;
  const int facet_dofs = polyset::dim(facettype, degree);
  const int internal_dofs = ndofs - facet_count * facet_dofs;

  Eigen::ArrayXXd points_facet;
  Eigen::MatrixXd matrix_facet;
  std::tie(points_facet, matrix_facet) = moments::make_normal_integral_moments(
      create_dlagrange(facettype, degree), celltype, tdim, degree, quad_deg);

  Eigen::ArrayXXd points_cell;
  Eigen::MatrixXd matrix_cell;
  // Add integral moments on interior
  if (degree > 1)
  {
    // Interior integral moment
    std::tie(points_cell, matrix_cell) = moments::make_dot_integral_moments(
        create_nedelec(celltype, degree - 1), celltype, tdim, degree, quad_deg);
  }

  // Interpolation points and matrix
  Eigen::ArrayXXd points;
  Eigen::MatrixXd matrix;

  std::tie(points, matrix) = combine_interpolation_data(
      points_facet, points_cell, {}, matrix_facet, matrix_cell, {}, tdim, tdim);

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  int perm_count = 0;
  for (int i = 1; i < tdim; ++i)
    perm_count += topology[i].size() * i;

  std::vector<Eigen::MatrixXd> base_permutations(
      perm_count, Eigen::MatrixXd::Identity(ndofs, ndofs));
  if (tdim == 2)
  {
    const std::vector<int> edge_ref = dofperms::interval_reflection(degree + 1);
    ndarray<double, 2> edge_dir
        = dofperms::interval_reflection_tangent_directions(degree + 1);
    Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        _edge_dir(edge_dir.data(), edge_dir.shape[0], edge_dir.shape[1]);
    for (int edge = 0; edge < facet_count; ++edge)
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
  }
  else if (tdim == 3)
  {
    const std::vector<int> face_ref = dofperms::triangle_reflection(degree + 1);
    const std::vector<int> face_rot = dofperms::triangle_rotation(degree + 1);
    for (int face = 0; face < facet_count; ++face)
    {
      const int start = face_ref.size() * face;
      for (std::size_t i = 0; i < face_rot.size(); ++i)
      {
        base_permutations[6 + 2 * face](start + i, start + i) = 0;
        base_permutations[6 + 2 * face](start + i, start + face_rot[i]) = 1;
        base_permutations[6 + 2 * face + 1](start + i, start + i) = 0;
        base_permutations[6 + 2 * face + 1](start + i, start + face_ref[i])
            = -1;
      }
    }
  }

  // BDM has facet_dofs dofs on each facet, and
  // ndofs-facet_count*facet_dofs in the interior
  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (int i = 0; i < tdim - 1; ++i)
    entity_dofs[i].resize(topology[i].size(), 0);
  entity_dofs[tdim - 1].resize(topology[tdim - 1].size(), facet_dofs);
  entity_dofs[tdim] = {internal_dofs};

  Eigen::MatrixXd coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, matrix, points, degree);

  return FiniteElement(element::family::BDM, celltype, degree, {tdim}, coeffs,
                       entity_dofs, base_permutations, points, matrix,
                       mapping::type::contravariantPiola);
}
//-----------------------------------------------------------------------------
