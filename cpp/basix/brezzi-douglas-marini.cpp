// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "brezzi-douglas-marini.h"
#include "element-families.h"
#include "lagrange.h"
#include "mappings.h"
#include "moments.h"
#include "nedelec.h"
#include "polyset.h"
#include "quadrature.h"
#include <Eigen/Dense>
#include <numeric>
#include <vector>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

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
  const std::size_t ndofs = npoly * tdim;

  // Create coefficients for order (degree-1) vector polynomials
  Eigen::MatrixXd wcoeffs = Eigen::MatrixXd::Identity(ndofs, ndofs);

  // quadrature degree
  int quad_deg = 5 * degree;

  // Add integral moments on facets
  const std::size_t facet_count = tdim + 1;
  const std::size_t facet_dofs = polyset::dim(facettype, degree);
  const int internal_dofs = ndofs - facet_count * facet_dofs;

  Eigen::ArrayXXd points_facet;
  Eigen::MatrixXd matrix_facet;
  FiniteElement facet_moment_space = create_dlagrange(facettype, degree);
  std::tie(points_facet, matrix_facet) = moments::make_normal_integral_moments(
      facet_moment_space, celltype, tdim, quad_deg);

  xt::xtensor<double, 3> facet_transforms
      = moments::create_normal_moment_dof_transformations(facet_moment_space);

  Eigen::ArrayXXd points_cell;
  Eigen::MatrixXd matrix_cell;
  // Add integral moments on interior
  if (degree > 1)
  {
    // Interior integral moment
    std::tie(points_cell, matrix_cell) = moments::make_dot_integral_moments(
        create_nedelec(celltype, degree - 1), celltype, tdim, quad_deg);
  }

  // Interpolation points and matrix
  Eigen::ArrayXXd points;
  Eigen::MatrixXd matrix;

  std::tie(points, matrix) = combine_interpolation_data(
      points_facet, points_cell, {}, matrix_facet, matrix_cell, {}, tdim, tdim);

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::size_t transform_count = 0;
  for (int i = 1; i < tdim; ++i)
    transform_count += topology[i].size() * i;

  xt::xtensor<double, 3> base_transformations
      = xt::zeros<double>({transform_count, ndofs, ndofs});
  for (std::size_t i = 0; i < base_transformations.shape()[0]; ++i)
  {
    xt::view(base_transformations, i, xt::all(), xt::all())
        = xt::eye<double>(ndofs);
  }

  if (tdim == 2)
  {
    for (std::size_t edge = 0; edge < facet_count; ++edge)
    {
      const std::size_t start = facet_dofs * edge;
      auto range = xt::range(start, start + facet_dofs);
      xt::view(base_transformations, edge, range, range)
          = xt::view(facet_transforms, 0, xt::all(), xt::all());
      // const int start = facet_dofs * edge;
      // base_transformations[edge].block(start, start, facet_dofs, facet_dofs)
      //     = facet_transforms[0];
    }
  }
  else if (tdim == 3)
  {
    for (std::size_t face = 0; face < facet_count; ++face)
    {
      const std::size_t start = facet_dofs * face;
      auto range = xt::range(start, start + facet_dofs);
      xt::view(base_transformations, 6 + 2 * face, range, range)
          = xt::view(facet_transforms, 0, xt::all(), xt::all());
      xt::view(base_transformations, 6 + 2 * face + 1, range, range)
          = xt::view(facet_transforms, 1, xt::all(), xt::all());
      // const int start = facet_dofs * face;
      // base_transformations[6 + 2 * face].block(start, start, facet_dofs,
      //                                          facet_dofs)
      //     = facet_transforms[0];
      // base_transformations[6 + 2 * face + 1].block(start, start, facet_dofs,
      //                                              facet_dofs)
      //     = facet_transforms[1];
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
                       entity_dofs, base_transformations, points, matrix,
                       mapping::type::contravariantPiola);
}
//-----------------------------------------------------------------------------
