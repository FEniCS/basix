// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "raviart-thomas.h"
#include "element-families.h"
#include "lagrange.h"
#include "mappings.h"
#include "moments.h"
#include "polyset.h"
#include "quadrature.h"
#include <Eigen/Dense>
#include <numeric>
#include <vector>

using namespace basix;

//----------------------------------------------------------------------------
FiniteElement basix::create_rt(cell::type celltype, int degree)
{
  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron)
    throw std::runtime_error("Unsupported cell type");

  const int tdim = cell::topological_dimension(celltype);

  const cell::type facettype
      = (tdim == 2) ? cell::type::interval : cell::type::triangle;

  // The number of order (degree-1) scalar polynomials
  const int nv = polyset::dim(celltype, degree - 1);
  // The number of order (degree-2) scalar polynomials
  const int ns0 = polyset::dim(celltype, degree - 2);
  // The number of additional polynomials in the polynomial basis for
  // Raviart-Thomas
  const int ns = polyset::dim(facettype, degree - 1);

  // Evaluate the expansion polynomials at the quadrature points
  auto [Qpts, Qwts]
      = quadrature::make_quadrature("default", celltype, 2 * degree);
  Eigen::ArrayXXd Pkp1_at_Qpts
      = polyset::tabulate(celltype, degree, 0, Qpts)[0];

  // The number of order (degree) polynomials
  const int psize = Pkp1_at_Qpts.cols();

  // Create coefficients for order (degree-1) vector polynomials
  Eigen::MatrixXd wcoeffs = Eigen::MatrixXd::Zero(nv * tdim + ns, psize * tdim);
  for (int j = 0; j < tdim; ++j)
  {
    wcoeffs.block(nv * j, psize * j, nv, nv)
        = Eigen::MatrixXd::Identity(nv, nv);
  }

  // Create coefficients for additional polynomials in Raviart-Thomas
  // polynomial basis
  for (int i = 0; i < ns; ++i)
  {
    for (int k = 0; k < psize; ++k)
    {
      for (int j = 0; j < tdim; ++j)
      {
        const double w_sum = (Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(j)
                              * Pkp1_at_Qpts.col(k))
                                 .sum();
        wcoeffs(nv * tdim + i, k + psize * j) = w_sum;
      }
    }
  }

  // quadrature degree
  int quad_deg = 5 * degree;

  // Add integral moments on facets
  Eigen::ArrayXXd points_facet;
  Eigen::MatrixXd matrix_facet;
  FiniteElement facet_moment_space = create_dlagrange(facettype, degree - 1);
  std::tie(points_facet, matrix_facet) = moments::make_normal_integral_moments(
      facet_moment_space, celltype, tdim, degree, quad_deg);
  std::vector<Eigen::MatrixXd> facet_transforms
      = moments::create_normal_moment_dof_transformations(facet_moment_space);

  const int facet_dofs = facet_transforms[0].rows();

  Eigen::ArrayXXd points_cell(0, tdim);
  Eigen::MatrixXd matrix_cell(0, 0);
  // Add integral moments on interior
  if (degree > 1)
  {
    // Interior integral moment
    std::tie(points_cell, matrix_cell)
        = moments::make_integral_moments(create_dlagrange(celltype, degree - 2),
                                         celltype, tdim, degree, quad_deg);
  }

  // Interpolation points and matrix
  Eigen::ArrayXXd points;
  Eigen::MatrixXd matrix;

  std::tie(points, matrix) = combine_interpolation_data(
      points_facet, points_cell, {}, matrix_facet, matrix_cell, {}, tdim, tdim);

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  const int facet_count = tdim + 1;
  const int ndofs = nv * tdim + ns;
  int transform_count = 0;
  for (int i = 1; i < tdim; ++i)
    transform_count += topology[i].size() * i;

  std::vector<Eigen::MatrixXd> base_transformations(
      transform_count, Eigen::MatrixXd::Identity(ndofs, ndofs));
  if (tdim == 2)
  {
    for (int edge = 0; edge < facet_count; ++edge)
    {
      const int start = facet_dofs * edge;
      base_transformations[edge].block(start, start, facet_dofs, facet_dofs)
          = facet_transforms[0];
    }
  }
  else if (tdim == 3)
  {
    for (int face = 0; face < facet_count; ++face)
    {
      const int start = facet_dofs * face;
      base_transformations[6 + 2 * face].block(start, start, facet_dofs,
                                               facet_dofs)
          = facet_transforms[0];
      base_transformations[6 + 2 * face + 1].block(start, start, facet_dofs,
                                                   facet_dofs)
          = facet_transforms[1];
    }
  }

  // Raviart-Thomas has ns dofs on each facet, and ns0*tdim in the interior
  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (int i = 0; i < tdim - 1; ++i)
    entity_dofs[i].resize(topology[i].size(), 0);
  entity_dofs[tdim - 1].resize(topology[tdim - 1].size(), facet_dofs);
  entity_dofs[tdim] = {ns0 * tdim};

  Eigen::MatrixXd coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, matrix, points, degree);
  return FiniteElement(element::family::RT, celltype, degree, {tdim}, coeffs,
                       entity_dofs, base_transformations, points, matrix,
                       mapping::type::contravariantPiola);
}
//-----------------------------------------------------------------------------
