// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "bubble.h"
#include "core/dof-transformations.h"
#include "core/element-families.h"
#include "core/lattice.h"
#include "core/mappings.h"
#include "core/polyset.h"
#include "core/quadrature.h"
#include <Eigen/Dense>
#include <numeric>

using namespace basix;

//----------------------------------------------------------------------------
FiniteElement basix::create_bubble(cell::type celltype, int degree)
{
  if (celltype == cell::type::point)
    throw std::runtime_error("Invalid celltype");
  if (celltype == cell::type::interval && degree < 2)
    throw std::runtime_error(
        "Bubble element on an interval must have degree at least 2");
  if (celltype == cell::type::triangle && degree < 3)
    throw std::runtime_error(
        "Bubble element on a triangle must have degree at least 3");
  if (celltype == cell::type::tetrahedron && degree < 4)
    throw std::runtime_error(
        "Bubble element on a tetrahedron must have degree at least 4");
  if (celltype == cell::type::quadrilateral && degree < 2)
    throw std::runtime_error("Bubble element on a quadrilateral interval must "
                             "have degree at least 2");
  if (celltype == cell::type::hexahedron && degree < 2)
    throw std::runtime_error(
        "Bubble element on a hexahedron must have degree at least 2");

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  std::vector<std::vector<int>> entity_dofs(topology.size());

  const int tdim = cell::topological_dimension(celltype);

  // Evaluate the expansion polynomials at the quadrature points
  auto [Qpts, Qwts]
      = quadrature::make_quadrature("default", celltype, 2 * degree);
  Eigen::ArrayXXd polyset_at_Qpts
      = polyset::tabulate(celltype, degree, 0, Qpts)[0];

  // The number of order (degree) polynomials
  const int psize = polyset_at_Qpts.cols();

  // Create points at nodes on interior
  const Eigen::ArrayXXd points
      = lattice::create(celltype, degree, lattice::type::equispaced, false);

  const int ndofs = points.rows();

  // Create coefficients for order (degree-1) vector polynomials
  Eigen::ArrayXXd lower_polyset_at_Qpts;
  Eigen::ArrayXd bubble;
  if (celltype == cell::type::interval)
  {
    lower_polyset_at_Qpts = polyset::tabulate(celltype, degree - 2, 0, Qpts)[0];
    bubble = Qpts.col(0) * (1 - Qpts.col(0));
  }
  else if (celltype == cell::type::triangle)
  {
    lower_polyset_at_Qpts = polyset::tabulate(celltype, degree - 3, 0, Qpts)[0];
    bubble = Qpts.col(0) * Qpts.col(1) * (1 - Qpts.col(0) - Qpts.col(1));
  }
  else if (celltype == cell::type::tetrahedron)
  {
    lower_polyset_at_Qpts = polyset::tabulate(celltype, degree - 4, 0, Qpts)[0];
    bubble = Qpts.col(0) * Qpts.col(1) * Qpts.col(2)
             * (1 - Qpts.col(0) - Qpts.col(1) - Qpts.col(2));
  }
  else if (celltype == cell::type::quadrilateral)
  {
    lower_polyset_at_Qpts = polyset::tabulate(celltype, degree - 2, 0, Qpts)[0];
    bubble = Qpts.col(0) * (1 - Qpts.col(0)) * Qpts.col(1) * (1 - Qpts.col(1));
  }
  else if (celltype == cell::type::hexahedron)
  {
    lower_polyset_at_Qpts = polyset::tabulate(celltype, degree - 2, 0, Qpts)[0];
    bubble = Qpts.col(0) * (1 - Qpts.col(0)) * Qpts.col(1) * (1 - Qpts.col(1))
             * Qpts.col(2) * (1 - Qpts.col(2));
  }

  Eigen::MatrixXd wcoeffs = Eigen::MatrixXd::Zero(ndofs, psize);
  for (int i = 0; i < lower_polyset_at_Qpts.cols(); ++i)
  {
    Eigen::ArrayXd integrand = lower_polyset_at_Qpts.col(i) * bubble;
    for (int k = 0; k < psize; ++k)
    {
      const double w_sum = (Qwts * integrand * polyset_at_Qpts.col(k)).sum();
      wcoeffs(i, k) = w_sum;
    }
  }

  for (int i = 0; i < tdim; ++i)
    for (std::size_t j = 0; j < topology[i].size(); ++j)
      entity_dofs[i].push_back(0);
  entity_dofs[tdim].push_back(points.rows());

  int transform_count = 0;
  for (std::size_t i = 1; i < topology.size() - 1; ++i)
    transform_count += topology[i].size() * i;

  std::vector<Eigen::MatrixXd> base_transformations(
      transform_count, Eigen::MatrixXd::Identity(ndofs, ndofs));

  Eigen::MatrixXd coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, Eigen::MatrixXd::Identity(ndofs, ndofs), points,
      degree);

  return FiniteElement(element::family::Bubble, celltype, degree, {1}, coeffs,
                       entity_dofs, base_transformations, points,
                       Eigen::MatrixXd::Identity(ndofs, ndofs),
                       mapping::type::identity);
}
//-----------------------------------------------------------------------------
