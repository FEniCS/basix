// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "nedelec-second-kind.h"
#include "integral-moments.h"
#include "lagrange.h"
#include "polynomial-set.h"
#include "quadrature.h"
#include "raviart-thomas.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <iostream>
#include <numeric>
#include <vector>

using namespace libtab;

namespace
{
//-----------------------------------------------------------------------------
Eigen::MatrixXd create_nedelec_2d_dual(int degree)
{
  // Number of dofs and size of polynomial set P(k+1)
  const int ndofs = (degree + 1) * (degree + 2);
  const int psize = (degree + 1) * (degree + 2) / 2;

  // Dual space
  Eigen::MatrixXd dual = Eigen::MatrixXd::Zero(ndofs, psize * 2);

  // dof counter
  int quad_deg = 5 * degree;

  // Integral representation for the boundary (edge) dofs
  FiniteElement moment_space_E
      = dlagrange::create(cell::Type::interval, degree);
  dual.block(0, 0, 3 * (degree + 1), psize * 2)
      = moments::make_tangent_integral_moments(
          moment_space_E, cell::Type::triangle, 2, degree, quad_deg);

  if (degree > 1)
  {
    // Interior integral moment
    FiniteElement moment_space_I = rt::create(cell::Type::triangle, degree - 1);
    dual.block(3 * (degree + 1), 0, (degree - 1) * (degree + 1), psize * 2)
        = moments::make_dot_integral_moments(
            moment_space_I, cell::Type::triangle, 2, degree, quad_deg);
  }

  return dual;
}
//-----------------------------------------------------------------------------
Eigen::MatrixXd create_nedelec_3d_dual(int degree)
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
  FiniteElement moment_space_E
      = dlagrange::create(cell::Type::interval, degree);
  dual.block(0, 0, 6 * (degree + 1), psize * 3)
      = moments::make_tangent_integral_moments(
          moment_space_E, cell::Type::tetrahedron, 3, degree, quad_deg);

  if (degree > 1)
  {
    // Integral moments on faces
    FiniteElement moment_space_F = rt::create(cell::Type::triangle, degree - 1);
    dual.block(6 * (degree + 1), 0, 4 * (degree - 1) * (degree + 1), psize * 3)
        = moments::make_dot_integral_moments(
            moment_space_F, cell::Type::tetrahedron, 3, degree, quad_deg);
  }

  if (degree > 2)
  {
    // Interior integral moment
    FiniteElement moment_space_I
        = dlagrange::create(cell::Type::tetrahedron, degree - 2);
    dual.block((6 + 4 * (degree - 1)) * (degree + 1), 0,
               (degree - 1) * (degree - 2) * (degree + 1) / 2, psize * 3)
        = moments::make_integral_moments(
            moment_space_I, cell::Type::tetrahedron, 3, degree, quad_deg);
  }

  return dual;
}

} // namespace

//-----------------------------------------------------------------------------
FiniteElement nedelec2::create(cell::Type celltype, int degree)
{
  const int tdim = cell::topological_dimension(celltype);
  const int psize = polyset::size(celltype, degree);
  Eigen::MatrixXd wcoeffs
      = Eigen::MatrixXd::Identity(tdim * psize, tdim * psize);

  Eigen::MatrixXd dual;
  if (celltype == cell::Type::triangle)
    dual = create_nedelec_2d_dual(degree);
  else if (celltype == cell::Type::tetrahedron)
    dual = create_nedelec_3d_dual(degree);
  else
    throw std::runtime_error("Invalid celltype in Nedelec");

  // TODO: Implement base permutations
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const int ndofs = dual.rows();
  int perm_count = 0;
  for (int i = 1; i < tdim; ++i)
    perm_count += topology[i].size() * i;
  std::vector<Eigen::MatrixXd> base_permutations(
      perm_count, Eigen::MatrixXd::Identity(ndofs, ndofs));

  const Eigen::MatrixXd coeffs
      = FiniteElement::compute_expansion_coefficients(wcoeffs, dual);

  // Nedelec(2nd kind) has (d+1) dofs on each edge, (d+1)(d-1) on each face
  // and (d-2)(d-1)(d+1)/2 on the interior in 3D
  std::vector<std::vector<int>> entity_dofs(topology.size());
  entity_dofs[0].resize(topology[0].size(), 0);
  entity_dofs[1].resize(topology[1].size(), degree + 1);
  entity_dofs[2].resize(topology[2].size(), (degree + 1) * (degree - 1));
  if (tdim > 2)
    entity_dofs[3] = {(degree - 2) * (degree - 1) * (degree + 1) / 2};

  return FiniteElement(nedelec2::family_name, celltype, degree, {tdim}, coeffs,
                       entity_dofs, base_permutations);
}
//-----------------------------------------------------------------------------
