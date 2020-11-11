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
#include <numeric>
#include <vector>

using namespace libtab;

namespace
{
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_nedelec_2d_dual(int degree)
{
  // Number of dofs and size of polynomial set P(k+1)
  const int ndofs = (degree + 1) * (degree + 2);
  const int psize = (degree + 1) * (degree + 2) / 2;

  // Dual space
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat(ndofs, psize * 2);
  dualmat.setZero();

  // dof counter
  int quad_deg = 5 * degree;

  // Integral representation for the boundary (edge) dofs
  FiniteElement moment_space_E
      = DiscontinuousLagrange::create(cell::Type::interval, degree);
  dualmat.block(0, 0, 3 * (degree + 1), psize * 2)
      = moments::make_tangent_integral_moments(
          moment_space_E, cell::Type::triangle, 2, degree, quad_deg);

  if (degree > 1)
  {
    // Interior integral moment
    FiniteElement moment_space_I
        = RaviartThomas::create(cell::Type::triangle, degree - 1);
    dualmat.block(3 * (degree + 1), 0, (degree - 1) * (degree + 1), psize * 2)
        = moments::make_dot_integral_moments(
            moment_space_I, cell::Type::triangle, 2, degree, quad_deg);
  }
  return dualmat;
}
//-----------------------------------------------------------------------------
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_nedelec_3d_dual(int degree)
{
  const int tdim = 3;

  // Size of polynomial set P(k+1)
  const int psize = (degree + 1) * (degree + 2) * (degree + 3) / 6;

  // Work out number of dofs
  int ndofs = (degree + 1) * (degree + 2) * (degree + 3) / 2;

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat(ndofs, psize * tdim);
  dualmat.setZero();

  // Create quadrature scheme on the edge
  int quad_deg = 5 * degree;

  // Integral representation for the boundary (edge) dofs
  FiniteElement moment_space_E
      = DiscontinuousLagrange::create(cell::Type::interval, degree);
  dualmat.block(0, 0, 6 * (degree + 1), psize * 3)
      = moments::make_tangent_integral_moments(
          moment_space_E, cell::Type::tetrahedron, 3, degree, quad_deg);

  if (degree > 1)
  {
    // Integral moments on faces
    FiniteElement moment_space_F
        = RaviartThomas::create(cell::Type::triangle, degree - 1);
    dualmat.block(6 * (degree + 1), 0, 4 * (degree - 1) * (degree + 1),
                  psize * 3)
        = moments::make_dot_integral_moments(
            moment_space_F, cell::Type::tetrahedron, 3, degree, quad_deg);
  }

  if (degree > 2)
  {
    // Interior integral moment
    FiniteElement moment_space_I
        = DiscontinuousLagrange::create(cell::Type::tetrahedron, degree - 2);
    dualmat.block((6 + 4 * (degree - 1)) * (degree + 1), 0,
                  (degree + 1) * (degree * degree - 5 * degree - 2) / 2,
                  psize * 3)
        = moments::make_integral_moments(
            moment_space_I, cell::Type::tetrahedron, 3, degree, quad_deg);
  }

  return dualmat;
}

} // namespace

//-----------------------------------------------------------------------------
FiniteElement NedelecSecondKind::create(cell::Type celltype, int degree)
{
  const int tdim = cell::topological_dimension(celltype);
  const int psize = polyset::size(celltype, degree);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> wcoeffs
      = Eigen::MatrixXd::Identity(tdim * psize, tdim * psize);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat;

  if (celltype == cell::Type::triangle)
    dualmat = create_nedelec_2d_dual(degree);
  else if (celltype == cell::Type::tetrahedron)
    dualmat = create_nedelec_3d_dual(degree);
  else
    throw std::runtime_error("Invalid celltype in Nedelec");

  auto new_coeffs
      = FiniteElement::compute_expansion_coefficents(wcoeffs, dualmat);

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t i = 0; i < topology.size(); ++i)
    entity_dofs[i].resize(topology[i].size(), 0);

  FiniteElement el(celltype, degree, {tdim}, new_coeffs, entity_dofs);
  return el;
}
//-----------------------------------------------------------------------------
