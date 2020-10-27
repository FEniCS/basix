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

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_nedelec_2d_space(int degree)
{
  // 2D space on triangle
  const int tdim = 2;
  const int nv = (degree + 2) * (degree + 3) / 2;

  // Tabulate P(k+1) at quadrature points
  auto [Qpts, Qwts] = quadrature::make_quadrature(tdim, 2 * degree + 2);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts
      = polyset::tabulate(cell::Type::triangle, degree + 1, 0, Qpts)[0];

  const int psize = Pkp1_at_Qpts.cols();

  // Create initial coefficients of Pkp1.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      wcoeffs(nv * 2, psize * 2);
  wcoeffs.setZero();
  wcoeffs.block(0, 0, nv, nv) = Eigen::MatrixXd::Identity(nv, nv);
  wcoeffs.block(nv, psize, nv, nv) = Eigen::MatrixXd::Identity(nv, nv);

  return wcoeffs;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_nedelec_2d_dual(int degree)
{
  // Number of dofs and size of polynomial set P(k+1)
  const int ndofs = (degree + 2) * (degree + 3);
  const int psize = (degree + 2) * (degree + 3) / 2;

  // Dual space
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat(ndofs, psize * 2);
  dualmat.setZero();

  // dof counter
  int quad_deg = 5 * (degree + 1);

  // Integral representation for the boundary (edge) dofs
  DiscontinuousLagrange moment_space_E(cell::Type::interval, degree + 1);
  dualmat.block(0, 0, 3 * (degree + 2), psize * 2)
      = moments::make_tangent_integral_moments(
          moment_space_E, cell::Type::triangle, 2, degree + 1, quad_deg);

  if (degree > 0)
  {
    // Interior integral moment
    RaviartThomas moment_space_I(cell::Type::triangle, degree);
    dualmat.block(3 * (degree + 2), 0, degree * (degree + 2), psize * 2)
        = moments::make_dot_integral_moments(
            moment_space_I, cell::Type::triangle, 2, degree + 1, quad_deg);
  }
  return dualmat;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_nedelec_3d_space(int degree)
{
  // Reference tetrahedron
  const int tdim = 3;

  // Vector subset
  const int nv = (degree + 2) * (degree + 3) * (degree + 4) / 6;

  // Tabulate P(k+1) at quadrature points
  auto [Qpts, Qwts] = quadrature::make_quadrature(tdim, 2 * degree + 2);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts
      = polyset::tabulate(cell::Type::tetrahedron, degree + 1, 0, Qpts)[0];
  const int psize = Pkp1_at_Qpts.cols();

  // Create initial coefficients of Pkp1.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      wcoeffs(nv * tdim, psize * tdim);
  wcoeffs.setZero();
  for (int i = 0; i < tdim; ++i)
    wcoeffs.block(nv * i, psize * i, nv, nv)
        = Eigen::MatrixXd::Identity(nv, nv);

  return wcoeffs;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_nedelec_3d_dual(int degree)
{
  const int tdim = 3;

  // Size of polynomial set P(k+1)
  const int psize = (degree + 2) * (degree + 3) * (degree + 4) / 6;

  // Work out number of dofs
  int ndofs = (degree + 2) * (degree + 3) * (degree + 4) / 2;

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat(ndofs, psize * tdim);
  dualmat.setZero();

  // Create quadrature scheme on the edge
  int quad_deg = 5 * (degree + 1);

  // Integral representation for the boundary (edge) dofs
  DiscontinuousLagrange moment_space_E(cell::Type::interval, degree + 1);
  dualmat.block(0, 0, 6 * (degree + 2), psize * 3)
      = moments::make_tangent_integral_moments(
          moment_space_E, cell::Type::tetrahedron, 3, degree + 1, quad_deg);

  if (degree > 0)
  {
    // Integral moments on faces
    RaviartThomas moment_space_F(cell::Type::triangle, degree);
    dualmat.block(6 * (degree + 2), 0, 4 * degree * (degree + 2), psize * 3)
        = moments::make_dot_integral_moments(
            moment_space_F, cell::Type::tetrahedron, 3, degree + 1, quad_deg);
  }

  if (degree > 1)
  {
    // Interior integral moment
    DiscontinuousLagrange moment_space_I(cell::Type::tetrahedron, degree - 1);
    dualmat.block((6 + 4 * degree) * (degree + 2), 0,
                  (degree + 2) * (degree * degree - 3 * degree - 6) / 2, psize * 3)
        = moments::make_integral_moments(
            moment_space_I, cell::Type::tetrahedron, 3, degree + 1, quad_deg);
  }

  return dualmat;
}

} // namespace

//-----------------------------------------------------------------------------
NedelecSecondKind::NedelecSecondKind(cell::Type celltype, int k)
    : FiniteElement(celltype, k)
{
  const int tdim = cell::topological_dimension(celltype);
  this->_value_size = tdim;

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> wcoeffs;
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dualmat;

  if (celltype == cell::Type::triangle)
  {
    wcoeffs = create_nedelec_2d_space(k - 1);
    dualmat = create_nedelec_2d_dual(k - 1);
  }
  else if (celltype == cell::Type::tetrahedron)
  {
    wcoeffs = create_nedelec_3d_space(k - 1);
    dualmat = create_nedelec_3d_dual(k - 1);
  }
  else
    throw std::runtime_error("Invalid celltype in Nedelec");

  apply_dualmat_to_basis(wcoeffs, dualmat);
}
//-----------------------------------------------------------------------------
