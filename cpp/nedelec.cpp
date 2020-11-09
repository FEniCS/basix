// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "nedelec.h"
#include "integral-moments.h"
#include "lagrange.h"
#include "polynomial-set.h"
#include "quadrature.h"
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

  // Number of order (degree) vector polynomials
  const int nv = degree * (degree + 1) / 2;
  // Number of order (degree-1) vector polynomials
  const int ns0 = (degree - 1) * degree / 2;
  // Number of additional polynomials in Nedelec set
  const int ns = degree;

  // Tabulate polynomial set at quadrature points
  auto [Qpts, Qwts] = quadrature::make_quadrature(tdim, 2 * degree);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts
      = polyset::tabulate(cell::Type::triangle, degree, 0, Qpts)[0];

  const int psize = Pkp1_at_Qpts.cols();

  // Create coefficients for order (degree-1) vector polynomials
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      wcoeffs(nv * 2 + ns, psize * 2);
  wcoeffs.setZero();
  wcoeffs.block(0, 0, nv, nv) = Eigen::MatrixXd::Identity(nv, nv);
  wcoeffs.block(nv, psize, nv, nv) = Eigen::MatrixXd::Identity(nv, nv);

  // Create coefficients for the additional Nedelec polynomials
  for (int i = 0; i < ns; ++i)
  {
    for (int k = 0; k < psize; ++k)
    {
      auto w0 = Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(1)
                * Pkp1_at_Qpts.col(k);
      wcoeffs(2 * nv + i, k) = w0.sum();

      auto w1 = -Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(0)
                * Pkp1_at_Qpts.col(k);
      wcoeffs(2 * nv + i, k + psize) = w1.sum();
    }
  }
  return wcoeffs;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_nedelec_2d_dual(int degree)
{
  // Number of dofs and size of polynomial set P(k+1)
  const int ndofs = 3 * degree + degree * (degree - 1);
  const int psize = (degree + 1) * (degree + 2) / 2;

  // Dual space
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat(ndofs, psize * 2);
  dualmat.setZero();

  // dof counter
  int quad_deg = 5 * degree;

  // Integral representation for the boundary (edge) dofs
  FiniteElement moment_space_E
      = DiscontinuousLagrange::create(cell::Type::interval, degree - 1);
  dualmat.block(0, 0, 3 * degree, psize * 2)
      = moments::make_tangent_integral_moments(
          moment_space_E, cell::Type::triangle, 2, degree, quad_deg);

  if (degree > 1)
  {
    // Interior integral moment
    FiniteElement moment_space_I
        = DiscontinuousLagrange::create(cell::Type::triangle, degree - 2);
    dualmat.block(3 * degree, 0, degree * (degree - 1), psize * 2)
        = moments::make_integral_moments(moment_space_I, cell::Type::triangle,
                                         2, degree, quad_deg);
  }
  return dualmat;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_nedelec_3d_space(int degree)
{
  // Reference tetrahedron
  const int tdim = 3;

  // Number of order (degree) vector polynomials
  const int nv = degree * (degree + 1) * (degree + 2) / 6;

  // Number of order (degree-1) vector polynomials
  const int ns0 = (degree - 1) * degree * (degree + 1) / 6;
  // Number of additional Nedelec polynomials that could be added
  const int ns = degree * (degree + 1) / 2;
  // Number of polynomials that would be included that are not independent so
  // are removed
  const int ns_remove = degree * (degree - 1) / 2;

  // Number of dofs in the space, ie size of polynomial set
  const int ndofs = 6 * degree + 4 * degree * (degree - 1)
                    + (degree - 2) * (degree - 1) * degree / 2;

  // Tabulate polynomial basis at quadrature points
  auto [Qpts, Qwts] = quadrature::make_quadrature(tdim, 2 * degree);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts
      = polyset::tabulate(cell::Type::tetrahedron, degree, 0, Qpts)[0];
  const int psize = Pkp1_at_Qpts.cols();

  // Create coefficients for order (degree-1) polynomials
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      wcoeffs(ndofs, psize * tdim);
  wcoeffs.setZero();
  for (int i = 0; i < tdim; ++i)
    wcoeffs.block(nv * i, psize * i, nv, nv)
        = Eigen::MatrixXd::Identity(nv, nv);

  // Create coefficients for additional Nedelec polynomials
  for (int i = 0; i < ns; ++i)
  {
    for (int k = 0; k < psize; ++k)
    {
      auto w = Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(2)
               * Pkp1_at_Qpts.col(k);
      // Don't include polynomials (*, *, 0) that are dependant
      if (i >= ns_remove)
        wcoeffs(tdim * nv + i - ns_remove, psize + k) = -w.sum();
      wcoeffs(tdim * nv + i + ns - ns_remove, k) = w.sum();
    }
  }
  for (int i = 0; i < ns; ++i)
  {
    for (int k = 0; k < psize; ++k)
    {
      auto w = Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(1)
               * Pkp1_at_Qpts.col(k);
      wcoeffs(tdim * nv + i + ns * 2 - ns_remove, k) = -w.sum();
      // Don't include polynomials (*, *, 0) that are dependant
      if (i >= ns_remove)
        wcoeffs(tdim * nv + i - ns_remove, psize * 2 + k) = w.sum();
    }
  }
  for (int i = 0; i < ns; ++i)
  {
    for (int k = 0; k < psize; ++k)
    {
      auto w = Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(0)
               * Pkp1_at_Qpts.col(k);
      wcoeffs(tdim * nv + i + ns - ns_remove, psize * 2 + k) = -w.sum();
      wcoeffs(tdim * nv + i + ns * 2 - ns_remove, psize + k) = w.sum();
    }
  }

  return wcoeffs;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_nedelec_3d_dual(int degree)
{
  const int tdim = 3;

  // Size of polynomial set P(k+1)
  const int psize = (degree + 1) * (degree + 2) * (degree + 3) / 6;

  // Work out number of dofs
  const int ndofs = 6 * degree + 4 * degree * (degree - 1)
                    + (degree - 2) * (degree - 1) * degree / 2;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat(ndofs, psize * tdim);
  dualmat.setZero();

  // Create quadrature scheme on the edge
  const int quad_deg = 5 * degree;

  // Integral representation for the boundary (edge) dofs
  FiniteElement moment_space_E
      = DiscontinuousLagrange::create(cell::Type::interval, degree - 1);
  dualmat.block(0, 0, 6 * degree, psize * 3)
      = moments::make_tangent_integral_moments(
          moment_space_E, cell::Type::tetrahedron, 3, degree, quad_deg);

  if (degree > 1)
  {
    // Integral moments on faces
    FiniteElement moment_space_F
        = DiscontinuousLagrange::create(cell::Type::triangle, degree - 2);
    dualmat.block(6 * degree, 0, 4 * (degree - 1) * degree, psize * 3)
        = moments::make_integral_moments(
            moment_space_F, cell::Type::tetrahedron, 3, degree, quad_deg);
  }

  if (degree > 2)
  {
    // Interior integral moment
    FiniteElement moment_space_I
        = DiscontinuousLagrange::create(cell::Type::tetrahedron, degree - 3);
    dualmat.block(6 * degree + 4 * degree * (degree - 1), 0,
                  (degree - 2) * (degree - 1) * degree / 2, psize * 3)
        = moments::make_integral_moments(
            moment_space_I, cell::Type::tetrahedron, 3, degree, quad_deg);
  }

  return dualmat;
}

} // namespace

//-----------------------------------------------------------------------------
FiniteElement Nedelec::create(cell::Type celltype, int degree)
{
  const int tdim = cell::topological_dimension(celltype);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> wcoeffs;
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dualmat;

  if (celltype == cell::Type::triangle)
  {
    wcoeffs = create_nedelec_2d_space(degree);
    dualmat = create_nedelec_2d_dual(degree);
  }
  else if (celltype == cell::Type::tetrahedron)
  {
    wcoeffs = create_nedelec_3d_space(degree);
    dualmat = create_nedelec_3d_dual(degree);
  }
  else
    throw std::runtime_error("Invalid celltype in Nedelec");

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t i = 0; i < topology.size(); ++i)
    entity_dofs[i].resize(topology[i].size(), 0);

  auto new_coeffs
      = FiniteElement::compute_expansion_coefficents(wcoeffs, dualmat);
  FiniteElement el(celltype, degree, tdim, new_coeffs, entity_dofs);
  return el;
}
//-----------------------------------------------------------------------------
