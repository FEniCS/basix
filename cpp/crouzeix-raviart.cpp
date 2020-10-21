
// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "crouzeix-raviart.h"
#include "polynomial-set.h"
#include "quadrature.h"
#include <Eigen/Dense>
#include <iostream>
#include <numeric>
#include <vector>

using namespace libtab;

CrouzeixRaviart::CrouzeixRaviart(Cell::Type celltype, int k)
    : FiniteElement(celltype, k)
{
  if (k != 1)
    throw std::runtime_error("Only defined for degree =1");

  // Compute facet midpoints
  int tdim = Cell::topological_dimension(celltype);
  auto facet_topology = Cell::topology(celltype)[tdim - 1];
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> geometry
      = Cell::geometry(celltype);

  const int ndofs = facet_topology.size();
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pts(
      ndofs, tdim);
  pts.setZero();

  int c = 0;
  for (auto f : facet_topology)
  {
    for (int i : f)
      pts.row(c) += geometry.row(i);
    ++c;
  }

  pts /= static_cast<double>(tdim);

  // Initial coefficients are Identity Matrix
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(ndofs, ndofs);

  Eigen::MatrixXd dualmat = PolynomialSet::tabulate(celltype, 1, 0, pts)[0];

  apply_dualmat_to_basis(coeffs, dualmat);
}
//-----------------------------------------------------------------------------
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
CrouzeixRaviart::tabulate(
    int nderiv,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts) const
{
  const int tdim = Cell::topological_dimension(_cell_type);
  if (pts.cols() != tdim)
    throw std::runtime_error(
        "Point dimension does not match element dimension");

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dbasis_at_pts = PolynomialSet::tabulate(_cell_type, _degree, nderiv, pts);

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dresult(dbasis_at_pts.size());

  for (std::size_t p = 0; p < dresult.size(); ++p)
    dresult[p] = dbasis_at_pts[p].matrix() * _coeffs.transpose();

  return dresult;
}
