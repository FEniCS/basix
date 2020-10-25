
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

//-----------------------------------------------------------------------------
CrouzeixRaviart::CrouzeixRaviart(cell::Type celltype, int k)
    : FiniteElement(celltype, k)
{
  if (k != 1)
    throw std::runtime_error("Only defined for degree 1");

  this->_value_size = 1;

  // Compute facet midpoints
  const int tdim = cell::topological_dimension(celltype);
  const std::vector<std::vector<int>> facet_topology
      = cell::topology(celltype)[tdim - 1];
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> geometry
      = cell::geometry(celltype);

  const int ndofs = facet_topology.size();
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pts(
      ndofs, tdim);
  pts.setZero();

  int c = 0;
  for (const std::vector<int>& f : facet_topology)
  {
    for (int i : f)
      pts.row(c) += geometry.row(i);
    ++c;
  }

  pts /= static_cast<double>(tdim);

  // Initial coefficients are Identity Matrix
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(ndofs, ndofs);

  Eigen::MatrixXd dualmat = polyset::tabulate(celltype, 1, 0, pts)[0];

  apply_dualmat_to_basis(coeffs, dualmat);
}
//-----------------------------------------------------------------------------
