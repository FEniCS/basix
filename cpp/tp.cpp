// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "tp.h"
#include "polynomial-set.h"
#include <Eigen/Dense>

using namespace libtab;

TensorProduct::TensorProduct(cell::Type celltype, int degree)
    : FiniteElement(celltype, degree)
{
  if (celltype != cell::Type::quadrilateral and celltype != cell::Type::prism
      and celltype != cell::Type::pyramid
      and celltype != cell::Type::hexahedron)
    throw std::runtime_error("Invalid celltype");

  this->_value_size = 1;

  // Tabulate basis at nodes
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pt
      = cell::create_lattice(celltype, degree, true);

  Eigen::MatrixXd dualmat = polyset::tabulate(celltype, degree, 0, pt)[0];

  int ndofs = pt.rows();
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(ndofs, ndofs);
  apply_dualmat_to_basis(coeffs, dualmat);
}
