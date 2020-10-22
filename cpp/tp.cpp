// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "tp.h"
#include "polynomial-set.h"
#include <Eigen/Dense>

using namespace libtab;

TensorProduct::TensorProduct(Cell::Type celltype, int degree)
    : FiniteElement(celltype, degree)
{
  if (celltype != Cell::Type::quadrilateral and celltype != Cell::Type::prism
      and celltype != Cell::Type::pyramid
      and celltype != Cell::Type::hexahedron)
    throw std::runtime_error("Invalid celltype");

  this->_value_size = 1;

  // Tabulate basis at nodes
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pt
      = Cell::create_lattice(celltype, degree, true);

  Eigen::MatrixXd dualmat = PolynomialSet::tabulate(celltype, degree, 0, pt)[0];

  int ndofs = pt.rows();
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(ndofs, ndofs);
  apply_dualmat_to_basis(coeffs, dualmat);
}
