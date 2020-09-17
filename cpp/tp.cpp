// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "tp.h"
#include "polynomial-set.h"
#include "simplex.h"
#include <Eigen/Dense>

TensorProduct::TensorProduct(CellType celltype, int degree)
    : FiniteElement(0, degree)
{
  // Reference cell vertices
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> cell;

  if (celltype == CellType::quadrilateral)
  {
    _dim = 2;
    cell.resize(4, 2);
    cell << 0, 0, 1, 0, 0, 1, 1, 1;
  }
  else if (celltype == CellType::hexahedron)
  {
    _dim = 3;
    cell.resize(8, 3);
    cell << 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1,
        1;
  }
  else
    throw std::runtime_error("Invalid celltype");

  // Create orthonormal basis on cell
  std::vector<Polynomial> basis
      = PolynomialSet::compute_polynomial_set(celltype, degree);

  // Tabulate basis at nodes
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pt;

  // Create lattice
  if (celltype == CellType::quadrilateral)
  {
    pt.resize((degree + 1) * (degree + 1), 2);

    int c = 0;
    for (int i = 0; i < degree + 1; ++i)
      for (int j = 0; j < degree + 1; ++j)
      {
        pt(c, 0) = i;
        pt(c, 1) = j;
        ++c;
      }
    pt /= degree;
  }
  else
  {
    pt.resize((degree + 1) * (degree + 1) * (degree + 1), 3);

    int c = 0;
    for (int i = 0; i < degree + 1; ++i)
      for (int j = 0; j < degree + 1; ++j)
        for (int k = 0; k < degree + 1; ++k)
        {
          pt(c, 0) = i;
          pt(c, 1) = j;
          pt(c, 2) = k;
          ++c;
        }
    pt /= degree;
  }

  int ndofs = pt.rows();
  assert(ndofs == (int)basis.size());

  Eigen::MatrixXd dualmat(ndofs, ndofs);
  for (int j = 0; j < ndofs; ++j)
    dualmat.col(j) = basis[j].tabulate(pt);

  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(ndofs, ndofs);
  apply_dualmat_to_basis(coeffs, dualmat, basis, 1);
}
