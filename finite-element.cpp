// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"

//-----------------------------------------------------------------------------
FiniteElement::FiniteElement(int dim, int degree) : _dim(dim), _degree(degree)
{
  // Do nothing in base class
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
FiniteElement::tabulate_basis(
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts) const
{
  if (pts.cols() != _dim)
    throw std::runtime_error(
        "Point dimension does not match element dimension");

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
      pts.rows(), poly_set.size());
  for (std::size_t j = 0; j < poly_set.size(); ++j)
    result.col(j) = poly_set[j].tabulate(pts);

  return result;
}
//-----------------------------------------------------------------------------
void FiniteElement::apply_dualmat_to_basis(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& coeffs,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& dualmat,
    const std::vector<Polynomial>& basis, int ndim)
{
  auto A = coeffs * dualmat.transpose();
  auto Ainv = A.inverse();
  auto new_coeffs = Ainv * coeffs;
  std::cout << "new_coeffs = \n[" << new_coeffs << "]\n";

  int psize = basis.size();
  int ndofs = dualmat.rows();

  // Create polynomial sets for x and y components
  // stacking x0, x1, x2,... y0, y1, y2,...
  poly_set.resize(ndofs * ndim, Polynomial::zero(2));
  for (int j = 0; j < ndim; ++j)
    for (int i = 0; i < ndofs; ++i)
      for (int k = 0; k < psize; ++k)
        poly_set[i + ndofs * j] += basis[k] * new_coeffs(i, k + psize * j);
}
