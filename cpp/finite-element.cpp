// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"
#include "polynomial-set.h"
#include <iostream>

using namespace libtab;

//-----------------------------------------------------------------------------
FiniteElement::FiniteElement(cell::Type cell_type, int degree)
    : _cell_type(cell_type), _degree(degree)
{
  // Do nothing in base class
}
//-----------------------------------------------------------------------------
void FiniteElement::apply_dualmat_to_basis(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& coeffs,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& dualmat)
{
#ifndef NDEBUG
  std::cout << "Initial coeffs = \n[" << coeffs << "]\n";
  std::cout << "Dual matrix = \n[" << dualmat << "]\n";
#endif

  auto A = coeffs * dualmat.transpose();

  // _coeffs = A^-1(coeffs)
  _coeffs = A.colPivHouseholderQr().solve(coeffs);

#ifndef NDEBUG
  std::cout << "New coeffs = \n[" << _coeffs << "]\n";
#endif
}
//-----------------------------------------------------------------------------
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
FiniteElement::tabulate(
    int nderiv,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts) const
{
  const int tdim = cell::topological_dimension(_cell_type);
  if (pts.cols() != tdim)
    throw std::runtime_error(
        "Point dimension does not match element dimension");

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      basis_at_pts = polyset::tabulate(_cell_type, _degree, nderiv, pts);
  const int psize = polyset::size(_cell_type, _degree);
  const int ndofs = _coeffs.rows();

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dresult(basis_at_pts.size());

  for (std::size_t p = 0; p < dresult.size(); ++p)
  {
    dresult[p].resize(pts.rows(), ndofs * _value_size);
    for (int j = 0; j < _value_size; ++j)
      dresult[p].block(0, ndofs * j, pts.rows(), ndofs)
          = basis_at_pts[p].matrix()
            * _coeffs.block(0, psize * j, _coeffs.rows(), psize).transpose();
  }

  return dresult;
}
//-----------------------------------------------------------------------------
