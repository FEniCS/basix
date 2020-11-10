// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"
#include "polynomial-set.h"
#include <iostream>

using namespace libtab;

//-----------------------------------------------------------------------------
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
FiniteElement::compute_expansion_coefficents(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& coeffs,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& dualmat,
    bool condition_check)
{
#ifndef NDEBUG
  std::cout << "Initial coeffs = \n[" << coeffs << "]\n";
  std::cout << "Dual matrix = \n[" << dualmat << "]\n";
#endif

  auto A = coeffs * dualmat.transpose();

  if (condition_check)
  {
    double detA = A.determinant();
    if (std::fabs(detA) < 1e-6)
      throw std::runtime_error(
          "Poorly conditioned B.D^T when computing expansion coefficients");
  }

  // _coeffs = A^-1(coeffs)
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      new_coeffs = A.colPivHouseholderQr().solve(coeffs);

#ifndef NDEBUG
  std::cout << "New coeffs = \n[" << new_coeffs << "]\n";
#endif
  return new_coeffs;
}
//-----------------------------------------------------------------------------
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
FiniteElement::tabulate(
    int nd,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        x) const
{
  const int tdim = cell::topological_dimension(_cell_type);
  if (x.cols() != tdim)
    throw std::runtime_error("Point dim does not match element dim.");

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      basis = polyset::tabulate(_cell_type, _degree, nd, x);
  const int psize = polyset::size(_cell_type, _degree);
  const int ndofs = _coeffs.rows();

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dresult(
          basis.size(),
          Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(
              x.rows(), ndofs * _value_size));
  for (std::size_t p = 0; p < dresult.size(); ++p)
  {
    for (int j = 0; j < _value_size; ++j)
    {
      dresult[p].block(0, ndofs * j, x.rows(), ndofs)
          = basis[p].matrix()
            * _coeffs.block(0, psize * j, _coeffs.rows(), psize).transpose();
    }
  }

  return dresult;
}
//-----------------------------------------------------------------------------
