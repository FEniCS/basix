// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"
#include "polynomial-set.h"
#include <iostream>
#include <map>

using namespace libtab;
//-----------------------------------------------------------------------------
FiniteElement::FiniteElement(cell::Type cell_type, int degree,
                             std::vector<int> value_shape,
                             Eigen::ArrayXXd coeffs,
                             std::vector<std::vector<int>> entity_dofs,
                             std::vector<Eigen::MatrixXd> base_permutations)
    : _cell_type(cell_type), _degree(degree), _value_shape(value_shape),
      _coeffs(coeffs), _entity_dofs(entity_dofs),
      _base_permutations(base_permutations)
{
  // Check that entity dofs add up to total number of dofs
  int sum = 0;
  for (const std::vector<int>& q : entity_dofs)
  {
    for (const int& w : q)
      sum += w;
  }

  if (sum != _coeffs.rows())
  {
    throw std::runtime_error(
        "Number of entity dofs does not match total number of dofs");
  }
}
//-----------------------------------------------------------------------------
Eigen::MatrixXd
FiniteElement::compute_expansion_coefficients(const Eigen::MatrixXd& coeffs,
                                             const Eigen::MatrixXd& dualmat,
                                             bool condition_check)
{
#ifndef NDEBUG
  std::cout << "Initial coeffs = \n[" << coeffs << "]\n";
  std::cout << "Dual matrix = \n[" << dualmat << "]\n";
#endif

  const Eigen::MatrixXd A = coeffs * dualmat.transpose();
  if (condition_check)
  {
    Eigen::JacobiSVD svd(A);
    const int size = svd.singularValues().size();
    const double kappa
        = svd.singularValues()(0) / svd.singularValues()(size - 1);
    if (kappa > 1e6)
    {
      throw std::runtime_error("Poorly conditioned B.D^T when computing "
                               "expansion coefficients");
    }
  }

  Eigen::MatrixXd new_coeffs = A.colPivHouseholderQr().solve(coeffs);
#ifndef NDEBUG
  std::cout << "New coeffs = \n[" << new_coeffs << "]\n";
#endif

  return new_coeffs;
}
//-----------------------------------------------------------------------------
std::vector<Eigen::ArrayXXd>
FiniteElement::tabulate(int nd, const Eigen::ArrayXXd& x) const
{
  const int tdim = cell::topological_dimension(_cell_type);
  if (x.cols() != tdim)
    throw std::runtime_error("Point dim does not match element dim.");

  std::vector<Eigen::ArrayXXd> basis
      = polyset::tabulate(_cell_type, _degree, nd, x);
  const int psize = polyset::size(_cell_type, _degree);
  const int ndofs = _coeffs.rows();
  const int vs = value_size();

  std::vector<Eigen::ArrayXXd> dresult(basis.size(),
                                       Eigen::ArrayXXd(x.rows(), ndofs * vs));
  for (std::size_t p = 0; p < dresult.size(); ++p)
  {
    for (int j = 0; j < vs; ++j)
    {
      dresult[p].block(0, ndofs * j, x.rows(), ndofs)
          = basis[p].matrix()
            * _coeffs.block(0, psize * j, _coeffs.rows(), psize).transpose();
    }
  }

  return dresult;
}
//-----------------------------------------------------------------------------
