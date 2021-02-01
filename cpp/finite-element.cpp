// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"
#include "brezzi-douglas-marini.h"
#include "crouzeix-raviart.h"
#include "lagrange.h"
#include "nedelec.h"
#include "polyset.h"
#include "raviart-thomas.h"
#include "regge.h"
#include <iostream>
#include <numeric>

#define str_macro(X) #X
#define str(X) str_macro(X)

using namespace basix;

//-----------------------------------------------------------------------------
basix::FiniteElement basix::create_element(std::string family, std::string cell,
                                           int degree)
{
  return basix::create_element(element::str_to_type(family),
                               cell::str_to_type(cell), degree);
}
//-----------------------------------------------------------------------------
basix::FiniteElement basix::create_element(element::family family,
                                           cell::type cell, int degree)
{

  if (family == element::family::P)
    return create_lagrange(cell, degree);
  else if (family == element::family::DP)
    return create_dlagrange(cell, degree);
  else if (family == element::family::BDM)
    return create_bdm(cell, degree);
  else if (family == element::family::RT)
  {
    //  if (cell == cell::type::quadrilateral or cell == cell::type::hexahedron)
    //    return create_rtc(cell, degree);
    //  else
    return create_rt(cell, degree);
  }
  else if (family == element::family::N1E)
  {
    //  if (cell == cell::type::quadrilateral or cell == cell::type::hexahedron)
    //    return create_nce(cell, degree);
    //  else
    return create_nedelec(cell, degree);
  }
  else if (family == element::family::N2E)
    return create_nedelec2(cell, degree);
  else if (family == element::family::Regge)
    return create_regge(cell, degree);
  else if (family == element::family::CR)
    return create_cr(cell, degree);
  else
    throw std::runtime_error("Family not found");
}
//-----------------------------------------------------------------------------
Eigen::MatrixXd
basix::compute_expansion_coefficients(const Eigen::MatrixXd& coeffs,
                                       const Eigen::MatrixXd& dual,
                                       bool condition_check)
{
#ifndef NDEBUG
  std::cout << "Initial coeffs = \n[" << coeffs << "]\n";
  std::cout << "Dual matrix = \n[" << dual << "]\n";
#endif

  const Eigen::MatrixXd A = coeffs * dual.transpose();
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
//-----------------------------------------------------------------------------
FiniteElement::FiniteElement(
    element::family family, cell::type cell_type, int degree,
    const std::vector<int>& value_shape, const Eigen::ArrayXXd& coeffs,
    const std::vector<std::vector<int>>& entity_dofs,
    const std::vector<Eigen::MatrixXd>& base_permutations,
    const Eigen::ArrayXXd& points, const Eigen::MatrixXd interpolation_matrix,
    const std::string mapping_name)
    : _cell_type(cell_type), _family(family), _degree(degree),
      _value_shape(value_shape), _mapping_name(mapping_name), _coeffs(coeffs),
      _entity_dofs(entity_dofs), _base_permutations(base_permutations),
      _points(points), _interpolation_matrix(interpolation_matrix)
{
  // Check that entity dofs add up to total number of dofs
  int sum = 0;
  for (const std::vector<int>& q : entity_dofs)
    sum = std::accumulate(q.begin(), q.end(), sum);

  if (sum != _coeffs.rows())
  {
    throw std::runtime_error(
        "Number of entity dofs does not match total number of dofs");
  }
}
//-----------------------------------------------------------------------------
cell::type FiniteElement::cell_type() const { return _cell_type; }
//-----------------------------------------------------------------------------
int FiniteElement::degree() const { return _degree; }
//-----------------------------------------------------------------------------
int FiniteElement::value_size() const
{
  int value_size = 1;
  for (const int& d : _value_shape)
    value_size *= d;
  return value_size;
}
//-----------------------------------------------------------------------------
const std::vector<int>& FiniteElement::value_shape() const
{
  return _value_shape;
}
//-----------------------------------------------------------------------------
int FiniteElement::dim() const { return _coeffs.rows(); }
//-----------------------------------------------------------------------------
element::family FiniteElement::family() const
{
  return _family;
}
//-----------------------------------------------------------------------------
const std::string& FiniteElement::mapping_name() const { return _mapping_name; }
//-----------------------------------------------------------------------------
const Eigen::MatrixXd& FiniteElement::interpolation_matrix() const
{
  return _interpolation_matrix;
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<int>>& FiniteElement::entity_dofs() const
{
  return _entity_dofs;
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
  const int psize = polyset::dim(_cell_type, _degree);
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
std::vector<Eigen::MatrixXd> FiniteElement::base_permutations() const
{
  return _base_permutations;
}
//-----------------------------------------------------------------------------
const Eigen::ArrayXXd& FiniteElement::points() const { return _points; }
//-----------------------------------------------------------------------------
const std::string& basix::version()
{
  static const std::string version_str = str(BASIX_VERSION);
  return version_str;
}
//-----------------------------------------------------------------------------
