// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"
#include "brezzi-douglas-marini.h"
#include "crouzeix-raviart.h"
#include "lagrange.h"
#include "nce-rtc.h"
#include "nedelec.h"
#include "polyset.h"
#include "raviart-thomas.h"
#include "regge.h"
#include <iostream>
#include <numeric>

#define str_macro(X) #X
#define str(X) str_macro(X)

using namespace basix;

namespace
{
  int compute_value_size(const mapping::type mapping_type, const Eigen::MatrixXd& J)
  {
    switch (mapping_type)
    {
    case mapping::type::identity:
      return 1;
    case mapping::type::covariantPiola:
      return J.rows();
    case mapping::type::contravariantPiola:
      return J.rows();
    case mapping::type::doubleCovariantPiola:
      return J.rows() * J.rows();
    case mapping::type::doubleContravariantPiola:
      return J.rows() * J.rows();
    default:
      throw std::runtime_error("Mapping not yet implemented");
    }
  }
} // namespace

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
    if (cell == cell::type::quadrilateral or cell == cell::type::hexahedron)
      return create_rtc(cell, degree);
    else
      return create_rt(cell, degree);
  }
  else if (family == element::family::N1E)
  {
    if (cell == cell::type::quadrilateral or cell == cell::type::hexahedron)
      return create_nce(cell, degree);
    else
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
    mapping::type mapping_type)
    : _cell_type(cell_type), _family(family), _degree(degree),
      _value_shape(value_shape), _mapping_type(mapping_type), _coeffs(coeffs),
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
element::family FiniteElement::family() const { return _family; }
//-----------------------------------------------------------------------------
const mapping::type FiniteElement::mapping_type() const
{
  return _mapping_type;
}
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
  int ndsize = 1;
  for (int i = 1; i <= nd; ++i)
    ndsize *= (tdim + i);
  for (int i = 1; i <= nd; ++i)
    ndsize /= i;

  const int ndofs = _coeffs.rows();
  const int vs = value_size();

  std::vector<double> basis_data(ndsize * x.rows() * ndofs * vs);
  tabulate_to_memory(nd, x, basis_data.data());

  std::vector<Eigen::ArrayXXd> dresult;
  for (int p = 0; p < ndsize; ++p)
    dresult.push_back(Eigen::Map<Eigen::ArrayXXd>(
        basis_data.data() + p * x.rows() * ndofs * vs, x.rows(), ndofs * vs));

  return dresult;
}
//-----------------------------------------------------------------------------
void FiniteElement::tabulate_to_memory(int nd, const Eigen::ArrayXXd& x,
                                       double* basis_data) const
{
  const int tdim = cell::topological_dimension(_cell_type);
  if (x.cols() != tdim)
    throw std::runtime_error("Point dim does not match element dim.");

  std::vector<Eigen::ArrayXXd> basis
      = polyset::tabulate(_cell_type, _degree, nd, x);
  const int psize = polyset::dim(_cell_type, _degree);
  const int ndofs = _coeffs.rows();
  const int vs = value_size();

  for (std::size_t p = 0; p < basis.size(); ++p)
  {
    // Map block for current derivative
    Eigen::Map<Eigen::ArrayXXd> dresult(basis_data + p * x.rows() * ndofs * vs,
                                        x.rows(), ndofs * vs);
    for (int j = 0; j < vs; ++j)
    {
      dresult.block(0, ndofs * j, x.rows(), ndofs)
          = basis[p].matrix()
            * _coeffs.block(0, psize * j, _coeffs.rows(), psize).transpose();
    }
  }
}
//-----------------------------------------------------------------------------
std::vector<Eigen::MatrixXd> FiniteElement::base_permutations() const
{
  return _base_permutations;
}
//-----------------------------------------------------------------------------
int FiniteElement::num_points() const { return _points.rows(); }
//-----------------------------------------------------------------------------
const Eigen::ArrayXXd& FiniteElement::points() const { return _points; }
//-----------------------------------------------------------------------------
Eigen::ArrayXXd
FiniteElement::map_push_forward(const Eigen::ArrayXXd& reference_data,
                                const Eigen::MatrixXd& J, double detJ,
                                const Eigen::MatrixXd& K) const
{
  const int physical_value_size = compute_value_size(_mapping_type, J);
  Eigen::ArrayXXd physical_data(physical_value_size, reference_data.cols());
  for (int i = 0; i < reference_data.cols(); ++i)
    physical_data.col(i) = mapping::map_push_forward(reference_data.col(i), J,
                                                     detJ, K, _mapping_type);
  return physical_data;
}
//-----------------------------------------------------------------------------
Eigen::ArrayXXd
FiniteElement::map_pull_back(const Eigen::ArrayXXd& physical_data,
                             const Eigen::MatrixXd& J, double detJ,
                             const Eigen::MatrixXd& K) const
{
  Eigen::ArrayXXd reference_data(value_size(), physical_data.cols());
  for (int i = 0; i < physical_data.cols(); ++i)
    reference_data.col(i) = mapping::map_pull_back(physical_data.col(i), J,
                                                   detJ, K, _mapping_type);
  return reference_data;
}
//-----------------------------------------------------------------------------
const std::string& basix::version()
{
  static const std::string version_str = str(BASIX_VERSION);
  return version_str;
}
//-----------------------------------------------------------------------------
