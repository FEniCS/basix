// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"
#include "elements/brezzi-douglas-marini.h"
#include "elements/bubble.h"
#include "elements/crouzeix-raviart.h"
#include "elements/lagrange.h"
#include "elements/nce-rtc.h"
#include "elements/nedelec.h"
#include "elements/raviart-thomas.h"
#include "elements/regge.h"
#include "polyset.h"
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
  switch (family)
  {
  case element::family::P:
    return create_lagrange(cell, degree);
  case element::family::DP:
    return create_dlagrange(cell, degree);
  case element::family::BDM:
    return create_bdm(cell, degree);
  case element::family::RT:
  {
    switch (cell)
    {
    case cell::type::quadrilateral:
      return create_rtc(cell, degree);
    case cell::type::hexahedron:
      return create_rtc(cell, degree);
    default:
      return create_rt(cell, degree);
    }
  }
  case element::family::N1E:
  {
    switch (cell)
    {
    case cell::type::quadrilateral:
      return create_nce(cell, degree);
    case cell::type::hexahedron:
      return create_nce(cell, degree);
    default:
      return create_nedelec(cell, degree);
      return create_rt(cell, degree);
    }
  }
  case element::family::N2E:
    return create_nedelec2(cell, degree);
  case element::family::Regge:
    return create_regge(cell, degree);
  case element::family::CR:
    return create_cr(cell, degree);
  case element::family::Bubble:
    return create_bubble(cell, degree);
  default:
    throw std::runtime_error("Family not found");
  }
}
//-----------------------------------------------------------------------------
Eigen::MatrixXd basix::compute_expansion_coefficients(
    cell::type celltype, const Eigen::MatrixXd& B, const Eigen::MatrixXd& M,
    const Eigen::ArrayXXd& x, int degree, double kappa_tol)
{
  const Eigen::MatrixXd P = polyset::tabulate(celltype, degree, 0, x)[0];

  const int coeff_size = P.cols();
  const int value_size = B.cols() / coeff_size;
  const int m_size = M.cols() / value_size;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(B.rows(), M.rows());
  for (int row = 0; row < B.rows(); ++row)
  {
    for (int i = 0; i < value_size; ++i)
    {
      A.row(row) += B.block(row, coeff_size * i, 1, coeff_size) * P.transpose()
                    * M.block(0, i * m_size, M.rows(), m_size).transpose();
    }
  }

  if (kappa_tol >= 1.0)
  {
    Eigen::JacobiSVD svd(A);
    const int size = svd.singularValues().size();
    const double kappa
        = svd.singularValues()(0) / svd.singularValues()(size - 1);
    if (kappa > kappa_tol)
    {
      throw std::runtime_error("Condition number of B.D^T when computing "
                               "expansion coefficients exceeds tolerance.");
    }
  }

  return A.colPivHouseholderQr().solve(B);
}
//-----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd> basix::combine_interpolation_data(
    const Eigen::ArrayXXd& points_1d, const Eigen::ArrayXXd& points_2d,
    const Eigen::ArrayXXd& points_3d, const Eigen::MatrixXd& matrix_1d,
    const Eigen::MatrixXd& matrix_2d, const Eigen::MatrixXd& matrix_3d,
    const int tdim, const int value_size)
{
  Eigen::ArrayXXd points(points_1d.rows() + points_2d.rows() + points_3d.rows(),
                         tdim);

  points.block(0, 0, points_1d.rows(), tdim) = points_1d;
  points.block(points_1d.rows(), 0, points_2d.rows(), tdim) = points_2d;
  points.block(points_1d.rows() + points_2d.rows(), 0, points_3d.rows(), tdim)
      = points_3d;

  Eigen::MatrixXd matrix(matrix_1d.rows() + matrix_2d.rows() + matrix_3d.rows(),
                         matrix_1d.cols() + matrix_2d.cols()
                             + matrix_3d.cols());
  matrix.setZero();

  const int r1d = matrix_1d.rows();
  const int r2d = matrix_2d.rows();
  const int r3d = matrix_3d.rows();
  const int c1d = matrix_1d.cols() / value_size;
  const int c2d = matrix_2d.cols() / value_size;
  const int c3d = matrix_3d.cols() / value_size;
  for (int i = 0; i < value_size; ++i)
  {
    matrix.block(0, i * (c1d + c2d + c3d), r1d, c1d)
        = matrix_1d.block(0, i * c1d, r1d, c1d);
    matrix.block(r1d, i * (c1d + c2d + c3d) + c1d, r2d, c2d)
        = matrix_2d.block(0, i * c2d, r2d, c2d);
    matrix.block(r1d + r2d, i * (c1d + c2d + c3d) + c1d + c2d, r3d, c3d)
        = matrix_3d.block(0, i * c3d, r3d, c3d);
  }
  return std::make_pair(points, matrix);
}
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
  _map_push_forward = mapping::get_forward_map(mapping_type);
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
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
FiniteElement::map_push_forward(
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        reference_data,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        J,
    const Eigen::ArrayXd& detJ,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        K) const
{
  const int reference_dim = cell::topological_dimension(_cell_type);
  const int physical_dim = J.cols() / reference_dim;
  const int physical_value_size
      = compute_value_size(_mapping_type, physical_dim);
  const int reference_value_size = value_size();
  const int nresults = reference_data.cols() / reference_value_size;
  const int npoints = reference_data.rows();

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      physical_data(npoints, physical_value_size * nresults);
  for (int pt = 0; pt < npoints; ++pt)
  {
    Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        reference_block(reference_data.row(pt).data(), reference_value_size,
                        nresults);
    Eigen::Map<
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        physical_block(physical_data.row(pt).data(), physical_value_size,
                       nresults);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        current_J(J.row(pt).data(), physical_dim, reference_dim);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        current_K(K.row(pt).data(), reference_dim, physical_dim);
    for (int i = 0; i < reference_block.cols(); ++i)
      physical_block.col(i) = _map_push_forward(reference_block.col(i),
                                                current_J, detJ[pt], current_K);
  }
  return physical_data;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
FiniteElement::map_pull_back(
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        physical_data,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        J,
    const Eigen::ArrayXd& detJ,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        K) const
{
  const int reference_dim = cell::topological_dimension(_cell_type);
  const int physical_dim = J.cols() / reference_dim;
  const int physical_value_size
      = compute_value_size(_mapping_type, physical_dim);
  const int reference_value_size = value_size();
  const int nresults = physical_data.cols() / physical_value_size;
  const int npoints = physical_data.rows();

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      reference_data(npoints, reference_value_size * nresults);
  for (int pt = 0; pt < npoints; ++pt)
  {
    Eigen::Map<
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        reference_block(reference_data.row(pt).data(), reference_value_size,
                        nresults);
    Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        physical_block(physical_data.row(pt).data(), physical_value_size,
                       nresults);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        current_J(J.row(pt).data(), physical_dim, reference_dim);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        current_K(K.row(pt).data(), reference_dim, physical_dim);
    for (int i = 0; i < physical_block.cols(); ++i)
      reference_block.col(i) = _map_push_forward(
          physical_block.col(i), current_K, 1 / detJ[pt], current_J);
  }
  return reference_data;
}
//-----------------------------------------------------------------------------
std::string basix::version()
{
  static const std::string version_str = str(BASIX_VERSION);
  return version_str;
}
//-----------------------------------------------------------------------------
int FiniteElement::compute_value_size(mapping::type mapping_type, int dim)
{
  switch (mapping_type)
  {
  case mapping::type::identity:
    return 1;
  case mapping::type::covariantPiola:
    return dim;
  case mapping::type::contravariantPiola:
    return dim;
  case mapping::type::doubleCovariantPiola:
    return dim * dim;
  case mapping::type::doubleContravariantPiola:
    return dim * dim;
  default:
    throw std::runtime_error("Mapping not yet implemented");
  }
}
//-----------------------------------------------------------------------------
