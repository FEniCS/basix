// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"
#include "brezzi-douglas-marini.h"
#include "bubble.h"
#include "crouzeix-raviart.h"
#include "lagrange.h"
#include "nce-rtc.h"
#include "nedelec.h"
#include "polyset.h"
#include "raviart-thomas.h"
#include "regge.h"
#include "serendipity.h"
#include <numeric>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xlayout.hpp>
#include <xtensor/xview.hpp>

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
  case element::family::Serendipity:
    return create_serendipity(cell, degree);
  case element::family::DPC:
    return create_dpc(cell, degree);
  default:
    throw std::runtime_error("Family not found");
  }
}
//-----------------------------------------------------------------------------
Eigen::MatrixXd basix::compute_expansion_coefficients(
    cell::type celltype, const Eigen::MatrixXd& B, const Eigen::MatrixXd& M,
    const Eigen::ArrayXXd& x, int degree, double kappa_tol)
{
  std::vector<std::size_t> shape = {(std::size_t)x.rows()};
  if (x.cols() > 1)
    shape.push_back(x.cols());
  auto _x = xt::adapt<xt::layout_type::column_major>(x.data(), x.size(),
                                                     xt::no_ownership(), shape);
  const xt::xtensor<double, 3> P = polyset::tabulate(celltype, degree, 0, _x);

  const int coeff_size = P.shape()[2];
  const int value_size = B.cols() / coeff_size;
  const int m_size = M.cols() / value_size;

  std::array<std::size_t, 2> Bshape
      = {(std::size_t)B.rows(), (std::size_t)B.cols()};
  std::array<std::size_t, 2> Mshape
      = {(std::size_t)M.rows(), (std::size_t)M.cols()};
  xt::xtensor<double, 2, xt::layout_type::column_major> _B
      = xt::adapt<xt::layout_type::column_major>(B.data(), B.size(),
                                                 xt::no_ownership(), Bshape);
  auto _M = xt::adapt<xt::layout_type::column_major>(
      M.data(), M.size(), xt::no_ownership(), Mshape);
  xt::xtensor<double, 2, xt::layout_type::column_major> A
      = xt::zeros<double>({_B.shape()[0], _M.shape()[0]});
  for (int row = 0; row < B.rows(); ++row)
  {
    for (int v = 0; v < value_size; ++v)
    {
      auto Bview
          = xt::view(_B, row, xt::range(v * coeff_size, (v + 1) * coeff_size));
      auto Mview_t
          = xt::view(_M, xt::all(), xt::range(v * m_size, (v + 1) * m_size));

      // Compute Aview = Bview * Pt * Mview ( Aview_i = Bview_j * Pt_jk *
      // Mview_ki )
      for (std::size_t i = 0; i < A.shape()[1]; ++i)
        for (std::size_t k = 0; k < P.shape()[1]; ++k)
          for (std::size_t j = 0; j < P.shape()[2]; ++j)
            A(row, i) += Bview(j) * P(0, k, j) * Mview_t(i, k);
    }
  }

  if (kappa_tol >= 1.0)
  {
    if (xt::linalg::cond(A, 2) > kappa_tol)
    {
      throw std::runtime_error("Condition number of B.D^T when computing "
                               "expansion coefficients exceeds tolerance.");
    }
  }

  xt::xtensor<double, 2, xt::layout_type::column_major> coeff
      = xt::linalg::solve(A, _B);
  return Eigen::Map<const Eigen::MatrixXd>(coeff.data(), coeff.shape()[0],
                                           coeff.shape()[1]);
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

  if (points_1d.cols() > 0)
    points.block(0, 0, points_1d.rows(), tdim) = points_1d;

  if (points_2d.cols() > 0)
    points.block(points_1d.rows(), 0, points_2d.rows(), tdim) = points_2d;

  if (points_3d.cols() > 0)
  {
    points.block(points_1d.rows() + points_2d.rows(), 0, points_3d.rows(), tdim)
        = points_3d;
  }

  Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(
      matrix_1d.rows() + matrix_2d.rows() + matrix_3d.rows(),
      matrix_1d.cols() + matrix_2d.cols() + matrix_3d.cols());

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
    const std::vector<Eigen::MatrixXd>& base_transformations,
    const Eigen::ArrayXXd& points, const Eigen::MatrixXd M,
    mapping::type map_type)
    : _cell_type(cell_type), _family(family), _degree(degree),
      _value_shape(value_shape), _map_type(map_type), _coeffs(coeffs),
      _entity_dofs(entity_dofs), _base_transformations(base_transformations),
      _points(points), _matM(M)
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
  _map_push_forward = mapping::get_forward_map(map_type);
}
//-----------------------------------------------------------------------------
cell::type FiniteElement::cell_type() const { return _cell_type; }
//-----------------------------------------------------------------------------
int FiniteElement::degree() const { return _degree; }
//-----------------------------------------------------------------------------
int FiniteElement::value_size() const
{
  int value_size = 1;
  for (int d : _value_shape)
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
mapping::type FiniteElement::mapping_type() const { return _map_type; }
//-----------------------------------------------------------------------------
const Eigen::MatrixXd& FiniteElement::interpolation_matrix() const
{
  return _matM;
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
  tabulate(nd, x, basis_data.data());

  std::vector<Eigen::ArrayXXd> dresult;
  for (int p = 0; p < ndsize; ++p)
  {
    dresult.push_back(Eigen::Map<Eigen::ArrayXXd>(
        basis_data.data() + p * x.rows() * ndofs * vs, x.rows(), ndofs * vs));
  }

  return dresult;
}
//-----------------------------------------------------------------------------
void FiniteElement::tabulate(int nd, const Eigen::ArrayXXd& x,
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
std::vector<Eigen::MatrixXd> FiniteElement::base_transformations() const
{
  return _base_transformations;
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
    const tcb::span<const double>& detJ,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        K) const
{
  const int reference_dim = cell::topological_dimension(_cell_type);
  const int physical_dim = J.cols() / reference_dim;
  const int physical_value_size = compute_value_size(_map_type, physical_dim);
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
    {
      Eigen::ArrayXd col = reference_block.col(i);
      std::vector<double> u
          = _map_push_forward(col, current_J, detJ[pt], current_K);
      for (std::size_t j = 0; j < u.size(); ++j)
        physical_block(j, i) = u[j];
    }
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
    const tcb::span<const double>& detJ,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        K) const
{
  const int reference_dim = cell::topological_dimension(_cell_type);
  const int physical_dim = J.cols() / reference_dim;
  const int physical_value_size = compute_value_size(_map_type, physical_dim);
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
    {
      Eigen::ArrayXd col = physical_block.col(i);
      std::vector<double> U
          = _map_push_forward(col, current_K, 1 / detJ[pt], current_J);
      for (std::size_t j = 0; j < U.size(); ++j)
        reference_block(j, i) = U[j];
    }
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
int FiniteElement::compute_value_size(mapping::type map_type, int dim)
{
  switch (map_type)
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
