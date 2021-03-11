// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "moments.h"
#include "cell.h"
#include "finite-element.h"
#include "polyset.h"
#include "quadrature.h"

#include "utils.h"
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <xtensor/xio.hpp>

using namespace basix;

namespace
{
//----------------------------------------------------------------------------
std::vector<int> axis_points(const cell::type celltype)
{
  switch (celltype)
  {
  case cell::type::interval:
    return {1};
  case cell::type::triangle:
    return {1, 2};
  case cell::type::quadrilateral:
    return {1, 2};
  case cell::type::tetrahedron:
    return {1, 2, 3};
  case cell::type::hexahedron:
    return {1, 2, 4};
  default:
    throw std::runtime_error(
        "Integrals of this entity type not yet implemented.");
  }
}
//----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
// xt::xtensor<double, 3>
std::vector<Eigen::MatrixXd> moments::create_dot_moment_dof_transformations(
    const FiniteElement& moment_space)
{
  std::cout << "Momoment 0" << std::endl;

  cell::type celltype = moment_space.cell_type();
  if (celltype == cell::type::point)
    return {};

  std::cout << "Momoment 1" << std::endl;

  Eigen::ArrayXXd points = moment_space.points();
  Eigen::MatrixXd matrix = moment_space.interpolation_matrix();
  // std::vector<Eigen::ArrayXXd> transformed_pointsets;
  // std::vector<Eigen::ArrayXXd> J;
  // std::vector<Eigen::ArrayXXd> K;
  // std::vector<std::vector<double>> detJ;

  // xt::xtensor<double, 3> transformed_pointsets;
  // xt::xtensor<double, 3> J;
  // xt::xtensor<double, 3> K;
  // xt::xtensor<double, 2> detJ;

  xt::xarray<double> transformed_pointsets;
  xt::xarray<double> J;
  xt::xarray<double> K;
  xt::xarray<double> detJ;

  std::vector<std::size_t> s = {(std::size_t)points.rows()};
  if (points.cols() > 1)
    s.push_back(points.cols());
  auto pts = xt::adapt<xt::layout_type::column_major>(
      points.data(), points.size(), xt::no_ownership(), s);

  std::cout << "Momoment 2" << std::endl;

  if (celltype == cell::type::interval)
  {
    std::cout << "Momoment 3" << std::endl;

    // Eigen::ArrayXXd reflected_points(points.rows(), points.cols());
    // xt::xtensor<double ,1> reflected_points(points.rows());
    // for (int i = 0; i < points.rows(); ++i)
    //   reflected_points(i, 0) = 1 - points(i, 0);
    // transformed_pointsets.push_back(reflected_points);
    transformed_pointsets = 1.0 - pts;
    transformed_pointsets
        = xt::expand_dims(xt::expand_dims(transformed_pointsets, 1), 0);

    // Eigen::ArrayXXd J_part(points.rows(), 1);
    // J_part.col(0) = -1;
    // J.push_back(J_part);
    J = xt::full_like(pts, -1.0);
    J = xt::expand_dims(xt::expand_dims(J, 1), 0);

    // Eigen::ArrayXXd K_part(points.rows(), 1);
    // K_part.col(0) = -1;
    // K.push_back(K_part);
    K = xt::full_like(pts, -1.0);
    K = xt::expand_dims(xt::expand_dims(K, 1), 0);

    // std::vector<double> detJ_part(points.rows(), 1);
    // detJ.push_back(detJ_part);
    detJ = xt::ones_like(pts);
    detJ = xt::expand_dims(detJ, 0);
    std::cout << "Momoment 4" << std::endl;
  }
  else if (celltype == cell::type::triangle)
  {
    std::cout << "Momoment 5" << std::endl;

    std::array<std::size_t, 3> shape = {2, pts.shape()[0], pts.shape()[1]};
    transformed_pointsets = xt::zeros<double>(shape);
    for (std::size_t i = 0; i < pts.shape()[0]; ++i)
    {
      transformed_pointsets(0, i, 0) = pts(i, 1);
      transformed_pointsets(0, i, 1) = 1 - pts(i, 0) - pts(i, 1);
    }

    std::array<std::size_t, 3> shape2 = {2, pts.shape()[0], 4};
    J = xt::zeros<double>(shape2);
    K = xt::zeros<double>(shape2);
    detJ = xt::zeros<double>({(std::size_t)2, pts.shape()[0]});

    auto J0 = xt::view(J, 0, xt::all(), xt::all());
    xt::col(J0, 0) = 0;
    xt::col(J0, 1) = 1;
    xt::col(J0, 2) = -1;
    xt::col(J0, 3) = -1;

    auto K0 = xt::view(K, 0, xt::all(), xt::all());
    xt::col(K0, 0) = -1;
    xt::col(K0, 1) = -1;
    xt::col(K0, 2) = 1;
    xt::col(K0, 3) = 0;

    xt::view(detJ, 0, xt::all()) = 1.0;

    for (std::size_t i = 0; i < pts.shape()[0]; ++i)
    {
      transformed_pointsets(1, i, 0) = pts(i, 1);
      transformed_pointsets(1, i, 1) = pts(i, 0);
    }

    auto J1 = xt::view(J, 1, xt::all(), xt::all());
    xt::col(J1, 0) = 0;
    xt::col(J1, 1) = 1;
    xt::col(J1, 2) = 1;
    xt::col(J1, 3) = 0;

    auto K1 = xt::view(K, 1, xt::all(), xt::all());
    xt::col(K1, 0) = 0;
    xt::col(K1, 1) = 1;
    xt::col(K1, 2) = 1;
    xt::col(K1, 3) = 0;

    xt::view(detJ, 1, xt::all()) = 1.0;
    std::cout << "Momoment 6" << std::endl;
  }
  else if (celltype == cell::type::quadrilateral)
  {
    std::cout << "Momoment 7" << std::endl;

    std::array<std::size_t, 3> shape = {2, pts.shape()[0], pts.shape()[1]};
    transformed_pointsets = xt::zeros<double>(shape);
    for (std::size_t i = 0; i < pts.shape()[0]; ++i)
    {
      transformed_pointsets(0, i, 0) = pts(i, 1);
      transformed_pointsets(0, i, 1) = 1 - pts(i, 0);
    }

    std::array<std::size_t, 3> shape2 = {2, pts.shape()[0], 4};
    J = xt::zeros<double>(shape2);
    K = xt::zeros<double>(shape2);

    auto J0 = xt::view(J, 0, xt::all(), xt::all());
    xt::col(J0, 0) = 0;
    xt::col(J0, 1) = 1;
    xt::col(J0, 2) = -1;
    xt::col(J0, 3) = 0;

    auto K0 = xt::view(K, 0, xt::all(), xt::all());
    xt::col(K0, 0) = 0;
    xt::col(K0, 1) = -1;
    xt::col(K0, 2) = 1;
    xt::col(K0, 3) = 0;

    xt::view(detJ, 0, xt::all()) = 1.0;

    for (int i = 0; i < points.rows(); ++i)
    {
      transformed_pointsets(1, i, 0) = pts(i, 1);
      transformed_pointsets(1, i, 1) = pts(i, 0);
    }

    auto J1 = xt::view(J, 1, xt::all(), xt::all());
    xt::col(J1, 0) = 0;
    xt::col(J1, 1) = 1;
    xt::col(J1, 2) = 1;
    xt::col(J1, 3) = 0;

    auto K1 = xt::view(K, 1, xt::all(), xt::all());
    xt::col(K1, 0) = 0;
    xt::col(K1, 1) = 1;
    xt::col(K1, 2) = 1;
    xt::col(K1, 3) = 0;

    xt::view(detJ, 1, xt::all()) = 1.0;
    std::cout << "Momoment 8" << std::endl;
  }
  else
  {
    throw std::runtime_error(
        "DOF transformations only implemented for tdim <= 2.");
  }

  std::cout << "Momoment 9: " << std::endl;

  std::array<std::size_t, 3> shape
      = {transformed_pointsets.shape()[0], (std::size_t)moment_space.dim(),
         (std::size_t)moment_space.dim()};
  xt::xtensor<double, 3> out = xt::zeros<double>(shape);
  for (std::size_t i = 0; i < transformed_pointsets.shape()[0]; ++i)
  {
    // Eigen::ArrayXXd transformed_points = transformed_pointsets[i];
    // auto tpts = xt::view(transformed_pointsets, i, xt::all(), xt::all());

    Eigen::ArrayXXd _tpoint(transformed_pointsets.shape()[1],
                            transformed_pointsets.shape()[2]);
    std::cout << "OOOOppppps" << std::endl;
    for (std::size_t j = 0; j < transformed_pointsets.shape()[1]; ++j)
      for (std::size_t k = 0; k < transformed_pointsets.shape()[2]; ++k)
        _tpoint(j, k) = transformed_pointsets(i, j, k);

    std::cout << "Momoment 9b: " << std::endl;
    Eigen::ArrayXXd _J(J.shape()[1], J.shape()[2]);
    for (std::size_t j = 0; j < J.shape()[1]; ++j)
      for (std::size_t k = 0; k < J.shape()[2]; ++k)
        _J(j, k) = J(i, j, k);

    std::cout << "Momoment 9c: " << std::endl;
    std::vector<double> _detJ(detJ.shape()[1]);
    for (std::size_t j = 0; j < J.shape()[1]; ++j)
      _detJ[j] = detJ(i, j);

    std::cout << "Momoment 9d: " << std::endl;
    Eigen::ArrayXXd _K(K.shape()[1], K.shape()[2]);
    for (std::size_t j = 0; j < K.shape()[1]; ++j)
      for (std::size_t k = 0; k < K.shape()[2]; ++k)
        _K(j, k) = K(i, j, k);

    std::cout << "Momoment 9e: " << std::endl;
    Eigen::ArrayXXd moment_space_at_pts = moment_space.tabulate(0, _tpoint)[0];
    Eigen::ArrayXXd pulled
        = moment_space.map_pull_back(moment_space_at_pts, _J, _detJ, _K);

    std::array<std::size_t, 2> shape0
        = {(std::size_t)moment_space_at_pts.rows(),
           (std::size_t)moment_space_at_pts.cols()};
    auto _moment_space_at_pts = xt::adapt<xt::layout_type::column_major>(
        moment_space_at_pts.data(), moment_space_at_pts.size(),
        xt::no_ownership(), shape0);

    std::array<std::size_t, 2> shape1
        = {(std::size_t)pulled.rows(), (std::size_t)pulled.cols()};
    auto _pulled = xt::adapt<xt::layout_type::column_major>(
        pulled.data(), pulled.size(), xt::no_ownership(), shape1);

    std::array<std::size_t, 2> shape2
        = {(std::size_t)matrix.rows(), (std::size_t)matrix.cols()};
    auto _matrix = xt::adapt<xt::layout_type::column_major>(
        matrix.data(), matrix.size(), xt::no_ownership(), shape2);

    // Eigen::MatrixXd result
    //     = Eigen::MatrixXd::Zero(moment_space.dim(), moment_space.dim());

    for (int v = 0; v < moment_space.value_size(); ++v)
    {
      auto r = xt::view(out, i, xt::all(), xt::all());
      auto tmp0
          = xt::view(_matrix, xt::range(0, matrix.rows()),
                     xt::range(v * pulled.rows(), (v + 1) + pulled.rows()));
      auto tmp1 = xt::view(
          _pulled, xt::range(0, pulled.rows()),
          xt::range(moment_space.dim() * v, moment_space.dim() * (v + 1)));

      r += basix::linalg::dot(tmp0, tmp1);

      // result += matrix.block(0, v * pulled.rows(), matrix.rows(),
      // pulled.rows())
      //           * pulled.block(0, moment_space.dim() * v, pulled.rows(),
      //                          moment_space.dim());
      // result += matrix.block(0, v * pulled.rows(),
      // matrix.rows(), pulled.rows())
      //           * pulled
      //                 .block(0, moment_space.dim() * v,
      //                 pulled.rows(),
      //                        moment_space.dim())
      //                 .matrix();
    }

    // out.push_back(result);
  }

  std::cout << "Momoment 10" << std::endl;

  std::vector<Eigen::MatrixXd> outold;
  for (std::size_t i = 0; i < out.shape()[0]; ++i)
  {
    Eigen::MatrixXd mat(out.shape()[1], out.shape()[2]);
    for (std::size_t j = 0; j < out.shape()[1]; ++j)
    {
      for (std::size_t k = 0; k < out.shape()[2]; ++k)
        mat(j, k) = out(i, j, k);
    }
    outold.push_back(mat);
  }

  std::cout << "Momoment 11" << std::endl;
  return outold;
}
//----------------------------------------------------------------------------
std::vector<Eigen::MatrixXd>
moments::create_moment_dof_transformations(const FiniteElement& moment_space)
{
  std::vector<Eigen::MatrixXd> t
      = create_dot_moment_dof_transformations(moment_space);
  if (moment_space.cell_type() == cell::type::interval)
    return t;

  Eigen::Matrix2d rot;
  Eigen::Matrix2d ref;
  if (moment_space.cell_type() == cell::type::triangle)
  {
    rot << -1, -1, 1, 0;
    ref << 0, 1, 1, 0;
  }
  else if (moment_space.cell_type() == cell::type::quadrilateral)
  {
    // TODO: check that these are correct
    rot << 0, -1, 1, 0;
    ref << 0, 1, 1, 0;
  }

  const int scalar_dofs = t[0].rows();

  std::vector<Eigen::MatrixXd> out;
  {
    Eigen::MatrixXd M(scalar_dofs * 2, scalar_dofs * 2);
    for (int i = 0; i < scalar_dofs; ++i)
      for (int j = 0; j < scalar_dofs; ++j)
        M.block(2 * i, 2 * j, 2, 2) = t[0](i, j) * rot;

    out.push_back(M);
  }
  {
    Eigen::MatrixXd M(scalar_dofs * 2, scalar_dofs * 2);
    for (int i = 0; i < scalar_dofs; ++i)
      for (int j = 0; j < scalar_dofs; ++j)
        M.block(2 * i, 2 * j, 2, 2) = t[1](i, j) * ref;

    out.push_back(M);
  }

  return out;
}
//----------------------------------------------------------------------------
std::vector<Eigen::MatrixXd> moments::create_normal_moment_dof_transformations(
    const FiniteElement& moment_space)
{
  std::vector<Eigen::MatrixXd> t
      = create_dot_moment_dof_transformations(moment_space);
  const int tdim = cell::topological_dimension(moment_space.cell_type());
  if (tdim == 1)
    t[0] *= -1;
  if (tdim == 2)
    t[1] *= -1;
  return t;
}
//----------------------------------------------------------------------------
std::vector<Eigen::MatrixXd> moments::create_tangent_moment_dof_transformations(
    const FiniteElement& moment_space)
{
  std::vector<Eigen::MatrixXd> t
      = create_dot_moment_dof_transformations(moment_space);
  const int tdim = cell::topological_dimension(moment_space.cell_type());
  if (tdim == 1)
    t[0] *= -1;
  if (tdim == 2)
    throw std::runtime_error("Tangent is only well-defined on an edge.");
  return t;
}
//----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
moments::make_integral_moments(const FiniteElement& moment_space,
                               cell::type celltype, int value_size, int q_deg)
{
  const cell::type sub_celltype = moment_space.cell_type();
  const int sub_entity_dim = cell::topological_dimension(sub_celltype);
  if (sub_entity_dim == 0)
    throw std::runtime_error("Cannot integrate over a dimension 0 entity.");
  const int sub_entity_count = cell::sub_entity_count(celltype, sub_entity_dim);
  const int tdim = cell::topological_dimension(celltype);

  auto [Qpts, Qwts]
      = quadrature::make_quadrature("default", sub_celltype, q_deg);

  // Evaluate moment space at quadrature points
  Eigen::ArrayXXd moment_space_at_Qpts = moment_space.tabulate(0, Qpts)[0];
  Eigen::ArrayXXd points(sub_entity_count * Qpts.rows(), tdim);
  Eigen::MatrixXd matrix(moment_space_at_Qpts.cols() * sub_entity_count
                             * (value_size == 1 ? 1 : sub_entity_dim),
                         sub_entity_count * Qpts.rows() * value_size);
  matrix.setZero();

  std::vector<int> axis_pts = axis_points(celltype);

  int c = 0;
  // Iterate over sub entities
  for (int i = 0; i < sub_entity_count; ++i)
  {
    xt::xtensor<double, 2> entity
        = cell::sub_entity_geometry(celltype, sub_entity_dim, i);
    Eigen::Map<
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        _entity(entity.data(), entity.shape()[0], entity.shape()[1]);

    // Parametrise entity coordinates
    Eigen::ArrayXXd axes(sub_entity_dim, tdim);
    for (int j = 0; j < sub_entity_dim; ++j)
      axes.row(j) = _entity.row(axis_pts[j]) - _entity.row(0);

    // Map quadrature points onto entity
    points.block(Qpts.rows() * i, 0, Qpts.rows(), tdim)
        = _entity.row(0).replicate(Qpts.rows(), 1)
          + (Qpts.matrix() * axes.matrix()).array();

    // Compute entity integral moments
    for (int j = 0; j < moment_space_at_Qpts.cols(); ++j)
    {
      Eigen::ArrayXd phi = moment_space_at_Qpts.col(j);
      if (value_size == 1)
      {
        Eigen::RowVectorXd q = phi * Qwts;
        matrix.block(c, i * Qpts.rows(), 1, Qpts.rows()) = q;
        ++c;
      }
      else
      {
        // FIXME: This assumed that the moment space has a certain mapping type
        for (int d = 0; d < sub_entity_dim; ++d)
        {
          Eigen::VectorXd axis = axes.row(d);
          for (int k = 0; k < value_size; ++k)
          {
            Eigen::RowVectorXd q = phi * Qwts * axis(k);
            matrix.block(c, (k * sub_entity_count + i) * Qpts.rows(), 1,
                         Qpts.rows())
                = q;
          }
          ++c;
        }
      }
    }
  }

  return std::make_pair(points, matrix);
}
//----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
moments::make_dot_integral_moments(const FiniteElement& moment_space,
                                   cell::type celltype, int value_size,
                                   int q_deg)
{
  const cell::type sub_celltype = moment_space.cell_type();
  const int sub_entity_dim = cell::topological_dimension(sub_celltype);
  const int sub_entity_count = cell::sub_entity_count(celltype, sub_entity_dim);
  const int tdim = cell::topological_dimension(celltype);

  auto [Qpts, Qwts]
      = quadrature::make_quadrature("default", sub_celltype, q_deg);

  // If this is always true, value_size input can be removed
  assert(tdim == value_size);

  // Evaluate moment space at quadrature points
  Eigen::ArrayXXd moment_space_at_Qpts = moment_space.tabulate(0, Qpts)[0];
  const int moment_space_size = moment_space_at_Qpts.cols() / sub_entity_dim;

  Eigen::ArrayXXd points(sub_entity_count * Qpts.rows(), tdim);
  Eigen::MatrixXd matrix
      = Eigen::MatrixXd::Zero(moment_space_size * sub_entity_count,
                              sub_entity_count * Qpts.rows() * value_size);

  std::vector<int> axis_pts = axis_points(celltype);

  int c = 0;
  // Iterate over sub entities
  for (int i = 0; i < sub_entity_count; ++i)
  {
    xt::xtensor<double, 2> entity
        = cell::sub_entity_geometry(celltype, sub_entity_dim, i);
    Eigen::Map<
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        _entity(entity.data(), entity.shape()[0], entity.shape()[1]);

    // Parametrise entity coordinates
    Eigen::ArrayXXd axes(sub_entity_dim, tdim);
    for (int j = 0; j < sub_entity_dim; ++j)
      axes.row(j) = _entity.row(axis_pts[j]) - _entity.row(0);

    // Map quadrature points onto entity
    points.block(Qpts.rows() * i, 0, Qpts.rows(), tdim)
        = _entity.row(0).replicate(Qpts.rows(), 1)
          + (Qpts.matrix() * axes.matrix()).array();

    // Compute entity integral moments
    for (int j = 0; j < moment_space_size; ++j)
    {
      for (int k = 0; k < value_size; ++k)
      {
        Eigen::RowVectorXd q = Eigen::VectorXd::Zero(Qwts.rows());
        for (int d = 0; d < sub_entity_dim; ++d)
        {
          Eigen::ArrayXd phi
              = moment_space_at_Qpts.col(d * moment_space_size + j);
          Eigen::RowVectorXd axis = axes.row(d);
          // FIXME: This assumed that the moment space has a certain mapping
          // type
          Eigen::RowVectorXd qpart = phi * Qwts * axis(k);
          q += qpart;
        }
        matrix.block(c, (k * sub_entity_count + i) * Qpts.rows(), 1,
                     Qpts.rows())
            = q;
      }
      ++c;
    }
  }

  return std::make_pair(points, matrix);
}
//----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
moments::make_tangent_integral_moments(const FiniteElement& moment_space,
                                       cell::type celltype, int value_size,
                                       int q_deg)
{
  const cell::type sub_celltype = moment_space.cell_type();
  const int sub_entity_dim = cell::topological_dimension(sub_celltype);
  const int sub_entity_count = cell::sub_entity_count(celltype, sub_entity_dim);
  const int tdim = cell::topological_dimension(celltype);

  if (sub_entity_dim != 1)
    throw std::runtime_error("Tangent is only well-defined on an edge.");

  auto [Qpts, Qwts]
      = quadrature::make_quadrature("default", cell::type::interval, q_deg);

  // If this is always true, value_size input can be removed
  assert(tdim == value_size);

  // Evaluate moment space at quadrature points
  Eigen::ArrayXXd moment_space_at_Qpts = moment_space.tabulate(0, Qpts)[0];

  Eigen::ArrayXXd points(sub_entity_count * Qpts.rows(), tdim);
  Eigen::MatrixXd matrix(moment_space_at_Qpts.cols() * sub_entity_count,
                         sub_entity_count * Qpts.rows() * value_size);
  matrix.setZero();

  int c = 0;

  // Iterate over sub entities
  for (int i = 0; i < sub_entity_count; ++i)
  {
    xt::xtensor<double, 2> edge = cell::sub_entity_geometry(celltype, 1, i);
    Eigen::Map<
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        _edge(edge.data(), edge.shape()[0], edge.shape()[1]);

    Eigen::VectorXd tangent = _edge.row(1) - _edge.row(0);
    // No need to normalise the tangent, as the size of this is equal to the
    // integral jacobian

    // Map quadrature points onto triangle edge
    Eigen::ArrayXXd Qpts_scaled(Qpts.rows(), tdim);
    for (int j = 0; j < Qpts.rows(); ++j)
    {
      points.row(i * Qpts.rows() + j)
          = _edge.row(0) + Qpts(j, 0) * (_edge.row(1) - _edge.row(0));
    }

    // Compute edge tangent integral moments
    for (int j = 0; j < moment_space_at_Qpts.cols(); ++j)
    {
      Eigen::ArrayXd phi = moment_space_at_Qpts.col(j);
      for (int k = 0; k < value_size; ++k)
      {
        Eigen::RowVectorXd data = phi * Qwts * tangent[k];
        matrix.block(c, k * sub_entity_count * Qpts.rows() + i * Qpts.rows(), 1,
                     Qpts.rows())
            = data;
      }
      ++c;
    }
  }

  return std::make_pair(points, matrix);
}
//----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
moments::make_normal_integral_moments(const FiniteElement& moment_space,
                                      const cell::type celltype,
                                      const int value_size, const int q_deg)
{
  const cell::type sub_celltype = moment_space.cell_type();
  const int sub_entity_dim = cell::topological_dimension(sub_celltype);
  const int sub_entity_count = cell::sub_entity_count(celltype, sub_entity_dim);
  const int tdim = cell::topological_dimension(celltype);

  if (sub_entity_dim != tdim - 1)
    throw std::runtime_error("Normal is only well-defined on a facet.");

  auto [Qpts, Qwts]
      = quadrature::make_quadrature("default", sub_celltype, q_deg);

  // If this is always true, value_size input can be removed
  assert(tdim == value_size);

  // Evaluate moment space at quadrature points
  Eigen::ArrayXXd moment_space_at_Qpts = moment_space.tabulate(0, Qpts)[0];

  Eigen::ArrayXXd points(sub_entity_count * Qpts.rows(), tdim);
  Eigen::MatrixXd matrix(moment_space_at_Qpts.cols() * sub_entity_count,
                         sub_entity_count * Qpts.rows() * value_size);
  matrix.setZero();

  int c = 0;

  // Iterate over sub entities
  Eigen::VectorXd normal(tdim);
  Eigen::ArrayXXd Qpts_scaled(Qpts.rows(), tdim);
  for (int i = 0; i < sub_entity_count; ++i)
  {
    xt::xtensor<double, 2> facet
        = cell::sub_entity_geometry(celltype, tdim - 1, i);
    Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        _facet(facet.data(), facet.shape()[0], facet.shape()[1]);

    if (tdim == 2)
    {
      Eigen::Vector2d tangent = _facet.row(1) - _facet.row(0);
      normal << -tangent(1), tangent(0);
      // No need to normalise the normal, as the size of this is equal to the
      // integral jacobian

      // Map quadrature points onto facet
      for (int j = 0; j < Qpts.rows(); ++j)
      {
        points.row(i * Qpts.rows() + j)
            = _facet.row(0) + Qpts(j, 0) * (_facet.row(1) - _facet.row(0));
      }
    }
    else if (tdim == 3)
    {
      Eigen::Vector3d t0 = _facet.row(1) - _facet.row(0);
      Eigen::Vector3d t1 = _facet.row(2) - _facet.row(0);
      normal = t0.cross(t1);

      // No need to normalise the normal, as the size of this is equal to the
      // integral jacobian

      // Map quadrature points onto facet
      for (int j = 0; j < Qpts.rows(); ++j)
      {
        points.row(i * Qpts.rows() + j)
            = _facet.row(0) + Qpts(j, 0) * (_facet.row(1) - _facet.row(0))
              + Qpts(j, 1) * (_facet.row(2) - _facet.row(0));
      }
    }
    else
      throw std::runtime_error("Normal on this cell cannot be computed.");

    // Compute facet normal integral moments
    for (int j = 0; j < moment_space_at_Qpts.cols(); ++j)
    {
      Eigen::ArrayXd phi = moment_space_at_Qpts.col(j);
      for (int k = 0; k < value_size; ++k)
      {
        Eigen::RowVectorXd q = phi * Qwts * normal[k];
        matrix.block(c, k * sub_entity_count * Qpts.rows() + i * Qpts.rows(), 1,
                     Qpts.rows())
            = q;
      }
      ++c;
    }
  }

  return std::make_pair(points, matrix);
}
//----------------------------------------------------------------------------
