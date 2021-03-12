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
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <xtensor-blas/xlinalg.hpp>
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
std::vector<Eigen::MatrixXd> myconvert(xt::xtensor<double, 3> M)
{
  std::vector<Eigen::MatrixXd> outold;
  for (std::size_t i = 0; i < M.shape()[0]; ++i)
  {
    Eigen::MatrixXd mat(M.shape()[1], M.shape()[2]);
    for (std::size_t j = 0; j < M.shape()[1]; ++j)
    {
      for (std::size_t k = 0; k < M.shape()[2]; ++k)
        mat(j, k) = M(i, j, k);
    }
    outold.push_back(mat);
  }
  return outold;
}
//----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::vector<Eigen::MatrixXd> moments::create_dot_moment_dof_transformations(
    const FiniteElement& moment_space)
{
  return myconvert(create_dot_moment_dof_transformations_new(moment_space));
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 3> moments::create_dot_moment_dof_transformations_new(
    const FiniteElement& moment_space)
{
  cell::type celltype = moment_space.cell_type();
  if (celltype == cell::type::point)
    return {};

  Eigen::ArrayXXd _points = moment_space.points();
  std::vector<std::size_t> s = {(std::size_t)_points.rows()};
  if (_points.cols() > 1)
    s.push_back(_points.cols());
  auto pts = xt::adapt<xt::layout_type::column_major>(
      _points.data(), _points.size(), xt::no_ownership(), s);

  Eigen::MatrixXd _matrix = moment_space.interpolation_matrix();
  s = {(std::size_t)_matrix.rows()};
  if (_matrix.cols() > 1)
    s.push_back(_matrix.cols());
  auto matrix = xt::adapt<xt::layout_type::column_major>(
      _matrix.data(), _matrix.size(), xt::no_ownership(), s);

  xt::xtensor<double, 3> tpts;
  xt::xtensor<double, 3> J;
  xt::xtensor<double, 3> K;
  xt::xtensor<double, 2> detJ;
  if (celltype == cell::type::interval)
  {
    tpts = xt::atleast_3d(1.0 - pts);
    J = xt::full_like(tpts, -1.0);
    K = xt::full_like(tpts, -1.0);
    detJ = xt::atleast_2d(xt::ones_like(pts));
  }
  else if (celltype == cell::type::triangle)
  {
    std::array<std::size_t, 3> shape = {2, pts.shape()[0], pts.shape()[1]};
    std::array<std::size_t, 3> shape2 = {2, pts.shape()[0], 4};
    tpts = xt::zeros<double>(shape);
    J = xt::zeros<double>(shape2);
    K = xt::zeros<double>(shape2);
    detJ = xt::zeros<double>({(std::size_t)2, pts.shape()[0]});

    // Case 0
    xt::view(detJ, 0, xt::all()) = 1.0;
    for (std::size_t i = 0; i < pts.shape()[0]; ++i)
    {
      tpts(0, i, 0) = pts(i, 1);
      tpts(0, i, 1) = 1 - pts(i, 0) - pts(i, 1);
    }

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

    // Case 1
    xt::view(detJ, 1, xt::all()) = 1.0;
    for (std::size_t i = 0; i < pts.shape()[0]; ++i)
    {
      tpts(1, i, 0) = pts(i, 1);
      tpts(1, i, 1) = pts(i, 0);
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
  }
  else if (celltype == cell::type::quadrilateral)
  {
    std::array<std::size_t, 3> shape0 = {2, pts.shape()[0], pts.shape()[1]};
    tpts = xt::zeros<double>(shape0);
    std::array<std::size_t, 3> shape1 = {2, pts.shape()[0], 4};
    J = xt::zeros<double>(shape1);
    K = xt::zeros<double>(shape1);
    detJ = xt::zeros<double>({(std::size_t)2, pts.shape()[0]});

    // Case 0
    xt::view(detJ, 0, xt::all()) = 1.0;
    for (std::size_t i = 0; i < pts.shape()[0]; ++i)
    {
      tpts(0, i, 0) = pts(i, 1);
      tpts(0, i, 1) = 1.0 - pts(i, 0);
    }

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

    // Case 1
    xt::view(detJ, 1, xt::all()) = 1.0;
    for (std::size_t i = 0; i < pts.shape()[0]; ++i)
    {
      tpts(1, i, 0) = pts(i, 1);
      tpts(1, i, 1) = pts(i, 0);
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
  }
  else
  {
    throw std::runtime_error(
        "DOF transformations only implemented for tdim <= 2.");
  }

  std::array<std::size_t, 3> shape
      = {tpts.shape()[0], (std::size_t)moment_space.dim(),
         (std::size_t)moment_space.dim()};
  xt::xtensor<double, 3> out = xt::zeros<double>(shape);
  for (std::size_t i = 0; i < tpts.shape()[0]; ++i)
  {
    // TMP: copy data into Eigen
    Eigen::ArrayXXd _tpoint(tpts.shape()[1], tpts.shape()[2]);
    for (std::size_t j = 0; j < tpts.shape()[1]; ++j)
      for (std::size_t k = 0; k < tpts.shape()[2]; ++k)
        _tpoint(j, k) = tpts(i, j, k);
    Eigen::ArrayXXd _J(J.shape()[1], J.shape()[2]);
    for (std::size_t j = 0; j < J.shape()[1]; ++j)
      for (std::size_t k = 0; k < J.shape()[2]; ++k)
        _J(j, k) = J(i, j, k);
    std::vector<double> _detJ(detJ.shape()[1]);
    for (std::size_t j = 0; j < J.shape()[1]; ++j)
      _detJ[j] = detJ(i, j);
    Eigen::ArrayXXd _K(K.shape()[1], K.shape()[2]);
    for (std::size_t j = 0; j < K.shape()[1]; ++j)
      for (std::size_t k = 0; k < K.shape()[2]; ++k)
        _K(j, k) = K(i, j, k);

    Eigen::ArrayXXd moment_space_pts = moment_space.tabulate(0, _tpoint)[0];
    Eigen::ArrayXXd pulled
        = moment_space.map_pull_back(moment_space_pts, _J, _detJ, _K);

    std::array<std::size_t, 2> shape0 = {(std::size_t)moment_space_pts.rows(),
                                         (std::size_t)moment_space_pts.cols()};
    auto _moment_space_pts = xt::adapt<xt::layout_type::column_major>(
        moment_space_pts.data(), moment_space_pts.size(), xt::no_ownership(),
        shape0);

    std::array<std::size_t, 2> shape1
        = {(std::size_t)pulled.rows(), (std::size_t)pulled.cols()};
    auto _pulled = xt::adapt<xt::layout_type::column_major>(
        pulled.data(), pulled.size(), xt::no_ownership(), shape1);

    for (int v = 0; v < moment_space.value_size(); ++v)
    {
      auto tmp0
          = xt::view(matrix, xt::range(0, matrix.shape()[0]),
                     xt::range(v * pulled.rows(), (v + 1) * pulled.rows()));
      auto tmp1 = xt::view(
          _pulled, xt::range(0, pulled.rows()),
          xt::range(moment_space.dim() * v, moment_space.dim() * (v + 1)));
      xt::view(out, i, xt::all(), xt::all()) += xt::linalg::dot(tmp0, tmp1);
    }
  }

  return out;
}
//----------------------------------------------------------------------------
std::vector<Eigen::MatrixXd>
moments::create_moment_dof_transformations(const FiniteElement& moment_space)
{
  const xt::xtensor<double, 3> t
      = create_dot_moment_dof_transformations_new(moment_space);
  if (moment_space.cell_type() == cell::type::interval)
    return myconvert(t);

  xt::xtensor_fixed<double, xt::xshape<2, 2>> rot, ref;
  if (moment_space.cell_type() == cell::type::triangle)
  {
    rot = {{-1, -1}, {1, 0}};
    ref = {{0, 1}, {1, 0}};
  }
  else if (moment_space.cell_type() == cell::type::quadrilateral)
  {
    // TODO: check that these are correct
    rot = {{0, -1}, {1, 0}};
    ref = {{0, 1}, {1, 0}};
  }

  const std::size_t scalar_dofs = t.shape()[1];
  xt::xtensor<double, 3> M({2, 2 * scalar_dofs, 2 * scalar_dofs});
  for (std::size_t i = 0; i < scalar_dofs; ++i)
  {
    for (std::size_t j = 0; j < scalar_dofs; ++j)
    {
      xt::view(M, 0, xt::range(2 * i, 2 * i + 2), xt::range(2 * j, 2 * j + 2))
          = t(0, i, j) * rot;
    }
  }

  for (std::size_t i = 0; i < scalar_dofs; ++i)
  {
    for (std::size_t j = 0; j < scalar_dofs; ++j)
    {
      xt::view(M, 1, xt::range(2 * i, 2 * i + 2), xt::range(2 * j, 2 * j + 2))
          = t(1, i, j) * ref;
    }
  }

  return myconvert(M);
}
//----------------------------------------------------------------------------
std::vector<Eigen::MatrixXd> moments::create_normal_moment_dof_transformations(
    const FiniteElement& moment_space)
{
  xt::xtensor<double, 3> t
      = create_dot_moment_dof_transformations_new(moment_space);
  const int tdim = cell::topological_dimension(moment_space.cell_type());
  if (tdim == 1 or tdim == 2)
    xt::view(t, tdim - 1, xt::all(), xt::all()) *= -1.0;
  return myconvert(t);
}
//----------------------------------------------------------------------------
std::vector<Eigen::MatrixXd> moments::create_tangent_moment_dof_transformations(
    const FiniteElement& moment_space)
{
  const int tdim = cell::topological_dimension(moment_space.cell_type());
  // FIXME: Should this check by tdim != 1?
  if (tdim == 2)
    throw std::runtime_error("Tangent is only well-defined on an edge.");

  xt::xtensor<double, 3> t
      = create_dot_moment_dof_transformations_new(moment_space);
  if (tdim == 1)
    xt::view(t, 0, xt::all(), xt::all()) *= -1.0;

  return myconvert(t);
}
//----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
moments::make_integral_moments(const FiniteElement& moment_space,
                               cell::type celltype, std::size_t value_size,
                               int q_deg)
{
  auto [points, matrix]
      = make_integral_moments_new(moment_space, celltype, value_size, q_deg);

  // TMP: Copy into Eigen
  Eigen::ArrayXXd _points(points.shape()[0], points.shape()[1]);
  Eigen::MatrixXd _matrix(matrix.shape()[0], matrix.shape()[1]);
  for (std::size_t i = 0; i < points.shape()[0]; ++i)
    for (std::size_t j = 0; j < points.shape()[1]; ++j)
      _points(i, j) = points(i, j);
  for (std::size_t i = 0; i < matrix.shape()[0]; ++i)
    for (std::size_t j = 0; j < matrix.shape()[1]; ++j)
      _matrix(i, j) = matrix(i, j);

  return std::make_pair(_points, _matrix);
}
//----------------------------------------------------------------------------
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
moments::make_integral_moments_new(const FiniteElement& moment_space,
                                   cell::type celltype, std::size_t value_size,
                                   int q_deg)
{
  const cell::type sub_celltype = moment_space.cell_type();
  const std::size_t sub_entity_dim = cell::topological_dimension(sub_celltype);
  if (sub_entity_dim == 0)
    throw std::runtime_error("Cannot integrate over a dimension 0 entity.");
  const std::size_t sub_entity_count
      = cell::sub_entity_count(celltype, sub_entity_dim);
  const std::size_t tdim = cell::topological_dimension(celltype);

  auto [Qpts, _Qwts]
      = quadrature::make_quadrature_new("default", sub_celltype, q_deg);
  auto Qwts = xt::adapt(_Qwts);
  if (Qpts.dimension() == 1)
    Qpts = Qpts.reshape({Qpts.shape()[0], 1});

  // TMP: Copy into Eigen array
  Eigen::ArrayXXd _Qpts(Qpts.shape()[0], Qpts.shape()[1]);
  for (std::size_t i = 0; i < Qpts.shape()[0]; ++i)
    for (std::size_t j = 0; j < Qpts.shape()[1]; ++j)
      _Qpts(i, j) = Qpts(i, j);

  // Evaluate moment space at quadrature points
  Eigen::ArrayXXd _moment_space_at_Qpts = moment_space.tabulate(0, _Qpts)[0];
  std::array<std::size_t, 2> shape1
      = {(std::size_t)_moment_space_at_Qpts.rows(),
         (std::size_t)_moment_space_at_Qpts.cols()};
  auto moment_space_at_Qpts = xt::adapt<xt::layout_type::column_major>(
      _moment_space_at_Qpts.data(), _moment_space_at_Qpts.size(),
      xt::no_ownership(), shape1);
  xt::xtensor<double, 2> points({sub_entity_count * Qpts.shape()[0], tdim});
  const std::array<std::size_t, 2> shape
      = {moment_space_at_Qpts.shape()[1] * sub_entity_count
             * (value_size == 1 ? 1 : sub_entity_dim),
         sub_entity_count * Qpts.shape()[0] * value_size};
  xt::xtensor<double, 2> matrix = xt::zeros<double>(shape);

  // Iterate over sub entities
  int c = 0;
  std::vector<int> axis_pts = axis_points(celltype);
  for (std::size_t i = 0; i < sub_entity_count; ++i)
  {
    xt::xtensor<double, 2> entity
        = cell::sub_entity_geometry(celltype, sub_entity_dim, i);
    Eigen::Map<
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        _entity(entity.data(), entity.shape()[0], entity.shape()[1]);

    // Parametrise entity coordinates
    xt::xtensor<double, 2> axes({sub_entity_dim, tdim});
    for (std::size_t j = 0; j < sub_entity_dim; ++j)
    {
      xt::view(axes, j, xt::all()) = xt::view(entity, axis_pts[j], xt::all())
                                     - xt::view(entity, 0, xt::all());
    }

    // See
    // https://github.com/xtensor-stack/xtensor/issues/1922#issuecomment-586317746
    // for why xt::newaxis() is required
    auto points_view = xt::view(
        points, xt::range(i * Qpts.shape()[0], (i + 1) * Qpts.shape()[0]),
        xt::range(0, tdim));
    auto p = xt::tile(xt::view(entity, xt::newaxis(), 0), Qpts.shape()[0]);
    points_view = p + xt::linalg::dot(Qpts, axes);

    // Compute entity integral moments
    for (std::size_t j = 0; j < moment_space_at_Qpts.shape()[1]; ++j)
    {
      auto phi = xt::col(moment_space_at_Qpts, j);
      if (value_size == 1)
      {
        xt::view(matrix, c,
                 xt::range(i * Qpts.shape()[0], (i + 1) * Qpts.shape()[0]))
            = phi * Qwts;
        ++c;
      }
      else
      {
        // FIXME: This assumed that the moment space has a certain
        // mapping type
        for (std::size_t d = 0; d < sub_entity_dim; ++d)
        {
          auto axis = xt::row(axes, d);
          for (std::size_t k = 0; k < value_size; ++k)
          {
            std::size_t offset = (k * sub_entity_count + i) * Qpts.shape()[0];
            xt::view(matrix, c, xt::range(offset, offset + Qpts.shape()[0]))
                = phi * Qwts * axis[k];
          }
          ++c;
        }
      }
    }
  }

  return {points, matrix};
}
//----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
moments::make_dot_integral_moments(const FiniteElement& moment_space,
                                   cell::type celltype, int value_size,
                                   int q_deg)
{
  auto [points, matrix] = make_dot_integral_moments_new(moment_space, celltype,
                                                        value_size, q_deg);

  // TMP: Copy into Eigen
  Eigen::ArrayXXd _points(points.shape()[0], points.shape()[1]);
  Eigen::MatrixXd _matrix(matrix.shape()[0], matrix.shape()[1]);
  for (std::size_t i = 0; i < points.shape()[0]; ++i)
    for (std::size_t j = 0; j < points.shape()[1]; ++j)
      _points(i, j) = points(i, j);
  for (std::size_t i = 0; i < matrix.shape()[0]; ++i)
    for (std::size_t j = 0; j < matrix.shape()[1]; ++j)
      _matrix(i, j) = matrix(i, j);

  return std::make_pair(_points, _matrix);
}
//----------------------------------------------------------------------------
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
moments::make_dot_integral_moments_new(const FiniteElement& moment_space,
                                       cell::type celltype,
                                       std::size_t value_size, int q_deg)
{
  const cell::type sub_celltype = moment_space.cell_type();
  const std::size_t sub_entity_dim = cell::topological_dimension(sub_celltype);
  const std::size_t sub_entity_count
      = cell::sub_entity_count(celltype, sub_entity_dim);
  const std::size_t tdim = cell::topological_dimension(celltype);

  auto [qpts, _qwts]
      = quadrature::make_quadrature_new("default", sub_celltype, q_deg);
  auto qwts = xt::adapt(_qwts);
  if (qpts.dimension() == 1)
    qpts = qpts.reshape({qpts.shape()[0], 1});

  // If this is always true, value_size input can be removed
  assert(tdim == value_size);

  // TMP: Copy into Eigen array
  Eigen::ArrayXXd _qpts(qpts.shape()[0], qpts.shape()[1]);
  for (std::size_t i = 0; i < qpts.shape()[0]; ++i)
    for (std::size_t j = 0; j < qpts.shape()[1]; ++j)
      _qpts(i, j) = qpts(i, j);

  // Evaluate moment space at quadrature points
  Eigen::ArrayXXd _moment_space_at_Qpts = moment_space.tabulate(0, _qpts)[0];
  std::array<std::size_t, 2> shape1
      = {(std::size_t)_moment_space_at_Qpts.rows(),
         (std::size_t)_moment_space_at_Qpts.cols()};
  auto moment_space_at_Qpts = xt::adapt<xt::layout_type::column_major>(
      _moment_space_at_Qpts.data(), _moment_space_at_Qpts.size(),
      xt::no_ownership(), shape1);

  const std::size_t moment_space_size
      = moment_space_at_Qpts.shape()[1] / sub_entity_dim;

  // Eigen::ArrayXXd points(sub_entity_count * Qpts.rows(), tdim);
  xt::xtensor<double, 2> points({sub_entity_count * qpts.shape()[0], tdim});
  const std::array<std::size_t, 2> shape
      = {moment_space_size * sub_entity_count,
         sub_entity_count * qpts.shape()[0] * value_size};
  xt::xtensor<double, 2> matrix = xt::zeros<double>(shape);

  // Iterate over sub entities
  int c = 0;
  std::vector<int> axis_pts = axis_points(celltype);
  const std::size_t num_points = qpts.shape()[0];
  for (std::size_t i = 0; i < sub_entity_count; ++i)
  {
    xt::xtensor<double, 2> entity
        = cell::sub_entity_geometry(celltype, sub_entity_dim, i);
    Eigen::Map<
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        _entity(entity.data(), entity.shape()[0], entity.shape()[1]);

    // Parameterise entity coordinates
    xt::xtensor<double, 2> axes({sub_entity_dim, tdim});
    for (std::size_t j = 0; j < sub_entity_dim; ++j)
      xt::row(axes, j) = xt::row(entity, axis_pts[j]) - xt::row(entity, 0);

    // See
    // https://github.com/xtensor-stack/xtensor/issues/1922#issuecomment-586317746
    // for why xt::newaxis() is required
    auto points_view
        = xt::view(points, xt::range(i * num_points, (i + 1) * num_points),
                   xt::range(0, tdim));
    auto p = xt::tile(xt::view(entity, xt::newaxis(), 0), num_points);
    points_view = p + xt::linalg::dot(qpts, axes);

    // Compute entity integral moments
    for (std::size_t j = 0; j < moment_space_size; ++j)
    {
      for (std::size_t k = 0; k < value_size; ++k)
      {
        auto matrix_view
            = xt::view(matrix, c,
                       xt::range((k * sub_entity_count + i) * num_points,
                                 (k * sub_entity_count + i + 1) * num_points));
        xt::xtensor<double, 1> q = xt::zeros<double>({num_points});
        for (std::size_t d = 0; d < sub_entity_dim; ++d)
        {
          // FIXME: This assumed that the moment space has a certain mapping
          // type
          auto phi = xt::col(moment_space_at_Qpts, d * moment_space_size + j);
          matrix_view += phi * qwts * axes(d, k);
        }
      }
      ++c;
    }
  }

  return {points, matrix};
}
//----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
moments::make_tangent_integral_moments(const FiniteElement& moment_space,
                                       cell::type celltype, int value_size,
                                       int q_deg)
{
  auto [points, matrix] = make_tangent_integral_moments_new(
      moment_space, celltype, value_size, q_deg);

  // TMP: Copy into Eigen
  Eigen::ArrayXXd _points(points.shape()[0], points.shape()[1]);
  Eigen::MatrixXd _matrix(matrix.shape()[0], matrix.shape()[1]);
  for (std::size_t i = 0; i < points.shape()[0]; ++i)
    for (std::size_t j = 0; j < points.shape()[1]; ++j)
      _points(i, j) = points(i, j);
  for (std::size_t i = 0; i < matrix.shape()[0]; ++i)
    for (std::size_t j = 0; j < matrix.shape()[1]; ++j)
      _matrix(i, j) = matrix(i, j);

  return std::make_pair(_points, _matrix);
}
//----------------------------------------------------------------------------
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
moments::make_tangent_integral_moments_new(const FiniteElement& moment_space,
                                           cell::type celltype,
                                           std::size_t value_size, int q_deg)
{
  const cell::type sub_celltype = moment_space.cell_type();
  const std::size_t sub_entity_dim = cell::topological_dimension(sub_celltype);
  const std::size_t sub_entity_count
      = cell::sub_entity_count(celltype, sub_entity_dim);
  const std::size_t tdim = cell::topological_dimension(celltype);

  if (sub_entity_dim != 1)
    throw std::runtime_error("Tangent is only well-defined on an edge.");

  auto [Qpts, _Qwts]
      = quadrature::make_quadrature_new("default", cell::type::interval, q_deg);
  auto Qwts = xt::adapt(_Qwts);
  if (Qpts.dimension() == 1)
    Qpts = Qpts.reshape({Qpts.shape()[0], 1});

  // TMP: Copy into Eigen array
  Eigen::ArrayXXd _Qpts(Qpts.shape()[0], Qpts.shape()[1]);
  for (std::size_t i = 0; i < Qpts.shape()[0]; ++i)
    for (std::size_t j = 0; j < Qpts.shape()[1]; ++j)
      _Qpts(i, j) = Qpts(i, j);

  // If this is always true, value_size input can be removed
  assert(tdim == value_size);

  // Evaluate moment space at quadrature points
  Eigen::ArrayXXd _moment_space_at_Qpts = moment_space.tabulate(0, _Qpts)[0];
  std::array<std::size_t, 2> shape1
      = {(std::size_t)_moment_space_at_Qpts.rows(),
         (std::size_t)_moment_space_at_Qpts.cols()};
  auto moment_space_at_Qpts = xt::adapt<xt::layout_type::column_major>(
      _moment_space_at_Qpts.data(), _moment_space_at_Qpts.size(),
      xt::no_ownership(), shape1);

  xt::xtensor<double, 2> points({sub_entity_count * Qpts.shape()[0], tdim});
  const std::array<std::size_t, 2> shape
      = {moment_space_at_Qpts.shape()[1] * sub_entity_count,
         sub_entity_count * Qpts.shape()[0] * value_size};
  xt::xtensor<double, 2> matrix = xt::zeros<double>(shape);
  // Eigen::ArrayXXd points(sub_entity_count * Qpts.rows(), tdim);
  // Eigen::MatrixXd matrix(moment_space_at_Qpts.cols() * sub_entity_count,
  //                        sub_entity_count * Qpts.rows() * value_size);
  // matrix.setZero();

  // Iterate over sub entities
  int c = 0;
  for (std::size_t i = 0; i < sub_entity_count; ++i)
  {
    xt::xtensor<double, 2> edge = cell::sub_entity_geometry(celltype, 1, i);
    Eigen::Map<
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        _edge(edge.data(), edge.shape()[0], edge.shape()[1]);

    auto tangent = xt::row(edge, 1) - xt::row(edge, 0);
    // Eigen::VectorXd tangent = _edge.row(1) - _edge.row(0);

    // No need to normalise the tangent, as the size of this is equal to the
    // integral jacobian

    // Map quadrature points onto triangle edge
    // Eigen::ArrayXXd Qpts_scaled(Qpts.rows(), tdim);
    for (std::size_t j = 0; j < Qpts.shape()[0]; ++j)
    {
      xt::row(points, i * Qpts.shape()[0] + j)
          = xt::row(edge, 0) + Qpts(j, 0) * tangent;
    }

    // Compute edge tangent integral moments
    for (std::size_t j = 0; j < moment_space_at_Qpts.shape()[1]; ++j)
    {
      // Eigen::ArrayXd phi = moment_space_at_Qpts.col(j);
      auto phi = xt::col(moment_space_at_Qpts, j);
      for (std::size_t k = 0; k < value_size; ++k)
      {
        // Eigen::RowVectorXd data = phi * Qwts * tangent[k];
        // auto data = phi * Qwts * tangent[k];
        // matrix.block(c, k * sub_entity_count * Qpts.rows() + i * Qpts.rows(),
        // 1,
        //              Qpts.rows())
        //     = data;
        std::size_t offset
            = k * sub_entity_count * Qpts.shape()[0] + i * Qpts.shape()[0];
        xt::view(matrix, c, xt::range(offset, offset + Qpts.shape()[0]))
            = phi * Qwts * tangent[k];
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
      // No need to normalise the normal, as the size of this is equal to
      // the integral jacobian

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

      // No need to normalise the normal, as the size of this is equal to
      // the integral jacobian

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
