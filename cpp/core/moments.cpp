// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "moments.h"
#include "cell.h"
#include "finite-element.h"
#include "polyset.h"
#include "quadrature.h"

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

//----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
moments::make_integral_moments(const FiniteElement& moment_space,
                               const cell::type celltype, const int value_size,
                               const int poly_deg, const int q_deg)
{
  const cell::type sub_celltype = moment_space.cell_type();
  const int sub_entity_dim = cell::topological_dimension(sub_celltype);
  if (sub_entity_dim == 0)
    throw std::runtime_error("Cannot integrate over a dimension 0 entity.");
  const int sub_entity_count = cell::sub_entity_count(celltype, sub_entity_dim);
  const int tdim = cell::topological_dimension(celltype);

  auto [Qpts, Qwts]
      = quadrature::make_quadrature("default", sub_celltype, q_deg);

  // If this is always true, value_size input can be removed
  assert(tdim == value_size);

  // Evaluate moment space at quadrature points
  Eigen::ArrayXXd moment_space_at_Qpts = moment_space.tabulate(0, Qpts)[0];

  Eigen::ArrayXXd points(sub_entity_count * Qpts.rows(), tdim);
  Eigen::MatrixXd matrix(moment_space_at_Qpts.cols() * sub_entity_count
                             * sub_entity_dim,
                         sub_entity_count * Qpts.rows() * value_size);
  matrix.setZero();

  std::vector<int> axis_pts = axis_points(celltype);

  int c = 0;
  // Iterate over sub entities
  for (int i = 0; i < sub_entity_count; ++i)
  {
    ndarray<double, 2> entity
        = cell::sub_entity_geometry(celltype, sub_entity_dim, i);
    Eigen::Map<
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        _entity(entity.data(), entity.shape[0], entity.shape[1]);

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

  return std::make_pair(points, matrix);
}
//----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd> moments::make_dot_integral_moments(
    const FiniteElement& moment_space, const cell::type celltype,
    const int value_size, const int poly_deg, const int q_deg)
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
  Eigen::MatrixXd matrix(moment_space_size * sub_entity_count,
                         sub_entity_count * Qpts.rows() * value_size);
  matrix.setZero();

  std::vector<int> axis_pts = axis_points(celltype);

  int c = 0;
  // Iterate over sub entities
  for (int i = 0; i < sub_entity_count; ++i)
  {
    ndarray<double, 2> entity
        = cell::sub_entity_geometry(celltype, sub_entity_dim, i);
    Eigen::Map<
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        _entity(entity.data(), entity.shape[0], entity.shape[1]);

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
                                       const cell::type celltype,
                                       const int value_size, const int poly_deg,
                                       const int q_deg)
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
    ndarray<double, 2> edge = cell::sub_entity_geometry(celltype, 1, i);
    Eigen::Map<
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        _edge(edge.data(), edge.shape[0], edge.shape[1]);

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
                                      const int value_size, const int poly_deg,
                                      const int q_deg)
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
    ndarray<double, 2> facet = cell::sub_entity_geometry(celltype, tdim - 1, i);
    Eigen::Map<
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        _facet(facet.data(), facet.shape[0], facet.shape[1]);

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

    // Tabulate polynomial set at facet quadrature points
    Eigen::MatrixXd poly_set_at_Qpts
        = polyset::tabulate(celltype, poly_deg, 0, Qpts_scaled)[0].transpose();

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
