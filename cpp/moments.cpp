// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "moments.h"
#include "cell.h"
#include "finite-element.h"
#include "polyset.h"
#include "quadrature.h"

using namespace libtab;

namespace
{
//----------------------------------------------------------------------------
double integral_jacobian(const Eigen::MatrixXd& axes)
{
  if (axes.rows() == 1)
    return axes.row(0).norm();
  else if (axes.rows() == 2 and axes.cols() == 3)
  {
    Eigen::Vector3d a0 = axes.row(0);
    Eigen::Vector3d a1 = axes.row(1);
    return a0.cross(a1).norm();
  }
  else
    return axes.determinant();
}
//----------------------------------------------------------------------------
} // namespace

//----------------------------------------------------------------------------
Eigen::MatrixXd
moments::make_integral_moments(const FiniteElement& moment_space,
                               const cell::type celltype, const int value_size,
                               const int poly_deg, const int q_deg)
{
  const int psize = polyset::dim(celltype, poly_deg);

  const cell::type sub_celltype = moment_space.cell_type();
  const int sub_entity_dim = cell::topological_dimension(sub_celltype);
  if (sub_entity_dim == 0)
    throw std::runtime_error("Cannot integrate over a dimension 0 entity.");

  const int sub_entity_count = cell::sub_entity_count(celltype, sub_entity_dim);

  auto [Qpts, Qwts] = quadrature::make_quadrature(sub_celltype, q_deg);
  const int tdim = cell::topological_dimension(celltype);

  // If this is always true, value_size input can be removed
  assert(tdim == value_size);

  // Evaluate moment space at quadrature points
  Eigen::ArrayXXd moment_space_at_Qpts = moment_space.tabulate(0, Qpts)[0];

  Eigen::MatrixXd dual(moment_space_at_Qpts.cols() * sub_entity_dim
                           * sub_entity_count,
                       psize * value_size);

  int c = 0;
  // Iterate over sub entities
  for (int i = 0; i < sub_entity_count; ++i)
  {
    Eigen::ArrayXXd entity
        = cell::sub_entity_geometry(celltype, sub_entity_dim, i);

    // Parametrise entity coordinates
    Eigen::ArrayXXd axes(sub_entity_dim, tdim);
    for (int j = 0; j < sub_entity_dim; ++j)
      axes.row(j) = entity.row(j + 1) - entity.row(0);

    // Map quadrature points onto entity
    Eigen::ArrayXXd Qpts_scaled = entity.row(0).replicate(Qpts.rows(), 1)
                                  + (Qpts.matrix() * axes.matrix()).array();

    const double integral_jac = integral_jacobian(axes);

    // Tabulate polynomial set at entity quadrature points
    Eigen::MatrixXd poly_set_at_Qpts
        = polyset::tabulate(celltype, poly_deg, 0, Qpts_scaled)[0].transpose();

    // Compute entity integral moments
    for (int j = 0; j < moment_space_at_Qpts.cols(); ++j)
    {
      Eigen::ArrayXd phi = moment_space_at_Qpts.col(j);
      for (int d = 0; d < sub_entity_dim; ++d)
      {
        Eigen::VectorXd axis = axes.row(d);
        for (int k = 0; k < value_size; ++k)
        {
          Eigen::VectorXd q
              = phi * Qwts * (integral_jac * axis(k) / axis.norm());
          Eigen::RowVectorXd qcoeffs = poly_set_at_Qpts * q;
          assert(qcoeffs.size() == psize);
          dual.block(c, psize * k, 1, psize) = qcoeffs;
        }
        ++c;
      }
    }
  }

  return dual;
}
//----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
moments::make_integral_moments_interpolation(const FiniteElement& moment_space,
                                             const cell::type celltype,
                                             const int value_size,
                                             const int poly_deg,
                                             const int q_deg)
{
  const cell::type sub_celltype = moment_space.cell_type();
  const int sub_entity_dim = cell::topological_dimension(sub_celltype);
  const int sub_entity_count = cell::sub_entity_count(celltype, sub_entity_dim);
  const int tdim = cell::topological_dimension(celltype);

  auto [Qpts, Qwts] = quadrature::make_quadrature(sub_celltype, q_deg);

  // If this is always true, value_size input can be removed
  assert(tdim == value_size);

  // Evaluate moment space at quadrature points
  Eigen::ArrayXXd moment_space_at_Qpts = moment_space.tabulate(0, Qpts)[0];

  Eigen::ArrayXXd points(sub_entity_count * Qpts.rows(), tdim);
  Eigen::MatrixXd matrix(moment_space_at_Qpts.cols() * sub_entity_count
                             * sub_entity_dim,
                         sub_entity_count * Qpts.rows() * value_size);
  matrix.setZero();

  int c = 0;

  // Iterate over sub entities
  for (int i = 0; i < sub_entity_count; ++i)
  {
    Eigen::ArrayXXd entity
        = cell::sub_entity_geometry(celltype, sub_entity_dim, i);

    // Parametrise entity coordinates
    Eigen::ArrayXXd axes(sub_entity_dim, tdim);
    for (int j = 0; j < sub_entity_dim; ++j)
      axes.row(j) = entity.row(j + 1) - entity.row(0);

    // Map quadrature points onto entity
    Eigen::ArrayXXd Qpts_scaled = entity.row(0).replicate(Qpts.rows(), 1)
                                  + (Qpts.matrix() * axes.matrix()).array();
    points.block(Qpts.rows() * i, 0, Qpts.rows(), tdim)
        = entity.row(0).replicate(Qpts.rows(), 1)
          + (Qpts.matrix() * axes.matrix()).array();

    const double integral_jac = integral_jacobian(axes);

    // Compute entity integral moments
    for (int j = 0; j < moment_space_at_Qpts.cols(); ++j)
    {
      Eigen::ArrayXd phi = moment_space_at_Qpts.col(j);
      for (int d = 0; d < sub_entity_dim; ++d)
      {
        Eigen::VectorXd axis = axes.row(d);
        for (int k = 0; k < value_size; ++k)
        {
          Eigen::VectorXd q
              = phi * Qwts * (integral_jac * axis(k) / axis.norm());
          for (int l = 0; l < Qpts.rows(); ++l)
            matrix(c, (i * value_size + k) * Qpts.rows() + l) = q(l);
        }
        ++c;
      }
    }
  }

  return std::make_pair(points, matrix);
}
//----------------------------------------------------------------------------
Eigen::MatrixXd moments::make_dot_integral_moments(
    const FiniteElement& moment_space, const cell::type celltype,
    const int value_size, const int poly_deg, const int q_deg)
{
  const int psize = polyset::dim(celltype, poly_deg);

  const cell::type sub_celltype = moment_space.cell_type();
  const int sub_entity_dim = cell::topological_dimension(sub_celltype);
  if (sub_entity_dim == 0)
    throw std::runtime_error("Cannot integrate over a dimension 0 entity.");

  const int sub_entity_count = cell::sub_entity_count(celltype, sub_entity_dim);

  auto [Qpts, Qwts] = quadrature::make_quadrature(sub_celltype, q_deg);
  const int tdim = cell::topological_dimension(celltype);

  // If this is always true, value_size input can be removed
  assert(tdim == value_size);

  // Evaluate moment space at quadrature points
  Eigen::ArrayXXd moment_space_at_Qpts = moment_space.tabulate(0, Qpts)[0];

  const int moment_space_size = moment_space_at_Qpts.cols() / sub_entity_dim;
  Eigen::MatrixXd dual(moment_space_size * sub_entity_count,
                       psize * value_size);

  int c = 0;
  // Iterate over sub entities
  for (int i = 0; i < sub_entity_count; ++i)
  {
    Eigen::ArrayXXd entity
        = cell::sub_entity_geometry(celltype, sub_entity_dim, i);

    // Parametrise entity coordinates
    Eigen::ArrayXXd axes(sub_entity_dim, tdim);
    for (int j = 0; j < sub_entity_dim; ++j)
      axes.row(j) = entity.row(j + 1) - entity.row(0);

    // Map quadrature points onto entity
    Eigen::ArrayXXd Qpts_scaled = entity.row(0).replicate(Qpts.rows(), 1)
                                  + (Qpts.matrix() * axes.matrix()).array();

    const double integral_jac = integral_jacobian(axes);

    // Tabulate polynomial set at entity quadrature points
    Eigen::MatrixXd poly_set_at_Qpts
        = polyset::tabulate(celltype, poly_deg, 0, Qpts_scaled)[0].transpose();

    // Compute entity integral moments
    for (int j = 0; j < moment_space_size; ++j)
    {
      for (int k = 0; k < value_size; ++k)
      {
        Eigen::VectorXd q = Eigen::VectorXd::Zero(Qwts.rows());
        for (int d = 0; d < sub_entity_dim; ++d)
        {
          Eigen::ArrayXd phi
              = moment_space_at_Qpts.col(d * moment_space_size + j);
          Eigen::VectorXd axis = axes.row(d);
          Eigen::VectorXd qpart
              = phi * Qwts * (integral_jac * axis(k) / axis.norm());
          q += qpart;
        }
        Eigen::RowVectorXd qcoeffs = poly_set_at_Qpts * q;
        assert(qcoeffs.size() == psize);
        dual.block(c, psize * k, 1, psize) = qcoeffs;
      }
      ++c;
    }
  }

  return dual;
}
//----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
moments::make_dot_integral_moments_interpolation(
    const FiniteElement& moment_space, const cell::type celltype,
    const int value_size, const int poly_deg, const int q_deg)
{
  const cell::type sub_celltype = moment_space.cell_type();
  const int sub_entity_dim = cell::topological_dimension(sub_celltype);
  const int sub_entity_count = cell::sub_entity_count(celltype, sub_entity_dim);
  const int tdim = cell::topological_dimension(celltype);

  auto [Qpts, Qwts] = quadrature::make_quadrature(celltype, q_deg);

  // If this is always true, value_size input can be removed
  assert(tdim == value_size);

  // Evaluate moment space at quadrature points
  Eigen::ArrayXXd moment_space_at_Qpts = moment_space.tabulate(0, Qpts)[0];
  const int moment_space_size = moment_space_at_Qpts.cols() / sub_entity_dim;

  Eigen::ArrayXXd points(sub_entity_count * Qpts.rows(), tdim);
  Eigen::MatrixXd matrix(moment_space_size * sub_entity_count,
                         sub_entity_count * Qpts.rows() * value_size);
  matrix.setZero();

  int c = 0;

  // Iterate over sub entities
  for (int i = 0; i < sub_entity_count; ++i)
  {
    Eigen::ArrayXXd entity
        = cell::sub_entity_geometry(celltype, sub_entity_dim, i);

    // Parametrise entity coordinates
    Eigen::ArrayXXd axes(sub_entity_dim, tdim);
    for (int j = 0; j < sub_entity_dim; ++j)
      axes.row(j) = entity.row(j + 1) - entity.row(0);

    // Map quadrature points onto entity
    points.block(Qpts.rows() * i, 0, Qpts.rows(), tdim)
        = entity.row(0).replicate(Qpts.rows(), 1)
          + (Qpts.matrix() * axes.matrix()).array();

    const double integral_jac = integral_jacobian(axes);

    // Compute entity integral moments
    for (int j = 0; j < moment_space_size; ++j)
    {
      for (int k = 0; k < value_size; ++k)
      {
        Eigen::VectorXd q = Eigen::VectorXd::Zero(Qwts.rows());
        for (int d = 0; d < sub_entity_dim; ++d)
        {
          Eigen::ArrayXd phi
              = moment_space_at_Qpts.col(d * moment_space_size + j);
          Eigen::VectorXd axis = axes.row(d);
          Eigen::VectorXd qpart
              = phi * Qwts * (integral_jac * axis(k) / axis.norm());
          q += qpart;
        }
        matrix.block(c, (i * value_size + k) * Qpts.rows(), 1, Qpts.rows()) = q;
      }
      ++c;
    }
  }

  return std::make_pair(points, matrix);
}
//----------------------------------------------------------------------------
Eigen::MatrixXd moments::make_tangent_integral_moments(
    const FiniteElement& moment_space, const cell::type celltype,
    const int value_size, const int poly_deg, const int q_deg)
{
  const int psize = polyset::dim(celltype, poly_deg);
  const cell::type sub_celltype = moment_space.cell_type();
  const int sub_entity_dim = cell::topological_dimension(sub_celltype);
  const int sub_entity_count = cell::sub_entity_count(celltype, sub_entity_dim);
  const int tdim = cell::topological_dimension(celltype);

  if (sub_entity_dim != 1)
    throw std::runtime_error("Tangent is only well-defined on an edge.");

  auto [Qpts, Qwts] = quadrature::make_quadrature(cell::type::interval, q_deg);

  // If this is always true, value_size input can be removed
  assert(tdim == value_size);

  // Evaluate moment space at quadrature points
  Eigen::ArrayXXd moment_space_at_Qpts = moment_space.tabulate(0, Qpts)[0];

  Eigen::MatrixXd dual(moment_space_at_Qpts.cols() * sub_entity_count,
                       psize * value_size);

  int c = 0;

  // Iterate over sub entities
  for (int i = 0; i < sub_entity_count; ++i)
  {
    Eigen::Array2Xd edge = cell::sub_entity_geometry(celltype, 1, i);
    Eigen::VectorXd tangent = edge.row(1) - edge.row(0);
    // No need to normalise the tangent, as the size of this is equal to the
    // integral jacobian

    // Map quadrature points onto triangle edge
    Eigen::ArrayXXd Qpts_scaled(Qpts.rows(), tdim);
    for (int j = 0; j < Qpts.rows(); ++j)
    {
      Qpts_scaled.row(j)
          = edge.row(0) + Qpts(j, 0) * (edge.row(1) - edge.row(0));
    }

    // Tabulate polynomial set at edge quadrature points
    Eigen::MatrixXd poly_set_at_Qpts
        = polyset::tabulate(celltype, poly_deg, 0, Qpts_scaled)[0].transpose();

    // Compute edge tangent integral moments
    for (int j = 0; j < moment_space_at_Qpts.cols(); ++j)
    {
      Eigen::ArrayXd phi = moment_space_at_Qpts.col(j);
      for (int k = 0; k < value_size; ++k)
      {
        Eigen::VectorXd q = phi * Qwts * tangent[k];
        Eigen::RowVectorXd qcoeffs = poly_set_at_Qpts * q;
        dual.block(c, psize * k, 1, psize) = qcoeffs;
      }
      ++c;
    }
  }

  return dual;
}
//----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
moments::make_tangent_integral_moments_interpolation(
    const FiniteElement& moment_space, const cell::type celltype,
    const int value_size, const int poly_deg, const int q_deg)
{
  const cell::type sub_celltype = moment_space.cell_type();
  const int sub_entity_dim = cell::topological_dimension(sub_celltype);
  const int sub_entity_count = cell::sub_entity_count(celltype, sub_entity_dim);
  const int tdim = cell::topological_dimension(celltype);

  if (sub_entity_dim != 1)
    throw std::runtime_error("Tangent is only well-defined on an edge.");

  auto [Qpts, Qwts] = quadrature::make_quadrature(cell::type::interval, q_deg);

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
    Eigen::Array2Xd edge = cell::sub_entity_geometry(celltype, 1, i);
    Eigen::VectorXd tangent = edge.row(1) - edge.row(0);
    // No need to normalise the tangent, as the size of this is equal to the
    // integral jacobian

    // Map quadrature points onto triangle edge
    Eigen::ArrayXXd Qpts_scaled(Qpts.rows(), tdim);
    for (int j = 0; j < Qpts.rows(); ++j)
    {
      points.row(i * Qpts.rows() + j)
          = edge.row(0) + Qpts(j, 0) * (edge.row(1) - edge.row(0));
    }

    // Compute edge tangent integral moments
    for (int j = 0; j < moment_space_at_Qpts.cols(); ++j)
    {
      Eigen::ArrayXd phi = moment_space_at_Qpts.col(j);
      for (int k = 0; k < value_size; ++k)
      {
        Eigen::ArrayXd data = phi * Qwts * tangent[k];
        for (int l = 0; l < data.rows(); ++l)
          // matrix(c, i * value_size * Qpts.rows() + l * value_size + k) =
          // data(l);
          matrix(c, k * sub_entity_count * Qpts.rows() + i * Qpts.rows() + l)
              = data(l);
      }
      ++c;
    }
  }

  return std::make_pair(points, matrix);
}
//----------------------------------------------------------------------------
Eigen::MatrixXd moments::make_normal_integral_moments(
    const FiniteElement& moment_space, const cell::type celltype,
    const int value_size, const int poly_deg, const int q_deg)
{
  const int psize = polyset::dim(celltype, poly_deg);
  const cell::type sub_celltype = moment_space.cell_type();
  const int sub_entity_dim = cell::topological_dimension(sub_celltype);
  const int sub_entity_count = cell::sub_entity_count(celltype, sub_entity_dim);
  const int tdim = cell::topological_dimension(celltype);

  if (sub_entity_dim != tdim - 1)
    throw std::runtime_error("Normal is only well-defined on a facet.");

  auto [Qpts, Qwts] = quadrature::make_quadrature(sub_celltype, q_deg);

  // If this is always true, value_size input can be removed
  assert(tdim == value_size);

  // Evaluate moment space at quadrature points
  Eigen::ArrayXXd moment_space_at_Qpts = moment_space.tabulate(0, Qpts)[0];

  Eigen::MatrixXd dual(moment_space_at_Qpts.cols() * sub_entity_count,
                       psize * value_size);

  int c = 0;

  // Iterate over sub entities
  Eigen::VectorXd normal(tdim);
  Eigen::ArrayXXd Qpts_scaled(Qpts.rows(), tdim);
  for (int i = 0; i < sub_entity_count; ++i)
  {
    Eigen::ArrayXXd facet = cell::sub_entity_geometry(celltype, tdim - 1, i);
    if (tdim == 2)
    {
      Eigen::Vector2d tangent = facet.row(1) - facet.row(0);
      normal << -tangent(1), tangent(0);
      // No need to normalise the normal, as the size of this is equal to the
      // integral jacobian

      // Map quadrature points onto facet
      for (int j = 0; j < Qpts.rows(); ++j)
      {
        Qpts_scaled.row(j)
            = facet.row(0) + Qpts(j, 0) * (facet.row(1) - facet.row(0));
      }
    }
    else if (tdim == 3)
    {
      Eigen::Vector3d t0 = facet.row(1) - facet.row(0);
      Eigen::Vector3d t1 = facet.row(2) - facet.row(0);
      normal = t0.cross(t1);

      // No need to normalise the normal, as the size of this is equal to the
      // integral jacobian

      // Map quadrature points onto facet
      for (int j = 0; j < Qpts.rows(); ++j)
      {
        Qpts_scaled.row(j) = facet.row(0)
                             + Qpts(j, 0) * (facet.row(1) - facet.row(0))
                             + Qpts(j, 1) * (facet.row(2) - facet.row(0));
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
        Eigen::VectorXd q = phi * Qwts * normal[k];
        Eigen::RowVectorXd qcoeffs = poly_set_at_Qpts * q;
        dual.block(c, psize * k, 1, psize) = qcoeffs;
      }
      ++c;
    }
  }

  return dual;
}
//----------------------------------------------------------------------------
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
moments::make_normal_integral_moments_interpolation(
    const FiniteElement& moment_space, const cell::type celltype,
    const int value_size, const int poly_deg, const int q_deg)
{
  const cell::type sub_celltype = moment_space.cell_type();
  const int sub_entity_dim = cell::topological_dimension(sub_celltype);
  const int sub_entity_count = cell::sub_entity_count(celltype, sub_entity_dim);
  const int tdim = cell::topological_dimension(celltype);

  if (sub_entity_dim != tdim - 1)
    throw std::runtime_error("Normal is only well-defined on a facet.");

  auto [Qpts, Qwts] = quadrature::make_quadrature(sub_celltype, q_deg);

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
    Eigen::ArrayXXd facet = cell::sub_entity_geometry(celltype, tdim - 1, i);
    if (tdim == 2)
    {
      Eigen::Vector2d tangent = facet.row(1) - facet.row(0);
      normal << -tangent(1), tangent(0);
      // No need to normalise the normal, as the size of this is equal to the
      // integral jacobian

      // Map quadrature points onto facet
      for (int j = 0; j < Qpts.rows(); ++j)
      {
        points.row(i * Qpts.rows() + j)
            = facet.row(0) + Qpts(j, 0) * (facet.row(1) - facet.row(0));
      }
    }
    else if (tdim == 3)
    {
      Eigen::Vector3d t0 = facet.row(1) - facet.row(0);
      Eigen::Vector3d t1 = facet.row(2) - facet.row(0);
      normal = t0.cross(t1);

      // No need to normalise the normal, as the size of this is equal to the
      // integral jacobian

      // Map quadrature points onto facet
      for (int j = 0; j < Qpts.rows(); ++j)
      {
        points.row(i * Qpts.rows() + j)
            = facet.row(0) + Qpts(j, 0) * (facet.row(1) - facet.row(0))
              + Qpts(j, 1) * (facet.row(2) - facet.row(0));
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
        Eigen::VectorXd q = phi * Qwts * normal[k];
        for (int l = 0; l < q.rows(); ++l)
          matrix(c, k * sub_entity_count * Qpts.rows() + i * Qpts.rows() + l)
              = q(l);
      }
      ++c;
    }
  }

  return std::make_pair(points, matrix);
}
//----------------------------------------------------------------------------
