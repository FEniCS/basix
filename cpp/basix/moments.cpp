// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "moments.h"
#include "cell.h"
#include "finite-element.h"
#include "polyset.h"
#include "quadrature.h"
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

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
xt::xtensor<double, 3> moments::create_dot_moment_dof_transformations(
    const FiniteElement& moment_space)
{
  // This function can be dramatically simplified and made
  // understandable by using tensor to give more logic to the objects

  cell::type celltype = moment_space.cell_type();
  if (celltype == cell::type::point)
    return {};

  xt::xarray<double> pts = moment_space.points();
  if (pts.shape()[1] == 1)
    pts.reshape({pts.shape()[0]});

  const xt::xtensor<double, 2>& matrix = moment_space.interpolation_matrix();
  xt::xtensor<double, 3> tpts;
  xt::xtensor<double, 3> J;
  xt::xtensor<double, 3> K;
  xt::xtensor<double, 2> detJ;
  switch (celltype)
  {
  case cell::type::interval:
  {

    tpts = xt::atleast_3d(1.0 - pts);
    J = xt::full_like(tpts, -1.0);
    K = xt::full_like(tpts, -1.0);
    detJ = xt::atleast_2d(xt::ones_like(pts));
    break;
  }
  case cell::type::triangle:
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

    break;
  }
  case cell::type::quadrilateral:
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
    xt::view(J0, xt::all(), 0) = 0;
    xt::view(J0, xt::all(), 1) = 1;
    xt::view(J0, xt::all(), 2) = -1;
    xt::view(J0, xt::all(), 3) = 0;

    auto K0 = xt::view(K, 0, xt::all(), xt::all());
    xt::view(K0, xt::all(), 0) = 0;
    xt::view(K0, xt::all(), 1) = -1;
    xt::view(K0, xt::all(), 2) = 1;
    xt::view(K0, xt::all(), 3) = 0;

    // Case 1
    xt::view(detJ, 1, xt::all()) = 1.0;
    for (std::size_t i = 0; i < pts.shape()[0]; ++i)
    {
      tpts(1, i, 0) = pts(i, 1);
      tpts(1, i, 1) = pts(i, 0);
    }

    auto J1 = xt::view(J, 1, xt::all(), xt::all());
    xt::view(J1, xt::all(), 0) = 0;
    xt::view(J1, xt::all(), 1) = 1;
    xt::view(J1, xt::all(), 2) = 1;
    xt::view(J1, xt::all(), 3) = 0;

    auto K1 = xt::view(K, 1, xt::all(), xt::all());
    xt::view(K1, xt::all(), 0) = 0;
    xt::view(K1, xt::all(), 1) = 1;
    xt::view(K1, xt::all(), 2) = 1;
    xt::view(K1, xt::all(), 3) = 0;

    break;
  }
  default:
  {
    throw std::runtime_error(
        "DOF transformations only implemented for tdim <= 2.");
  }
  }

  std::array<std::size_t, 3> shape
      = {tpts.shape()[0], (std::size_t)moment_space.dim(),
         (std::size_t)moment_space.dim()};
  xt::xtensor<double, 3> out = xt::zeros<double>(shape);
  for (std::size_t i = 0; i < tpts.shape()[0]; ++i)
  {
    std::vector<double> _detJ(detJ.shape()[1]);
    for (std::size_t j = 0; j < J.shape()[1]; ++j)
      _detJ[j] = detJ(i, j);

    auto _tpoint = xt::view(tpts, i, xt::all(), xt::all());
    xt::xtensor<double, 3> moment_space_pts
        = xt::view(moment_space.tabulate_x(0, _tpoint), 0, xt::all(), xt::all(),
                   xt::all());

    xt::xtensor<double, 3> Ji(
        {moment_space_pts.shape(0), _tpoint.shape(1), _tpoint.shape(1)});
    xt::xtensor<double, 3> Ki(
        {moment_space_pts.shape(0), _tpoint.shape(1), _tpoint.shape(1)});
    for (std::size_t j = 0; j < moment_space_pts.shape(0); ++j)
    {
      for (std::size_t k = 0; k < _tpoint.shape(1); ++k)
      {
        for (std::size_t l = 0; l < _tpoint.shape(1); ++l)
        {
          Ji(j, k, l) = J(i, j, k * _tpoint.shape(1) + l);
          Ki(j, k, l) = K(i, j, k * _tpoint.shape(1) + l);
        }
      }
    }

    xt::xtensor<double, 3> F
        = moment_space.map_pull_back(moment_space_pts, Ji, _detJ, Ki);

    // Copy onto 2D array
    xt::xtensor<double, 2> _pulled({F.shape(0), F.shape(1) * F.shape(2)});
    for (std::size_t p = 0; p < F.shape(0); ++p)
    {
      {
        for (std::size_t i = 0; i < F.shape(1); ++i)
          for (std::size_t j = 0; j < F.shape(2); ++j)
            _pulled(p, j * F.shape(1) + i) = F(p, i, j);
      }
    }

    xt::xtensor<double, 2> tmp0, tmp1;
    for (int v = 0; v < moment_space.value_size(); ++v)
    {
      tmp0 = xt::view(
          matrix, xt::range(0, matrix.shape()[0]),
          xt::range(v * _pulled.shape(0), (v + 1) * _pulled.shape(0)));
      tmp1 = xt::view(
          _pulled, xt::range(0, _pulled.shape(0)),
          xt::range(moment_space.dim() * v, moment_space.dim() * (v + 1)));
      xt::view(out, i, xt::all(), xt::all()) += xt::linalg::dot(tmp0, tmp1);
    }
  }

  return out;
}
//----------------------------------------------------------------------------
xt::xtensor<double, 3>
moments::create_moment_dof_transformations(const FiniteElement& moment_space)
{
  const xt::xtensor<double, 3> t
      = create_dot_moment_dof_transformations(moment_space);

  xt::xtensor_fixed<double, xt::xshape<2, 2>> rot, ref;

  cell::type celltype = moment_space.cell_type();
  switch (celltype)
  {
  case cell::type::interval:
    return t;
  case cell::type::triangle:
    rot = {{-1, -1}, {1, 0}};
    ref = {{0, 1}, {1, 0}};
    break;
  case cell::type::quadrilateral:
    // TODO: check that these are correct
    rot = {{0, -1}, {1, 0}};
    ref = {{0, 1}, {1, 0}};
    break;
  default:
    throw std::runtime_error("Unexpected cell type");
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

  return M;
}
//----------------------------------------------------------------------------
xt::xtensor<double, 3> moments::create_normal_moment_dof_transformations(
    const FiniteElement& moment_space)
{
  xt::xtensor<double, 3> t
      = create_dot_moment_dof_transformations(moment_space);
  const int tdim = cell::topological_dimension(moment_space.cell_type());
  if (tdim == 1 or tdim == 2)
    xt::view(t, tdim - 1, xt::all(), xt::all()) *= -1.0;
  return t;
}
//----------------------------------------------------------------------------
xt::xtensor<double, 3> moments::create_tangent_moment_dof_transformations(
    const FiniteElement& moment_space)
{
  const int tdim = cell::topological_dimension(moment_space.cell_type());
  // FIXME: Should this check by tdim != 1?
  if (tdim == 2)
    throw std::runtime_error("Tangent is only well-defined on an edge.");

  xt::xtensor<double, 3> t
      = create_dot_moment_dof_transformations(moment_space);
  if (tdim == 1)
    xt::view(t, 0, xt::all(), xt::all()) *= -1.0;

  return t;
}
//----------------------------------------------------------------------------
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
moments::make_integral_moments(const FiniteElement& moment_space,
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
      = quadrature::make_quadrature("default", sub_celltype, q_deg);
  auto Qwts = xt::adapt(_Qwts);
  if (Qpts.dimension() == 1)
    Qpts = Qpts.reshape({Qpts.shape()[0], 1});

  // Evaluate moment space at quadrature points
  xt::xtensor<double, 2> moment_space_at_Qpts
      = xt::view(moment_space.tabulate_new(0, Qpts), 0, xt::all(), xt::all());

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
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
moments::make_dot_integral_moments(const FiniteElement& moment_space,
                                   cell::type celltype, std::size_t value_size,
                                   int q_deg)
{
  const cell::type sub_celltype = moment_space.cell_type();
  const std::size_t sub_entity_dim = cell::topological_dimension(sub_celltype);
  const std::size_t sub_entity_count
      = cell::sub_entity_count(celltype, sub_entity_dim);
  const std::size_t tdim = cell::topological_dimension(celltype);

  auto [qpts, _qwts]
      = quadrature::make_quadrature("default", sub_celltype, q_deg);
  auto qwts = xt::adapt(_qwts);
  if (qpts.dimension() == 1)
    qpts = qpts.reshape({qpts.shape()[0], 1});

  // If this is always true, value_size input can be removed
  assert(tdim == value_size);

  // Evaluate moment space at quadrature points
  xt::xtensor<double, 2> moment_space_at_Qpts
      = xt::view(moment_space.tabulate_new(0, qpts), 0, xt::all(), xt::all());

  const std::size_t moment_space_size
      = moment_space_at_Qpts.shape()[1] / sub_entity_dim;

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
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
moments::make_tangent_integral_moments(const FiniteElement& moment_space,
                                       cell::type celltype,
                                       std::size_t value_size, int q_deg)
{
  const cell::type sub_celltype = moment_space.cell_type();
  const std::size_t sub_entity_dim = cell::topological_dimension(sub_celltype);
  const std::size_t sub_entity_count
      = cell::sub_entity_count(celltype, sub_entity_dim);
  const std::size_t tdim = cell::topological_dimension(celltype);

  if (sub_entity_dim != 1)
    throw std::runtime_error("XXXTangent is only well-defined on an edge.");

  auto [Qpts, _Qwts]
      = quadrature::make_quadrature("default", cell::type::interval, q_deg);
  auto Qwts = xt::adapt(_Qwts);
  if (Qpts.dimension() == 1)
    Qpts = Qpts.reshape({Qpts.shape()[0], 1});

  // If this is always true, value_size input can be removed
  assert(tdim == value_size);

  // Evaluate moment space at quadrature points
  xt::xtensor<double, 2> moment_space_at_Qpts
      = xt::view(moment_space.tabulate_new(0, Qpts), 0, xt::all(), xt::all());

  xt::xtensor<double, 2> points({sub_entity_count * Qpts.shape()[0], tdim});
  const std::array<std::size_t, 2> shape
      = {moment_space_at_Qpts.shape()[1] * sub_entity_count,
         sub_entity_count * Qpts.shape()[0] * value_size};
  xt::xtensor<double, 2> matrix = xt::zeros<double>(shape);

  // Iterate over sub entities
  int c = 0;
  for (std::size_t i = 0; i < sub_entity_count; ++i)
  {
    xt::xtensor<double, 2> edge = cell::sub_entity_geometry(celltype, 1, i);
    auto tangent = xt::row(edge, 1) - xt::row(edge, 0);

    // No need to normalise the tangent, as the size of this is equal to
    // the integral Jacobian

    // Map quadrature points onto triangle edge
    for (std::size_t j = 0; j < Qpts.shape()[0]; ++j)
    {
      xt::row(points, i * Qpts.shape()[0] + j)
          = xt::row(edge, 0) + Qpts(j, 0) * tangent;
    }

    // Compute edge tangent integral moments
    for (std::size_t j = 0; j < moment_space_at_Qpts.shape()[1]; ++j)
    {
      auto phi = xt::col(moment_space_at_Qpts, j);
      for (std::size_t k = 0; k < value_size; ++k)
      {
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
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
moments::make_normal_integral_moments(const FiniteElement& moment_space,
                                      cell::type celltype,
                                      std::size_t value_size, int q_deg)
{
  const cell::type sub_celltype = moment_space.cell_type();
  const std::size_t sub_entity_dim = cell::topological_dimension(sub_celltype);
  const std::size_t sub_entity_count
      = cell::sub_entity_count(celltype, sub_entity_dim);
  const std::size_t tdim = cell::topological_dimension(celltype);

  // If this is always true, value_size input can be removed
  assert(tdim == value_size);

  if (static_cast<int>(sub_entity_dim) != static_cast<int>(tdim) - 1)
    throw std::runtime_error("Normal is only well-defined on a facet.");

  auto [Qpts, _Qwts]
      = quadrature::make_quadrature("default", sub_celltype, q_deg);
  auto Qwts = xt::adapt(_Qwts);
  if (Qpts.dimension() == 1)
    Qpts = Qpts.reshape({Qpts.shape()[0], 1});

  // Evaluate moment space at quadrature points
  xt::xtensor<double, 2> moment_space_at_Qpts
      = xt::view(moment_space.tabulate_new(0, Qpts), 0, xt::all(), xt::all());

  xt::xtensor<double, 2> points({sub_entity_count * Qpts.shape()[0], tdim});
  const std::array<std::size_t, 2> shape
      = {moment_space_at_Qpts.shape()[1] * sub_entity_count,
         sub_entity_count * Qpts.shape()[0] * value_size};
  xt::xtensor<double, 2> matrix = xt::zeros<double>(shape);

  // Iterate over sub entities
  int c = 0;
  xt::xtensor<double, 1> normal;
  for (std::size_t i = 0; i < sub_entity_count; ++i)
  {
    xt::xtensor<double, 2> facet
        = cell::sub_entity_geometry(celltype, tdim - 1, i);
    if (tdim == 2)
    {
      auto tangent = xt::row(facet, 1) - xt::row(facet, 0);
      normal = {-tangent(1), tangent(0)};

      // No need to normalise the normal, as the size of this is equal to
      // the integral jacobian

      // Map quadrature points onto facet
      for (std::size_t j = 0; j < Qpts.shape()[0]; ++j)
      {
        xt::row(points, i * Qpts.shape()[0] + j)
            = xt::row(facet, 0) + Qpts(j, 0) * tangent;
      }
    }
    else if (tdim == 3)
    {
      auto t0 = xt::row(facet, 1) - xt::row(facet, 0);
      auto t1 = xt::row(facet, 2) - xt::row(facet, 0);
      normal = xt::linalg::cross(t0, t1);

      // No need to normalise the normal, as the size of this is equal
      // to the integral Jacobian

      // Map quadrature points onto facet
      for (std::size_t j = 0; j < Qpts.shape()[0]; ++j)
      {
        xt::row(points, i * Qpts.shape()[0] + j)
            = xt::row(facet, 0) + Qpts(j, 0) * t0 + Qpts(j, 1) * t1;
      }
    }
    else
      throw std::runtime_error("Normal on this cell cannot be computed.");

    // Compute facet normal integral moments
    for (std::size_t j = 0; j < moment_space_at_Qpts.shape()[1]; ++j)
    {
      auto phi = xt::col(moment_space_at_Qpts, j);
      for (std::size_t k = 0; k < value_size; ++k)
      {
        std::size_t offset
            = k * sub_entity_count * Qpts.shape()[0] + i * Qpts.shape()[0];
        xt::view(matrix, c, xt::range(offset, offset + Qpts.shape()[0]))
            = phi * Qwts * normal[k];
      }
      ++c;
    }
  }

  return std::make_pair(points, matrix);
}
//----------------------------------------------------------------------------
