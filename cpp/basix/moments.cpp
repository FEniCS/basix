// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "moments.h"
#include "cell.h"
#include "finite-element.h"
#include "math.h"
#include "quadrature.h"
#include <xtensor/xbuilder.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

namespace stdex = std::experimental;
using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;

using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
using mdspan3_t = stdex::mdspan<double, stdex::dextents<std::size_t, 3>>;

using mdarray2_t = stdex::mdarray<double, stdex::dextents<std::size_t, 2>>;
using mdarray3_t = stdex::mdarray<double, stdex::dextents<std::size_t, 3>>;

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

/// Map points defined on a cell entity into the full cell space
/// @param[in] celltype0 Parent cell type
/// @param[in] celltype1 Sub-entity of `celltype0` type
/// @param[in] x Coordinates defined on an entity of type `celltype1`
/// @return (0) Coordinates of points in the full space of `celltype1`
/// (the shape is (num_entities, num points per entity, tdim of
/// celltype0) and (1) local axes on each entity (num_entities,
/// entity_dim, tdim).
template <typename P>
std::pair<std::vector<xt::xtensor<double, 2>>, xt::xtensor<double, 3>>
map_points(const cell::type celltype0, const cell::type celltype1, const P& x)
{
  const std::size_t tdim = cell::topological_dimension(celltype0);
  std::size_t entity_dim = cell::topological_dimension(celltype1);
  std::size_t num_entities = cell::num_sub_entities(celltype0, entity_dim);

  std::vector<xt::xtensor<double, 2>> p(num_entities,
                                        xt::zeros<double>({x.shape(0), tdim}));
  xt::xtensor<double, 3> axes({num_entities, entity_dim, tdim});

  const std::vector<int> axis_pts = axis_points(celltype0);
  for (std::size_t e = 0; e < num_entities; ++e)
  {
    // Get entity geometry
    xt::xtensor<double, 2> entity_x
        = cell::sub_entity_geometry(celltype0, entity_dim, e);

    // Axes on the cell entity
    for (std::size_t i = 0; i < entity_dim; ++i)
      for (std::size_t j = 0; j < axes.shape(2); ++j)
        axes(e, i, j) = entity_x(axis_pts[i], j) - entity_x(0, j);

    // Compute x = x0 + \Delta x
    xt::xtensor<double, 2> axes_e({axes.shape(1), axes.shape(2)});
    for (std::size_t i = 0; i < axes_e.shape(0); ++i)
      for (std::size_t j = 0; j < axes_e.shape(1); ++j)
        axes_e(i, j) = axes(e, i, j);

    xt::xtensor<double, 2> dx = math::dot(x, axes_e);
    for (std::size_t i = 0; i < p[e].shape(0); ++i)
      for (std::size_t j = 0; j < p[e].shape(1); ++j)
        p[e](i, j) = entity_x(0, j) + dx(i, j);
  }

  return {p, axes};
}
//----------------------------------------------------------------------------
/// Map points defined on a cell entity into the full cell space
/// @param[in] celltype0 Parent cell type
/// @param[in] celltype1 Sub-entity of `celltype0` type
/// @param[in] x Coordinates defined on an entity of type `celltype1`
/// @return (0) Coordinates of points in the full space of `celltype1`
/// (the shape is (num_entities, num points per entity, tdim of
/// celltype0) and (1) local axes on each entity (num_entities,
/// entity_dim, tdim).
std::pair<std::vector<mdarray2_t>, mdarray3_t>
map_points_new(const cell::type celltype0, const cell::type celltype1,
               cmdspan2_t x)
{
  const std::size_t tdim = cell::topological_dimension(celltype0);
  std::size_t entity_dim = cell::topological_dimension(celltype1);
  std::size_t num_entities = cell::num_sub_entities(celltype0, entity_dim);

  std::vector<mdarray2_t> p(num_entities, mdarray2_t(x.extent(0), tdim));
  mdarray3_t axes(num_entities, entity_dim, tdim);

  const std::vector<int> axis_pts = axis_points(celltype0);
  for (std::size_t e = 0; e < num_entities; ++e)
  {
    // Get entity geometry
    const auto [entity_buffer, eshape]
        = cell::sub_entity_geometry_new(celltype0, entity_dim, e);
    cmdspan2_t entity_x(entity_buffer.data(), eshape);

    // Axes on the cell entity
    for (std::size_t i = 0; i < axes.extent(1); ++i)
      for (std::size_t j = 0; j < axes.extent(2); ++j)
        axes(e, i, j) = entity_x(axis_pts[i], j) - entity_x(0, j);

    // Compute x = x0 + \Delta x
    mdarray2_t axes_e(axes.extent(1), axes.extent(2));
    for (std::size_t i = 0; i < axes_e.extent(0); ++i)
      for (std::size_t j = 0; j < axes_e.extent(1); ++j)
        axes_e(i, j) = axes(e, i, j);

    const auto [dxbuffer, dxshape] = math::dot_new(x, axes_e);
    cmdspan2_t dx(dxbuffer.data(), dxshape);
    for (std::size_t i = 0; i < p[e].extent(0); ++i)
      for (std::size_t j = 0; j < p[e].extent(1); ++j)
        p[e](i, j) = entity_x(0, j) + dx(i, j);
  }

  return {p, axes};
}
//----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::pair<std::vector<xt::xtensor<double, 2>>,
          std::vector<xt::xtensor<double, 4>>>
moments::make_integral_moments(const FiniteElement& V, cell::type celltype,
                               std::size_t value_size, int q_deg)
{
  const cell::type sub_celltype = V.cell_type();
  const std::size_t entity_dim = cell::topological_dimension(sub_celltype);
  if (entity_dim == 0)
    throw std::runtime_error("Cannot integrate over a dimension 0 entity.");
  const std::size_t num_entities = cell::num_sub_entities(celltype, entity_dim);

  // Get the quadrature points and weights
  const auto [_pts, wts] = quadrature::make_quadrature(
      quadrature::type::Default, sub_celltype, q_deg);
  cmdspan2_t pts(_pts.data(), wts.size(), _pts.size() / wts.size());

  // Evaluate moment space at quadrature points
  assert(std::accumulate(V.value_shape().begin(), V.value_shape().end(), 1,
                         std::multiplies<int>())
         == 1);
  const xt::xtensor<double, 2> phi
      = xt::view(V.tabulate(0, pts), 0, xt::all(), xt::all(), 0);

  // Pad out \phi moment is against a vector-valued function
  std::size_t vdim = value_size == 1 ? 1 : entity_dim;

  // Storage for the interpolation matrix
  const std::size_t num_dofs = vdim * phi.shape(1);
  const std::array<std::size_t, 4> shape
      = {num_dofs, value_size, pts.extent(0), 1};
  std::vector<xt::xtensor<double, 4>> D(num_entities, xt::zeros<double>(shape));

  // Map quadrature points onto facet (cell entity e)
  const auto [points, axes] = map_points_new(celltype, sub_celltype, pts);

  // Compute entity integral moments

  // Iterate over cell entities
  if (value_size == 1)
  {
    for (std::size_t e = 0; e < num_entities; ++e)
    {
      xt::xtensor<double, 4>& _D = D[e];
      for (std::size_t i = 0; i < phi.shape(1); ++i)
        for (std::size_t j = 0; j < wts.size(); ++j)
          _D(i, 0, j, 0) = phi(j, i) * wts[j];
    }
  }
  else
  {
    for (std::size_t e = 0; e < num_entities; ++e)
    {
      xt::xtensor<double, 4>& _D = D[e];

      // Loop over each 'dof' on an entity (moment basis function index)
      for (std::size_t i = 0; i < phi.shape(1); ++i)
      {
        // TODO: Pad-out phi and call a updated
        // make_dot_integral_moments

        // FIXME: This assumed that the moment space has a certain
        // mapping type
        for (std::size_t d = 0; d < entity_dim; ++d)
        {
          // TODO: check that dof index is correct
          const std::size_t dof = i * entity_dim + d;
          for (std::size_t j = 0; j < value_size; ++j)
            for (std::size_t k = 0; k < wts.size(); ++k)
              _D(dof, j, k, 0) = phi(k, i) * wts[k] * axes(e, d, j);
        }
      }
    }
  }

  std::vector<xt::xtensor<double, 2>> _points;
  for (auto& p : points)
  {
    auto& _p = _points.emplace_back(
        xt::xtensor<double, 2>({p.extent(0), p.extent(1)}));
    std::copy_n(p.data(), p.size(), _p.data());
  }

  return {_points, D};
}
//----------------------------------------------------------------------------
std::tuple<
    std::vector<std::vector<double>>, std::vector<std::array<std::size_t, 2>>,
    std::vector<std::vector<double>>, std::vector<std::array<std::size_t, 4>>>
moments::make_integral_moments_new(const FiniteElement& V, cell::type celltype,
                                   std::size_t value_size, int q_deg)
{
  auto [xold, Mold] = make_integral_moments(V, celltype, value_size, q_deg);
  std::vector<std::vector<double>> x;
  std::vector<std::array<std::size_t, 2>> xshape;
  std::vector<std::vector<double>> M;
  std::vector<std::array<std::size_t, 4>> Mshape;
  for (auto& _x : xold)
  {
    x.emplace_back(_x.data(), _x.data() + _x.size());
    xshape.push_back({_x.shape(0), _x.shape(1)});
  }

  for (auto& _M : Mold)
  {
    M.emplace_back(_M.data(), _M.data() + _M.size());
    Mshape.push_back({_M.shape(0), _M.shape(1), _M.shape(2), _M.shape(3)});
  }

  return {x, xshape, M, Mshape};
}
//----------------------------------------------------------------------------
std::pair<std::vector<xt::xtensor<double, 2>>,
          std::vector<xt::xtensor<double, 4>>>
moments::make_dot_integral_moments(const FiniteElement& V, cell::type celltype,
                                   std::size_t value_size, int q_deg)
{
  const cell::type sub_celltype = V.cell_type();
  const std::size_t entity_dim = cell::topological_dimension(sub_celltype);
  const std::size_t num_entities = cell::num_sub_entities(celltype, entity_dim);

  auto [_pts, wts] = quadrature::make_quadrature(quadrature::type::Default,
                                                 sub_celltype, q_deg);
  cmdspan2_t pts(_pts.data(), wts.size(), _pts.size() / wts.size());

  // If this is always true, value_size input can be removed
  assert(std::size_t(cell::topological_dimension(celltype)) == value_size);

  // Evaluate moment space at quadrature points
  xt::xtensor<double, 3> phi
      = xt::view(V.tabulate(0, pts), 0, xt::all(), xt::all(), xt::all());
  assert(phi.shape(2) == entity_dim);

  // Note:
  // Number of quadrature points per entity: phi.shape(0)
  // Dimension of the moment space on each entity: phi.shape(1)
  // Value size of the moment function: phi.shape(2)

  // Map quadrature points onto facet (cell entity e)
  auto [points, axes] = map_points_new(celltype, sub_celltype, pts);

  // Shape (num dofs, value size, num points)
  const std::array<std::size_t, 4> shape
      = {phi.shape(1), value_size, pts.extent(0), 1};
  std::vector<xt::xtensor<double, 4>> D(num_entities, xt::zeros<double>(shape));

  // Compute entity integral moments

  // Iterate over cell entities
  for (std::size_t e = 0; e < num_entities; ++e)
  {
    // Loop over each 'dof' on an entity (moment basis function index)
    for (std::size_t dof = 0; dof < phi.shape(1); ++dof)
    {
      // Loop over value size of function to which moment function is
      // applied
      for (std::size_t j = 0; j < value_size; ++j)
      {
        // Loop over value topological dimension of cell entity (which
        // is equal to phi.shape(2))
        for (std::size_t d = 0; d < phi.shape(2); ++d)
        {
          // Add quadrature point on cell entity contributions
          for (std::size_t k = 0; k < wts.size(); ++k)
            D[e](dof, j, k, 0) += wts[k] * phi(k, dof, d) * axes(e, d, j);
        }
      }
    }
  }

  std::vector<xt::xtensor<double, 2>> _points;
  for (auto& p : points)
  {
    auto& _p = _points.emplace_back(
        xt::xtensor<double, 2>({p.extent(0), p.extent(1)}));
    std::copy_n(p.data(), p.size(), _p.data());
  }

  return {_points, D};
}
//----------------------------------------------------------------------------
std::tuple<
    std::vector<std::vector<double>>, std::vector<std::array<std::size_t, 2>>,
    std::vector<std::vector<double>>, std::vector<std::array<std::size_t, 4>>>
moments::make_dot_integral_moments_new(const FiniteElement& V,
                                       cell::type celltype,
                                       std::size_t value_size, int q_deg)
{
  auto [xold, Mold] = make_dot_integral_moments(V, celltype, value_size, q_deg);
  std::vector<std::vector<double>> x;
  std::vector<std::array<std::size_t, 2>> xshape;
  std::vector<std::vector<double>> M;
  std::vector<std::array<std::size_t, 4>> Mshape;
  for (auto& _x : xold)
  {
    x.emplace_back(_x.data(), _x.data() + _x.size());
    xshape.push_back({_x.shape(0), _x.shape(1)});
  }

  for (auto& _M : Mold)
  {
    M.emplace_back(_M.data(), _M.data() + _M.size());
    Mshape.push_back({_M.shape(0), _M.shape(1), _M.shape(2), _M.shape(3)});
  }

  return {x, xshape, M, Mshape};
}
//----------------------------------------------------------------------------
std::pair<std::vector<xt::xtensor<double, 2>>,
          std::vector<xt::xtensor<double, 4>>>
moments::make_tangent_integral_moments(const FiniteElement& V,
                                       cell::type celltype,
                                       std::size_t value_size, int q_deg)
{
  const cell::type sub_celltype = V.cell_type();
  const std::size_t entity_dim = cell::topological_dimension(sub_celltype);
  const std::size_t num_entities = cell::num_sub_entities(celltype, entity_dim);
  const std::size_t tdim = cell::topological_dimension(celltype);

  // If this is always true, value_size input can be removed
  assert(tdim == value_size);

  if (entity_dim != 1)
    throw std::runtime_error("Tangent is only well-defined on an edge.");

  auto [_pts, wts] = quadrature::make_quadrature(quadrature::type::Default,
                                                 cell::type::interval, q_deg);
  cmdspan2_t pts(_pts.data(), wts.size(), _pts.size() / wts.size());

  // Evaluate moment space at quadrature points
  assert(std::accumulate(V.value_shape().begin(), V.value_shape().end(), 1,
                         std::multiplies<int>())
         == 1);
  xt::xtensor<double, 2> phi
      = xt::view(V.tabulate(0, pts), 0, xt::all(), xt::all(), 0);

  std::vector<xt::xtensor<double, 2>> points(
      num_entities, xt::zeros<double>({pts.extent(0), tdim}));
  const std::array<std::size_t, 4> shape
      = {phi.shape(1), value_size, phi.shape(0), 1};
  std::vector<xt::xtensor<double, 4>> D(num_entities, xt::zeros<double>(shape));

  // Iterate over cell entities
  for (std::size_t e = 0; e < num_entities; ++e)
  {
    xt::xtensor<double, 2> edge_x = cell::sub_entity_geometry(celltype, 1, e);
    auto X0 = xt::row(edge_x, 0);
    auto tangent = xt::row(edge_x, 1) - X0;

    // No need to normalise the tangent, as the size of this is equal to
    // the integral Jacobian

    // Map quadrature points onto triangle edge
    for (std::size_t i = 0; i < pts.extent(0); ++i)
      for (std::size_t j = 0; j < points[e].shape(1); ++j)
        points[e](i, j) = X0[j] + pts(i, 0) * tangent[j];

    // Compute edge tangent integral moments
    for (std::size_t i = 0; i < phi.shape(1); ++i)
    {
      for (std::size_t j = 0; j < value_size; ++j)
        for (std::size_t k = 0; k < wts.size(); ++k)
          D[e](i, j, k, 0) = phi(k, i) * wts[k] * tangent[j];
    }
  }

  return {points, D};
}
//----------------------------------------------------------------------------
std::tuple<
    std::vector<std::vector<double>>, std::vector<std::array<std::size_t, 2>>,
    std::vector<std::vector<double>>, std::vector<std::array<std::size_t, 4>>>
moments::make_tangent_integral_moments_new(const FiniteElement& V,
                                           cell::type celltype,
                                           std::size_t value_size, int q_deg)
{
  auto [xold, Mold]
      = make_tangent_integral_moments(V, celltype, value_size, q_deg);
  std::vector<std::vector<double>> x;
  std::vector<std::array<std::size_t, 2>> xshape;
  std::vector<std::vector<double>> M;
  std::vector<std::array<std::size_t, 4>> Mshape;
  for (auto& _x : xold)
  {
    x.emplace_back(_x.data(), _x.data() + _x.size());
    xshape.push_back({_x.shape(0), _x.shape(1)});
  }

  for (auto& _M : Mold)
  {
    M.emplace_back(_M.data(), _M.data() + _M.size());
    Mshape.push_back({_M.shape(0), _M.shape(1), _M.shape(2), _M.shape(3)});
  }

  return {x, xshape, M, Mshape};
}
//----------------------------------------------------------------------------
std::pair<std::vector<xt::xtensor<double, 2>>,
          std::vector<xt::xtensor<double, 4>>>
moments::make_normal_integral_moments(const FiniteElement& V,
                                      cell::type celltype,
                                      std::size_t value_size, int q_deg)
{
  const std::size_t tdim = cell::topological_dimension(celltype);
  assert(tdim == value_size);
  const cell::type sub_celltype = V.cell_type();
  const std::size_t entity_dim = cell::topological_dimension(sub_celltype);
  const std::size_t num_entities = cell::num_sub_entities(celltype, entity_dim);

  if (static_cast<int>(entity_dim) != static_cast<int>(tdim) - 1)
    throw std::runtime_error("Normal is only well-defined on a facet.");

  // Compute quadrature points for evaluating integral
  auto [_pts, wts] = quadrature::make_quadrature(quadrature::type::Default,
                                                 sub_celltype, q_deg);
  cmdspan2_t pts(_pts.data(), wts.size(), _pts.size() / wts.size());

  // Evaluate moment space at quadrature points
  assert(std::accumulate(V.value_shape().begin(), V.value_shape().end(), 1,
                         std::multiplies<int>())
         == 1);
  xt::xtensor<double, 2> phi
      = xt::view(V.tabulate(0, pts), 0, xt::all(), xt::all(), 0);

  // Storage for coordinates of evaluations points in the reference cell
  std::vector<xt::xtensor<double, 2>> points(
      num_entities, xt::zeros<double>({pts.extent(0), tdim}));

  // Storage for interpolation matrix
  const std::array<std::size_t, 4> shape
      = {phi.shape(1), value_size, phi.shape(0), 1};
  std::vector<xt::xtensor<double, 4>> D(num_entities, xt::zeros<double>(shape));

  // Evaluate moment space at quadrature points

  // Iterate over cell entities
  std::array<double, 3> normal;
  for (std::size_t e = 0; e < num_entities; ++e)
  {
    xt::xtensor<double, 2>& _points = points[e];

    // Map quadrature points onto facet (cell entity e)
    xt::xtensor<double, 2> facet_x
        = cell::sub_entity_geometry(celltype, tdim - 1, e);
    auto x0 = xt::row(facet_x, 0);
    if (tdim == 2)
    {
      // No need to normalise the normal, as the size of this is equal
      // to the integral jacobian
      auto tangent = xt::row(facet_x, 1) - x0;
      for (std::size_t p = 0; p < _points.shape(0); ++p)
        for (std::size_t i = 0; i < _points.shape(1); ++i)
          _points(p, i) = x0[i] + pts(p, 0) * tangent[i];

      normal = {-tangent(1), tangent(0), 0.0};
    }
    else if (tdim == 3)
    {
      // No need to normalise the normal, as the size of this is equal
      // to the integral Jacobian
      auto t0 = xt::row(facet_x, 1) - x0;
      auto t1 = xt::row(facet_x, 2) - x0;
      for (std::size_t p = 0; p < _points.shape(0); ++p)
        for (std::size_t i = 0; i < _points.shape(1); ++i)
          _points(p, i) = x0[i] + pts(p, 0) * t0[i] + pts(p, 1) * t1[i];

      normal = basix::math::cross_new(t0, t1);
    }
    else
      throw std::runtime_error("Normal on this cell cannot be computed.");

    // Compute facet normal integral moments
    xt::xtensor<double, 4>& _D = D[e];
    for (std::size_t i = 0; i < phi.shape(1); ++i)
      for (std::size_t j = 0; j < value_size; ++j)
        for (std::size_t k = 0; k < D[e].shape(2); ++k)
          _D(i, j, k, 0) = phi(k, i) * wts[k] * normal[j];
  }

  return {points, D};
}
//----------------------------------------------------------------------------
std::tuple<
    std::vector<std::vector<double>>, std::vector<std::array<std::size_t, 2>>,
    std::vector<std::vector<double>>, std::vector<std::array<std::size_t, 4>>>
moments::make_normal_integral_moments_new(const FiniteElement& V,
                                          cell::type celltype,
                                          std::size_t value_size, int q_deg)
{
  auto [xold, Mold]
      = make_normal_integral_moments(V, celltype, value_size, q_deg);
  std::vector<std::vector<double>> x;
  std::vector<std::array<std::size_t, 2>> xshape;
  std::vector<std::vector<double>> M;
  std::vector<std::array<std::size_t, 4>> Mshape;
  for (auto& _x : xold)
  {
    x.emplace_back(_x.data(), _x.data() + _x.size());
    xshape.push_back({_x.shape(0), _x.shape(1)});
  }

  for (auto& _M : Mold)
  {
    M.emplace_back(_M.data(), _M.data() + _M.size());
    Mshape.push_back({_M.shape(0), _M.shape(1), _M.shape(2), _M.shape(3)});
  }

  return {x, xshape, M, Mshape};
}
//----------------------------------------------------------------------------
