// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "moments.h"
#include "cell.h"
#include "finite-element.h"
#include "math.h"
#include "quadrature.h"
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xpad.hpp>
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
// Map points defined on a cell entity into the full cell space
// @param[in] celltype0 Parent cell type
// @param[in] celltype1 Sub-entity of `celltype0` type
// @param[in] x Coordinates defined on an entity of type `celltype1`
// @return (0) Coordinates of points in the full space of `celltype1`
// (the shape is (num_entities, num points per entity, tdim of
// celltype0) and (1) local axes on each entity (num_entities,
// entity_dim, tdim).
template <typename P>
std::pair<std::vector<xt::xtensor<double, 2>>, xt::xtensor<double, 3>>
map_points(const cell::type celltype0, const cell::type celltype1, const P& x)
{
  assert(x.dimension() == 2);

  const std::size_t tdim = cell::topological_dimension(celltype0);
  std::size_t entity_dim = cell::topological_dimension(celltype1);
  std::size_t num_entities = cell::num_sub_entities(celltype0, entity_dim);

  std::vector<xt::xtensor<double, 2>> p(num_entities,
                                        xt::zeros<double>({x.shape(0), tdim}));
  xt::xtensor<double, 3> axes({num_entities, entity_dim, tdim});
  xt::xtensor<double, 2> axes_e({entity_dim, tdim});
  const std::vector<int> axis_pts = axis_points(celltype0);
  for (std::size_t e = 0; e < num_entities; ++e)
  {
    // Get entity geometry
    xt::xtensor<double, 2> entity_x
        = cell::sub_entity_geometry(celltype0, entity_dim, e);
    auto x0 = xt::row(entity_x, 0);

    // Axes on the cell entity
    for (std::size_t i = 0; i < entity_dim; ++i)
      xt::view(axes, e, i, xt::all()) = xt::row(entity_x, axis_pts[i]) - x0;

    // Compute x = x0 + \Delta x
    p[e] = xt::tile(xt::view(entity_x, xt::newaxis(), 0), x.shape(0));
    axes_e = xt::view(axes, e, xt::all(), xt::all());
    p[e] += basix::math::dot(x, axes_e);
  }

  return {p, axes};
}
//----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::pair<std::vector<xt::xtensor<double, 2>>,
          std::vector<xt::xtensor<double, 3>>>
moments::make_integral_moments(const FiniteElement& V, cell::type celltype,
                               std::size_t value_size, int q_deg)
{
  const cell::type sub_celltype = V.cell_type();
  const std::size_t entity_dim = cell::topological_dimension(sub_celltype);
  if (entity_dim == 0)
    throw std::runtime_error("Cannot integrate over a dimension 0 entity.");
  const std::size_t num_entities = cell::num_sub_entities(celltype, entity_dim);

  // Get the quadrature points and weights
  auto [pts, _wts] = quadrature::make_quadrature(quadrature::type::Default,
                                                 sub_celltype, q_deg);
  auto wts = xt::adapt(_wts);
  if (pts.dimension() == 1)
    pts = pts.reshape({pts.shape(0), 1});

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
  const std::array<std::size_t, 3> shape = {num_dofs, value_size, pts.shape(0)};
  std::vector<xt::xtensor<double, 3>> D(num_entities, xt::zeros<double>(shape));

  // Map quadrature points onto facet (cell entity e)
  const auto [points, axes] = map_points(celltype, sub_celltype, pts);

  // Compute entity integral moments

  // Iterate over cell entities
  if (value_size == 1)
  {
    for (std::size_t e = 0; e < num_entities; ++e)
    {
      for (std::size_t i = 0; i < phi.shape(1); ++i)
      {
        auto phi_i = xt::col(phi, i);
        xt::view(D[e], i, 0, xt::all()) = phi_i * wts;
      }
    }
  }
  else
  {
    for (std::size_t e = 0; e < num_entities; ++e)
    {
      // Loop over each 'dof' on an entity (moment basis function index)
      for (std::size_t i = 0; i < phi.shape(1); ++i)
      {
        auto phi_i = xt::col(phi, i);
        // TODO: Pad-out phi and call a updated
        // make_dot_integral_moments

        // FIXME: This assumed that the moment space has a certain
        // mapping type
        for (std::size_t d = 0; d < entity_dim; ++d)
        {
          // TODO: check that dof index is correct
          const std::size_t dof = i * entity_dim + d;
          for (std::size_t k = 0; k < value_size; ++k)
            xt::view(D[e], dof, k, xt::all()) = phi_i * wts * axes(e, d, k);
        }
      }
    }
  }

  return {points, D};
}
//----------------------------------------------------------------------------
std::pair<std::vector<xt::xtensor<double, 2>>,
          std::vector<xt::xtensor<double, 3>>>
moments::make_dot_integral_moments(const FiniteElement& V, cell::type celltype,
                                   std::size_t value_size, int q_deg)
{
  const cell::type sub_celltype = V.cell_type();
  const std::size_t entity_dim = cell::topological_dimension(sub_celltype);
  const std::size_t num_entities = cell::num_sub_entities(celltype, entity_dim);

  auto [pts, _wts] = quadrature::make_quadrature(quadrature::type::Default,
                                                 sub_celltype, q_deg);
  auto wts = xt::adapt(_wts);

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
  auto [points, axes] = map_points(celltype, sub_celltype, pts);

  // Shape (num dofs, value size, num points)
  const std::array shape = {phi.shape(1), value_size, pts.shape(0)};
  std::vector<xt::xtensor<double, 3>> D(num_entities, xt::zeros<double>(shape));

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
          xt::view(D[e], dof, j, xt::all())
              += wts * xt::view(phi, xt::all(), dof, d) * axes(e, d, j);
        }
      }
    }
  }

  return {points, D};
}
//----------------------------------------------------------------------------
std::pair<std::vector<xt::xtensor<double, 2>>,
          std::vector<xt::xtensor<double, 3>>>
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

  auto [pts, _wts] = quadrature::make_quadrature(quadrature::type::Default,
                                                 cell::type::interval, q_deg);
  auto wts = xt::adapt(_wts);

  // Evaluate moment space at quadrature points
  assert(std::accumulate(V.value_shape().begin(), V.value_shape().end(), 1,
                         std::multiplies<int>())
         == 1);
  xt::xtensor<double, 2> phi
      = xt::view(V.tabulate(0, pts), 0, xt::all(), xt::all(), 0);

  std::vector<xt::xtensor<double, 2>> points(
      num_entities, xt::zeros<double>({pts.shape(0), tdim}));
  const std::array shape = {phi.shape(1), value_size, phi.shape(0)};
  std::vector<xt::xtensor<double, 3>> D(num_entities, xt::zeros<double>(shape));

  // Iterate over cell entities
  for (std::size_t e = 0; e < num_entities; ++e)
  {
    xt::xtensor<double, 2> edge_x = cell::sub_entity_geometry(celltype, 1, e);
    auto X0 = xt::row(edge_x, 0);
    auto tangent = xt::row(edge_x, 1) - X0;

    // No need to normalise the tangent, as the size of this is equal to
    // the integral Jacobian

    // Map quadrature points onto triangle edge
    for (std::size_t i = 0; i < pts.shape(0); ++i)
      xt::view(points[e], i, xt::all()) = X0 + pts[i] * tangent;

    // Compute edge tangent integral moments
    for (std::size_t i = 0; i < phi.shape(1); ++i)
    {
      auto phi_i = xt::col(phi, i);
      for (std::size_t j = 0; j < value_size; ++j)
        xt::view(D[e], i, j, xt::all()) = phi_i * wts * tangent[j];
    }
  }

  return {points, D};
}
//----------------------------------------------------------------------------
std::pair<std::vector<xt::xtensor<double, 2>>,
          std::vector<xt::xtensor<double, 3>>>
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
  auto [pts, _wts] = quadrature::make_quadrature(quadrature::type::Default,
                                                 sub_celltype, q_deg);
  auto wts = xt::adapt(_wts);

  // Evaluate moment space at quadrature points
  assert(std::accumulate(V.value_shape().begin(), V.value_shape().end(), 1,
                         std::multiplies<int>())
         == 1);
  xt::xtensor<double, 2> phi
      = xt::view(V.tabulate(0, pts), 0, xt::all(), xt::all(), 0);

  // Storage for coordinates of evaluations points in the reference cell
  std::vector<xt::xtensor<double, 2>> points(
      num_entities, xt::zeros<double>({pts.shape(0), tdim}));

  // Storage for interpolation matrix
  const std::array shape = {phi.shape(1), value_size, phi.shape(0)};
  std::vector<xt::xtensor<double, 3>> D(num_entities, xt::zeros<double>(shape));

  // Evaluate moment space at quadrature points

  // Iterate over cell entities
  xt::xtensor<double, 1> normal;
  for (std::size_t e = 0; e < num_entities; ++e)
  {
    // Map quadrature points onto facet (cell entity e)
    xt::xtensor<double, 2> facet_x
        = cell::sub_entity_geometry(celltype, tdim - 1, e);
    auto x0 = xt::row(facet_x, 0);
    if (tdim == 2)
    {
      // No need to normalise the normal, as the size of this is equal
      // to the integral jacobian
      auto tangent = xt::row(facet_x, 1) - x0;
      normal = {-tangent(1), tangent(0)};
      for (std::size_t p = 0; p < pts.shape(0); ++p)
        xt::view(points[e], p, xt::all()) = x0 + pts[p] * tangent;
    }
    else if (tdim == 3)
    {
      // No need to normalise the normal, as the size of this is equal
      // to the integral Jacobian
      auto t0 = xt::row(facet_x, 1) - x0;
      auto t1 = xt::row(facet_x, 2) - x0;
      normal = basix::math::cross(t0, t1);
      for (std::size_t p = 0; p < pts.shape(0); ++p)
      {
        xt::view(points[e], p, xt::all())
            = x0 + pts(p, 0) * t0 + pts(p, 1) * t1;
      }
    }
    else
      throw std::runtime_error("Normal on this cell cannot be computed.");

    // Compute facet normal integral moments
    for (std::size_t i = 0; i < phi.shape(1); ++i)
    {
      auto phi_i = xt::col(phi, i);
      for (std::size_t j = 0; j < value_size; ++j)
        xt::view(D[e], i, j, xt::all()) = phi_i * wts * normal[j];
    }
  }

  return {points, D};
}
//----------------------------------------------------------------------------
