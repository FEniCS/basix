// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "moments.h"
#include "cell.h"
#include "finite-element.h"
#include "math.h"
#include "quadrature.h"

using namespace basix;

namespace stdex = std::experimental;
using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
using cmdspan4_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;

using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
using mdspan3_t = stdex::mdspan<double, stdex::dextents<std::size_t, 3>>;
using mdspan4_t = stdex::mdspan<double, stdex::dextents<std::size_t, 4>>;

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
std::pair<std::vector<mdarray2_t>, mdarray3_t>
map_points(const cell::type celltype0, const cell::type celltype1, cmdspan2_t x)
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
        = cell::sub_entity_geometry(celltype0, entity_dim, e);
    cmdspan2_t entity_x(entity_buffer.data(), eshape);

    // Axes on the cell entity
    for (std::size_t i = 0; i < axes.extent(1); ++i)
      for (std::size_t j = 0; j < axes.extent(2); ++j)
        axes(e, i, j) = entity_x(axis_pts[i], j) - entity_x(0, j);

    // Compute x = x0 + \Delta x
    std::vector<double> axes_b(axes.extent(1) * axes.extent(2));
    mdspan2_t axes_e(axes_b.data(), axes.extent(1), axes.extent(2));
    for (std::size_t i = 0; i < axes_e.extent(0); ++i)
      for (std::size_t j = 0; j < axes_e.extent(1); ++j)
        axes_e(i, j) = axes(e, i, j);

    std::vector<double> dxbuffer(x.extent(0) * axes_e.extent(1));
    mdspan2_t dx(dxbuffer.data(), x.extent(0), axes_e.extent(1));
    math::dot(x, axes_e, dx);

    for (std::size_t i = 0; i < p[e].extent(0); ++i)
      for (std::size_t j = 0; j < p[e].extent(1); ++j)
        p[e](i, j) = entity_x(0, j) + dx(i, j);
  }

  return {p, axes};
}
//----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::tuple<std::vector<std::vector<double>>, std::array<std::size_t, 2>,
           std::vector<std::vector<double>>, std::array<std::size_t, 4>>
moments::make_integral_moments(const FiniteElement& V, cell::type celltype,
                               std::size_t value_size, int q_deg)
{
  const cell::type sub_celltype = V.cell_type();
  const std::size_t entity_dim = cell::topological_dimension(sub_celltype);
  if (entity_dim == 0)
    throw std::runtime_error("Cannot integrate over a dimension 0 entity.");
  const std::size_t num_entities = cell::num_sub_entities(celltype, entity_dim);

  // Get the quadrature points and weights
  const auto [_pts, wts] = quadrature::make_quadrature<double>(
      quadrature::type::Default, sub_celltype, q_deg);
  cmdspan2_t pts(_pts.data(), wts.size(), _pts.size() / wts.size());

  // Evaluate moment space at quadrature points
  assert(std::accumulate(V.value_shape().begin(), V.value_shape().end(), 1,
                         std::multiplies{})
         == 1);
  const auto [phib, phishape] = V.tabulate(0, pts);
  cmdspan4_t phi(phib.data(), phishape);

  // Pad out \phi moment is against a vector-valued function
  const std::size_t vdim = value_size == 1 ? 1 : entity_dim;

  // Storage for the interpolation matrix
  const std::size_t num_dofs = vdim * phi.extent(2);
  const std::array<std::size_t, 4> Dshape
      = {num_dofs, value_size, pts.extent(0), 1};

  const std::size_t size
      = std::reduce(Dshape.begin(), Dshape.end(), 1, std::multiplies{});
  std::vector<std::vector<double>> Db(num_entities, std::vector<double>(size));
  std::vector<mdspan4_t> D;

  // Map quadrature points onto facet (cell entity e)
  const auto [points, axes] = map_points(celltype, sub_celltype, pts);

  // -- Compute entity integral moments

  // Iterate over cell entities
  if (value_size == 1)
  {
    for (std::size_t e = 0; e < num_entities; ++e)
    {
      mdspan4_t& _D = D.emplace_back(Db[e].data(), Dshape);
      for (std::size_t i = 0; i < phi.extent(2); ++i)
        for (std::size_t j = 0; j < wts.size(); ++j)
          _D(i, 0, j, 0) = phi(0, j, i, 0) * wts[j];
    }
  }
  else
  {
    for (std::size_t e = 0; e < num_entities; ++e)
    {
      mdspan4_t& _D = D.emplace_back(Db[e].data(), Dshape);

      // Loop over each 'dof' on an entity (moment basis function index)
      for (std::size_t i = 0; i < phi.extent(2); ++i)
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
              _D(dof, j, k, 0) = phi(0, k, i, 0) * wts[k] * axes(e, d, j);
        }
      }
    }
  }

  const std::array<std::size_t, 2> pshape
      = {points.front().extent(0), points.front().extent(1)};
  std::vector<std::vector<double>> pb;
  for (const mdarray2_t& p : points)
    pb.emplace_back(p.data(), p.data() + p.size());

  return {pb, pshape, Db, Dshape};
}
//----------------------------------------------------------------------------
std::tuple<std::vector<std::vector<double>>, std::array<std::size_t, 2>,
           std::vector<std::vector<double>>, std::array<std::size_t, 4>>
moments::make_dot_integral_moments(const FiniteElement& V, cell::type celltype,
                                   std::size_t value_size, int q_deg)
{
  const cell::type sub_celltype = V.cell_type();
  const std::size_t entity_dim = cell::topological_dimension(sub_celltype);
  const std::size_t num_entities = cell::num_sub_entities(celltype, entity_dim);

  const auto [_pts, wts] = quadrature::make_quadrature<double>(
      quadrature::type::Default, sub_celltype, q_deg);
  cmdspan2_t pts(_pts.data(), wts.size(), _pts.size() / wts.size());

  // If this is always true, value_size input can be removed
  assert(std::size_t(cell::topological_dimension(celltype)) == value_size);

  // Evaluate moment space at quadrature points
  const auto [phib, phishape] = V.tabulate(0, pts);
  cmdspan4_t phi(phib.data(), phishape);
  assert(phi.extent(3) == entity_dim);

  // Note:
  // Number of quadrature points per entity: phi.extent(0)
  // Dimension of the moment space on each entity: phi.extent(1)
  // Value size of the moment function: phi.extent(2)

  // Map quadrature points onto facet (cell entity e)
  const auto [points, axes] = map_points(celltype, sub_celltype, pts);

  // Shape (num dofs, value size, num points)
  const std::array<std::size_t, 4> Dshape
      = {phi.extent(2), value_size, pts.extent(0), 1};
  const std::size_t size
      = std::reduce(Dshape.begin(), Dshape.end(), 1, std::multiplies{});
  std::vector<std::vector<double>> Db(num_entities, std::vector<double>(size));
  std::vector<mdspan4_t> D;

  // Compute entity integral moments

  // Iterate over cell entities
  for (std::size_t e = 0; e < num_entities; ++e)
  {
    mdspan4_t& _D = D.emplace_back(Db[e].data(), Dshape);

    // Loop over each 'dof' on an entity (moment basis function index)
    for (std::size_t dof = 0; dof < phi.extent(2); ++dof)
    {
      // Loop over value size of function to which moment function is
      // applied
      for (std::size_t j = 0; j < value_size; ++j)
      {
        // Loop over value topological dimension of cell entity (which
        // is equal to phi.extent(3))
        for (std::size_t d = 0; d < phi.extent(3); ++d)
        {
          // Add quadrature point on cell entity contributions
          for (std::size_t k = 0; k < wts.size(); ++k)
            _D(dof, j, k, 0) += wts[k] * phi(0, k, dof, d) * axes(e, d, j);
        }
      }
    }
  }

  const std::array<std::size_t, 2> pshape
      = {points.front().extent(0), points.front().extent(1)};
  std::vector<std::vector<double>> pb;
  for (const mdarray2_t& p : points)
    pb.emplace_back(p.data(), p.data() + p.size());

  return {pb, pshape, Db, Dshape};
}
//----------------------------------------------------------------------------
std::tuple<std::vector<std::vector<double>>, std::array<std::size_t, 2>,
           std::vector<std::vector<double>>, std::array<std::size_t, 4>>
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

  const auto [_pts, wts] = quadrature::make_quadrature<double>(
      quadrature::type::Default, cell::type::interval, q_deg);
  cmdspan2_t pts(_pts.data(), wts.size(), _pts.size() / wts.size());

  // Evaluate moment space at quadrature points
  assert(std::accumulate(V.value_shape().begin(), V.value_shape().end(), 1,
                         std::multiplies{})
         == 1);
  const auto [phib, phishape] = V.tabulate(0, pts);
  cmdspan4_t phi(phib.data(), phishape);

  const std::array<std::size_t, 2> pshape = {pts.extent(0), tdim};
  std::vector<std::vector<double>> pb;

  const std::array<std::size_t, 4> Dshape
      = {phi.extent(2), value_size, phi.extent(1), 1};
  const std::size_t size
      = std::reduce(Dshape.begin(), Dshape.end(), 1, std::multiplies{});
  std::vector<std::vector<double>> Db(num_entities, std::vector<double>(size));
  std::vector<mdspan4_t> D;

  // Iterate over cell entities
  for (std::size_t e = 0; e < num_entities; ++e)
  {
    const auto [ebuffer, eshape] = cell::sub_entity_geometry(celltype, 1, e);
    impl::cmdspan2_t edge_x(ebuffer.data(), eshape);

    std::vector<double> tangent(edge_x.extent(1));
    for (std::size_t i = 0; i < edge_x.extent(1); ++i)
      tangent[i] = edge_x(1, i) - edge_x(0, i);

    // No need to normalise the tangent, as the size of this is equal to
    // the integral Jacobian

    // Map quadrature points onto triangle edge
    auto& _pb = pb.emplace_back(pshape[0] * pshape[1]);
    mdspan2_t _p(_pb.data(), pshape);
    for (std::size_t i = 0; i < pts.extent(0); ++i)
      for (std::size_t j = 0; j < _p.extent(1); ++j)
        _p(i, j) = edge_x(0, j) + pts(i, 0) * tangent[j];

    // Compute edge tangent integral moments
    mdspan4_t& _D = D.emplace_back(Db[e].data(), Dshape);
    for (std::size_t i = 0; i < phi.extent(2); ++i)
    {
      for (std::size_t j = 0; j < value_size; ++j)
        for (std::size_t k = 0; k < wts.size(); ++k)
          _D(i, j, k, 0) = phi(0, k, i, 0) * wts[k] * tangent[j];
    }
  }

  return {pb, pshape, Db, Dshape};
}
//----------------------------------------------------------------------------
std::tuple<std::vector<std::vector<double>>, std::array<std::size_t, 2>,
           std::vector<std::vector<double>>, std::array<std::size_t, 4>>
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
  const auto [_pts, wts] = quadrature::make_quadrature<double>(
      quadrature::type::Default, sub_celltype, q_deg);
  cmdspan2_t pts(_pts.data(), wts.size(), _pts.size() / wts.size());

  // Evaluate moment space at quadrature points
  assert(std::accumulate(V.value_shape().begin(), V.value_shape().end(), 1,
                         std::multiplies{})
         == 1);
  const auto [phib, phishape] = V.tabulate(0, pts);
  cmdspan4_t phi(phib.data(), phishape);

  // Storage for coordinates of evaluations points in the reference cell
  const std::array<std::size_t, 2> pshape = {pts.extent(0), tdim};
  std::vector<std::vector<double>> pb;

  // Storage for interpolation matrix
  const std::array<std::size_t, 4> Dshape
      = {phi.extent(2), value_size, phi.extent(1), 1};
  const std::size_t size
      = std::reduce(Dshape.begin(), Dshape.end(), 1, std::multiplies{});
  std::vector<std::vector<double>> Db(num_entities, std::vector<double>(size));
  std::vector<mdspan4_t> D;

  // Evaluate moment space at quadrature points

  // Iterate over cell entities
  std::array<double, 3> normal;
  for (std::size_t e = 0; e < num_entities; ++e)
  {
    // Map quadrature points onto facet (cell entity e)
    const auto [ebuffer, eshape]
        = cell::sub_entity_geometry(celltype, tdim - 1, e);
    impl::cmdspan2_t facet_x(ebuffer.data(), eshape);

    auto& _pb = pb.emplace_back(pshape[0] * pshape[1]);
    mdspan2_t _p(_pb.data(), pshape);
    if (tdim == 2)
    {
      // No need to normalise the normal, as the size of this is equal
      // to the integral jacobian
      std::array<double, 2> tangent
          = {facet_x(1, 0) - facet_x(0, 0), facet_x(1, 1) - facet_x(0, 1)};
      for (std::size_t p = 0; p < _p.extent(0); ++p)
        for (std::size_t i = 0; i < _p.extent(1); ++i)
          _p(p, i) = facet_x(0, i) + pts(p, 0) * tangent[i];

      normal = {-tangent[1], tangent[0], 0.0};
    }
    else if (tdim == 3)
    {
      // No need to normalise the normal, as the size of this is equal
      // to the integral Jacobian
      std::array<double, 3> t0
          = {facet_x(1, 0) - facet_x(0, 0), facet_x(1, 1) - facet_x(0, 1),
             facet_x(1, 2) - facet_x(0, 2)};
      std::array<double, 3> t1
          = {facet_x(2, 0) - facet_x(0, 0), facet_x(2, 1) - facet_x(0, 1),
             facet_x(2, 2) - facet_x(0, 2)};
      for (std::size_t p = 0; p < _p.extent(0); ++p)
        for (std::size_t i = 0; i < _p.extent(1); ++i)
          _p(p, i) = facet_x(0, i) + pts(p, 0) * t0[i] + pts(p, 1) * t1[i];

      normal = basix::math::cross(t0, t1);
    }
    else
      throw std::runtime_error("Normal on this cell cannot be computed.");

    // Compute facet normal integral moments
    mdspan4_t& _D = D.emplace_back(Db[e].data(), Dshape);
    for (std::size_t i = 0; i < phi.extent(2); ++i)
      for (std::size_t j = 0; j < value_size; ++j)
        for (std::size_t k = 0; k < _D.extent(2); ++k)
          _D(i, j, k, 0) = phi(0, k, i, 0) * wts[k] * normal[j];
  }

  return {pb, pshape, Db, Dshape};
}
//----------------------------------------------------------------------------
