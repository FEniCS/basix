// Copyright (c) 2020-2022 Chris Richardson, Matthew Scroggs and Garth N. Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-lagrange.h"
#include "lattice.h"
#include "maps.h"
#include "math.h"
#include "mdspan.hpp"
#include "moments.h"
#include "polynomials.h"
#include "polyset.h"
#include "quadrature.h"
#include "sobolev-spaces.h"
#include <concepts>

using namespace basix;
namespace stdex
    = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;

namespace
{
//----------------------------------------------------------------------------
std::tuple<lattice::type, lattice::simplex_method, bool>
variant_to_lattice(cell::type celltype, element::lagrange_variant variant)
{
  switch (variant)
  {
  case element::lagrange_variant::equispaced:
    return {lattice::type::equispaced, lattice::simplex_method::none, true};
  case element::lagrange_variant::gll_warped:
    return {lattice::type::gll, lattice::simplex_method::warp, true};
  case element::lagrange_variant::gll_isaac:
    return {lattice::type::gll, lattice::simplex_method::isaac, true};
  case element::lagrange_variant::gll_centroid:
    return {lattice::type::gll, lattice::simplex_method::centroid, true};
  case element::lagrange_variant::chebyshev_warped:
  {
    if (celltype == cell::type::interval
        or celltype == cell::type::quadrilateral
        or celltype == cell::type::hexahedron)
    {
      return {lattice::type::chebyshev, lattice::simplex_method::none, false};
    }
    else
    {
      // TODO: is this the best thing to do for simplices?
      return {lattice::type::chebyshev_plus_endpoints,
              lattice::simplex_method::warp, false};
    }
  }
  case element::lagrange_variant::chebyshev_isaac:
  {
    if (celltype == cell::type::interval
        or celltype == cell::type::quadrilateral
        or celltype == cell::type::hexahedron)
    {
      return {lattice::type::chebyshev, lattice::simplex_method::none, false};
    }
    else
    {
      // TODO: is this the best thing to do for simplices?
      return {lattice::type::chebyshev_plus_endpoints,
              lattice::simplex_method::isaac, false};
    }
  }
  case element::lagrange_variant::chebyshev_centroid:
    return {lattice::type::chebyshev, lattice::simplex_method::centroid, false};
  case element::lagrange_variant::gl_warped:
  {
    if (celltype == cell::type::interval
        or celltype == cell::type::quadrilateral
        or celltype == cell::type::hexahedron)
    {
      return {lattice::type::gl, lattice::simplex_method::none, false};
    }
    else
    {
      // TODO: is this the best thing to do for simplices?
      return {lattice::type::gl_plus_endpoints, lattice::simplex_method::warp,
              false};
    }
  }
  case element::lagrange_variant::gl_isaac:
  {
    if (celltype == cell::type::interval
        or celltype == cell::type::quadrilateral
        or celltype == cell::type::hexahedron)
    {
      return {lattice::type::gl, lattice::simplex_method::none, false};
    }
    else
    {
      // TODO: is this the best thing to do for simplices?
      return {lattice::type::gl_plus_endpoints, lattice::simplex_method::isaac,
              false};
    }
  }
  case element::lagrange_variant::gl_centroid:
    return {lattice::type::gl, lattice::simplex_method::centroid, false};
  default:
    throw std::runtime_error("Unsupported variant");
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T> create_d_lagrange(cell::type celltype, int degree,
                                   element::lagrange_variant variant,
                                   lattice::type lattice_type,
                                   lattice::simplex_method simplex_method)
{
  if (celltype == cell::type::prism or celltype == cell::type::pyramid)
  {
    throw std::runtime_error(
        "This variant is not yet supported on prisms and pyramids.");
  }

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t ndofs
      = polyset::dim(celltype, polyset::type::standard, degree);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;
  for (std::size_t i = 0; i < tdim; ++i)
  {
    std::size_t num_ent = cell::num_sub_entities(celltype, i);
    x[i] = std::vector<impl::mdarray_t<T, 2>>(num_ent,
                                              impl::mdarray_t<T, 2>(0, tdim));
    M[i] = std::vector<impl::mdarray_t<T, 4>>(
        num_ent, impl::mdarray_t<T, 4>(0, 1, 0, 1));
  }

  const int lattice_degree
      = celltype == cell::type::triangle
            ? degree + 3
            : (celltype == cell::type::tetrahedron ? degree + 4 : degree + 2);

  // Create points in interior
  const auto [pt, shape] = lattice::create<T>(
      celltype, lattice_degree, lattice_type, false, simplex_method);
  x[tdim].emplace_back(shape, pt);

  const std::size_t num_dofs = shape[0];
  auto& _M = M[tdim].emplace_back(num_dofs, 1, num_dofs, 1);
  for (std::size_t i = 0; i < _M.extent(0); ++i)
    _M[i, 0, i, 0] = 1.0;

  return FiniteElement(
      element::family::P, celltype, polyset::type::standard, degree, {},
      impl::mdspan_t<const T, 2>(math::eye<T>(ndofs).data(), ndofs, ndofs),
      impl::to_mdspan(x), impl::to_mdspan(M), 0, maps::type::identity,
      sobolev::space::L2, true, degree, degree, variant,
      element::dpc_variant::unset);
}
//----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T>
create_d_iso(cell::type celltype, int degree, element::lagrange_variant variant,
             lattice::type lattice_type, lattice::simplex_method simplex_method)
{
  if (celltype == cell::type::prism or celltype == cell::type::pyramid)
  {
    throw std::runtime_error(
        "This variant is not yet supported on prisms and pyramids.");
  }

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t ndofs
      = polyset::dim(celltype, polyset::type::macroedge, degree);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;
  for (std::size_t i = 0; i < tdim; ++i)
  {
    std::size_t num_ent = cell::num_sub_entities(celltype, i);
    x[i] = std::vector<impl::mdarray_t<T, 2>>(num_ent,
                                              impl::mdarray_t<T, 2>(0, tdim));
    M[i] = std::vector<impl::mdarray_t<T, 4>>(
        num_ent, impl::mdarray_t<T, 4>(0, 1, 0, 1));
  }

  const int lattice_degree
      = celltype == cell::type::triangle
            ? 2 * degree + 3
            : (celltype == cell::type::tetrahedron ? 2 * degree + 4
                                                   : 2 * degree + 2);

  // Create points in interior
  const auto [pt, shape] = lattice::create<T>(
      celltype, lattice_degree, lattice_type, false, simplex_method);
  x[tdim].emplace_back(shape, pt);

  const std::size_t num_dofs = shape[0];
  auto& _M = M[tdim].emplace_back(num_dofs, 1, num_dofs, 1);
  for (std::size_t i = 0; i < _M.extent(0); ++i)
    _M[i, 0, i, 0] = 1.0;

  return FiniteElement(
      element::family::iso, celltype, polyset::type::macroedge, degree, {},
      impl::mdspan_t<const T, 2>(math::eye<T>(ndofs).data(), ndofs, ndofs),
      impl::to_mdspan(x), impl::to_mdspan(M), 0, maps::type::identity,
      sobolev::space::L2, true, degree, degree, variant,
      element::dpc_variant::unset);
}
//----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T> create_legendre(cell::type celltype, int degree,
                                 bool discontinuous)
{
  if (!discontinuous)
    throw std::runtime_error("Legendre variant must be discontinuous");

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t ndofs
      = polyset::dim(celltype, polyset::type::standard, degree);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;

  // Evaluate moment space at quadrature points
  const auto [_pts, wts] = quadrature::make_quadrature<T>(
      quadrature::type::Default, celltype, polyset::type::standard, degree * 2);
  assert(!wts.empty());
  impl::mdspan_t<const T, 2> pts(_pts.data(), wts.size(),
                                 _pts.size() / wts.size());
  const auto [_phi, pshape] = polynomials::tabulate(polynomials::type::legendre,
                                                    celltype, degree, pts);
  impl::mdspan_t<const T, 2> phi(_phi.data(), pshape);
  for (std::size_t d = 0; d < tdim; ++d)
  {
    for (std::size_t e = 0; e < topology[d].size(); ++e)
    {
      x[d].emplace_back(0, tdim);
      M[d].emplace_back(0, 1, 0, 1);
    }
  }

  auto& _x = x[tdim].emplace_back(pts.extents());
  std::copy_n(pts.data_handle(), pts.size(), _x.data());
  auto& _M = M[tdim].emplace_back(ndofs, 1, pts.extent(0), 1);
  for (std::size_t i = 0; i < ndofs; ++i)
    for (std::size_t j = 0; j < pts.extent(0); ++j)
      _M[i, 0, j, 0] = phi[i, j] * wts[j];

  sobolev::space space
      = discontinuous ? sobolev::space::L2 : sobolev::space::H1;
  return FiniteElement<T>(
      element::family::P, celltype, polyset::type::standard, degree, {},
      impl::mdspan_t<T, 2>(math::eye<T>(ndofs).data(), ndofs, ndofs),
      impl::to_mdspan(x), impl::to_mdspan(M), 0, maps::type::identity, space,
      discontinuous, degree, degree, element::lagrange_variant::legendre,
      element::dpc_variant::unset);
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T> create_bernstein(cell::type celltype, int degree,
                                  bool discontinuous)
{
  assert(degree > 0);
  if (celltype != cell::type::interval and celltype != cell::type::triangle
      and celltype != cell::type::tetrahedron)
  {
    throw std::runtime_error(
        "Bernstein elements are currently only supported on simplices.");
  }

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;

  const std::array<std::size_t, 4> nb
      = {1,
         static_cast<std::size_t>(polynomials::dim(
             polynomials::type::bernstein, cell::type::interval, degree)),
         static_cast<std::size_t>(polynomials::dim(
             polynomials::type::bernstein, cell::type::triangle, degree)),
         static_cast<std::size_t>(polynomials::dim(
             polynomials::type::bernstein, cell::type::tetrahedron, degree))};

  constexpr std::array<cell::type, 4> ct
      = {cell::type::point, cell::type::interval, cell::type::triangle,
         cell::type::tetrahedron};

  const std::array<std::size_t, 4> nb_interior
      = {1, degree < 2 ? 0 : nb[1] - 2, degree < 3 ? 0 : nb[2] + 3 - 3 * nb[1],
         degree < 4 ? 0 : nb[3] + 6 * nb[1] - 4 * nb[2] - 4};

  std::array<std::vector<int>, 4> bernstein_bubbles;
  bernstein_bubbles[0].push_back(0);
  { // scope
    int ib = 0;
    for (int i = 0; i <= degree; ++i)
    {
      if (i > 0 and i < degree)
      {
        bernstein_bubbles[1].push_back(ib);
      }
      ++ib;
    }
  }
  { // scope
    int ib = 0;
    for (int i = 0; i <= degree; ++i)
    {
      for (int j = 0; j <= degree - i; ++j)
      {
        if (i > 0 and j > 0 and i + j < degree)
          bernstein_bubbles[2].push_back(ib);
        ++ib;
      }
    }
  }
  { // scope
    int ib = 0;
    for (int i = 0; i <= degree; ++i)
    {
      for (int j = 0; j <= degree - i; ++j)
      {
        for (int k = 0; k <= degree - i - j; ++k)
        {
          if (i > 0 and j > 0 and k > 0 and i + j + k < degree)
            bernstein_bubbles[3].push_back(ib);
          ++ib;
        }
      }
    }
  }

  for (std::size_t v = 0; v < topology[0].size(); ++v)
  {
    const auto [entity, shape] = cell::sub_entity_geometry<T>(celltype, 0, v);
    x[0].emplace_back(shape, entity);
    M[0].emplace_back(std::array<std::size_t, 4>{1, 1, 1, 1}, 1);
  }

  for (std::size_t d = 1; d <= tdim; ++d)
  {
    if (nb_interior[d] == 0)
    {
      for (std::size_t e = 0; e < topology[d].size(); ++e)
      {
        x[d].emplace_back(0, tdim);
        M[d].emplace_back(0, 1, 0, 1);
      }
    }
    else
    {
      const auto [_pts, wts]
          = quadrature::make_quadrature<T>(quadrature::type::Default, ct[d],
                                           polyset::type::standard, degree * 2);
      assert(!wts.empty());
      impl::mdspan_t<const T, 2> pts(_pts.data(), wts.size(),
                                     _pts.size() / wts.size());

      const auto [_phi, pshape] = polynomials::tabulate(
          polynomials::type::legendre, ct[d], degree, pts);
      impl::mdspan_t<const T, 2> phi(_phi.data(), pshape);
      const auto [_bern, bshape] = polynomials::tabulate(
          polynomials::type::bernstein, ct[d], degree, pts);
      impl::mdspan_t<const T, 2> bern(_bern.data(), bshape);

      assert(phi.extent(0) == nb[d]);
      const std::size_t npts = pts.extent(0);

      impl::mdarray_t<T, 2> mat(nb[d], nb[d]);
      for (std::size_t i = 0; i < nb[d]; ++i)
        for (std::size_t j = 0; j < nb[d]; ++j)
          for (std::size_t k = 0; k < wts.size(); ++k)
            mat[i, j] += wts[k] * bern[j, k] * phi[i, k];

      impl::mdarray_t<T, 2> minv(mat.extents());
      {
        std::vector<T> id = math::eye<T>(nb[d]);
        impl::mdspan_t<T, 2> _id(id.data(), nb[d], nb[d]);
        impl::mdspan_t<T, 2> _mat(mat.data(), mat.extents());
        std::vector<T> minv_data = math::solve<T>(_mat, _id);
        std::ranges::copy(minv_data, minv.data());
      }

      M[d] = std::vector<impl::mdarray_t<T, 4>>(
          cell::num_sub_entities(celltype, d),
          impl::mdarray_t<T, 4>(nb_interior[d], 1, npts, 1));
      for (std::size_t e = 0; e < topology[d].size(); ++e)
      {
        auto [_entity_x, shape] = cell::sub_entity_geometry<T>(celltype, d, e);
        impl::mdspan_t<T, 2> entity_x(_entity_x.data(), shape);
        std::span<const T> x0(entity_x.data_handle(), shape[1]);
        {
          auto& _x = x[d].emplace_back(pts.extent(0), shape[1]);
          for (std::size_t i = 0; i < _x.extent(0); ++i)
            for (std::size_t j = 0; j < _x.extent(1); ++j)
              _x[i, j] = x0[j];
        }

        for (std::size_t j = 0; j < pts.extent(0); ++j)
          for (std::size_t k0 = 0; k0 < pts.extent(1); ++k0)
            for (std::size_t k1 = 0; k1 < shape[1]; ++k1)
              x[d][e][j, k1] += (entity_x[k0 + 1, k1] - x0[k1]) * pts[j, k0];
        for (std::size_t i = 0; i < bernstein_bubbles[d].size(); ++i)
        {
          for (std::size_t p = 0; p < npts; ++p)
          {
            T tmp = 0.0;
            for (std::size_t k = 0; k < phi.extent(0); ++k)
              tmp += phi[k, p] * minv[bernstein_bubbles[d][i], k];
            M[d][e][i, 0, p, 0] = wts[p] * tmp;
          }
        }
      }
    }
  }

  sobolev::space space
      = discontinuous ? sobolev::space::L2 : sobolev::space::H1;
  const std::size_t ndofs
      = polyset::dim(celltype, polyset::type::standard, degree);
  return FiniteElement<T>(
      element::family::P, celltype, polyset::type::standard, degree, {},
      impl::mdspan_t<T, 2>(math::eye<T>(ndofs).data(), ndofs, ndofs),
      impl::to_mdspan(x), impl::to_mdspan(M), 0, maps::type::identity, space,
      discontinuous, degree, degree, element::lagrange_variant::bernstein,
      element::dpc_variant::unset);
}
//-----------------------------------------------------------------------------
} // namespace

//----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T>
basix::element::create_lagrange(cell::type celltype, int degree,
                                lagrange_variant variant, bool discontinuous,
                                std::vector<int> dof_ordering)
{
  if (celltype == cell::type::point)
  {
    if (degree != 0)
      throw std::runtime_error("Can only create order 0 Lagrange on a point");

    std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
    std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;
    x[0].emplace_back(1, 0);
    M[0].emplace_back(
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>{1, 1, 1, 1},
        1);
    return FiniteElement<T>(
        family::P, cell::type::point, polyset::type::standard, 0, {},
        impl::mdspan_t<T, 2>(math::eye<T>(1).data(), 1, 1), impl::to_mdspan(x),
        impl::to_mdspan(M), 0, maps::type::identity, sobolev::space::H1,
        discontinuous, degree, degree, element::lagrange_variant::unset,
        element::dpc_variant::unset, dof_ordering);
  }

  if (variant == lagrange_variant::legendre)
    return create_legendre<T>(celltype, degree, discontinuous);

  if (variant == element::lagrange_variant::bernstein)
  {
    if (degree == 0)
      variant = lagrange_variant::unset;
    else
      return create_bernstein<T>(celltype, degree, discontinuous);
  }

  if (variant == lagrange_variant::unset)
  {
    if (degree < 3)
      variant = element::lagrange_variant::gll_warped;
    else
    {
      throw std::runtime_error(
          "Lagrange elements of degree > 2 need to be given a variant.");
    }
  }

  auto [lattice_type, simplex_method, exterior]
      = variant_to_lattice(celltype, variant);

  if (!exterior)
  {
    // Points used to define this variant are all interior to the cell,
    // so this variant requires that the element is discontinuous
    if (!discontinuous)
    {
      throw std::runtime_error("This variant of Lagrange is only supported for "
                               "discontinuous elements");
    }
    return create_d_lagrange<T>(celltype, degree, variant, lattice_type,
                                simplex_method);
  }

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t ndofs
      = polyset::dim(celltype, polyset::type::standard, degree);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;
  if (degree == 0)
  {
    if (!discontinuous)
    {
      throw std::runtime_error(
          "Cannot create a continuous order 0 Lagrange basis function");
    }

    for (std::size_t i = 0; i < tdim; ++i)
    {
      std::size_t num_entities = cell::num_sub_entities(celltype, i);
      x[i] = std::vector(num_entities, impl::mdarray_t<T, 2>(0, tdim));
      M[i] = std::vector(num_entities, impl::mdarray_t<T, 4>(0, 1, 0, 1));
    }

    const auto [pt, shape]
        = lattice::create<T>(celltype, 0, lattice_type, true, simplex_method);
    x[tdim].emplace_back(
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>{shape[0],
                                                                 shape[1]},
        pt);
    auto& _M = M[tdim].emplace_back(shape[0], 1, shape[0], 1);
    std::fill(_M.data(), _M.data() + _M.size(), 0);
    for (std::size_t i = 0; i < shape[0]; ++i)
      _M[i, 0, i, 0] = 1;
  }
  else
  {
    // Create points at nodes, ordered by topology (vertices first)
    for (std::size_t dim = 0; dim <= tdim; ++dim)
    {
      // Loop over entities of dimension 'dim'
      for (std::size_t e = 0; e < topology[dim].size(); ++e)
      {
        const auto [entity_x, entity_x_shape]
            = cell::sub_entity_geometry<T>(celltype, dim, e);
        if (dim == 0)
        {
          x[dim].emplace_back(
              MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>{
                  entity_x_shape[0], entity_x_shape[1]},
              entity_x);
          auto& _M
              = M[dim].emplace_back(entity_x_shape[0], 1, entity_x_shape[0], 1);
          std::fill(_M.data(), _M.data() + _M.size(), 0);
          for (std::size_t i = 0; i < entity_x_shape[0]; ++i)
            _M[i, 0, i, 0] = 1;
        }
        else if (dim == tdim)
        {
          const auto [pt, shape] = lattice::create<T>(
              celltype, degree, lattice_type, false, simplex_method);
          x[dim].emplace_back(
              MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>{
                  shape[0], shape[1]},
              pt);
          auto& _M = M[dim].emplace_back(shape[0], 1, shape[0], 1);
          std::fill(_M.data(), _M.data() + _M.size(), 0);
          for (std::size_t i = 0; i < shape[0]; ++i)
            _M[i, 0, i, 0] = 1;
        }
        else
        {
          cell::type ct = cell::sub_entity_type(celltype, dim, e);
          const auto [pt, shape] = lattice::create<T>(ct, degree, lattice_type,
                                                      false, simplex_method);
          impl::mdspan_t<const T, 2> lattice(pt.data(), shape);
          std::span<const T> x0(entity_x.data(), entity_x_shape[1]);
          impl::mdspan_t<const T, 2> entity_x_view(entity_x.data(),
                                                   entity_x_shape);

          auto& _x = x[dim].emplace_back(shape[0], entity_x_shape[1]);
          for (std::size_t i = 0; i < shape[0]; ++i)
            for (std::size_t j = 0; j < entity_x_shape[1]; ++j)
              _x[i, j] = x0[j];

          for (std::size_t j = 0; j < shape[0]; ++j)
            for (std::size_t k = 0; k < shape[1]; ++k)
              for (std::size_t q = 0; q < tdim; ++q)
                _x[j, q] += (entity_x_view[k + 1, q] - x0[q]) * lattice[j, k];

          auto& _M = M[dim].emplace_back(shape[0], 1, shape[0], 1);
          std::fill(_M.data(), _M.data() + _M.size(), 0);
          for (std::size_t i = 0; i < shape[0]; ++i)
            _M[i, 0, i, 0] = 1;
        }
      }
    }
  }

  std::array<std::vector<mdspan_t<const T, 2>>, 4> xview = impl::to_mdspan(x);
  std::array<std::vector<mdspan_t<const T, 4>>, 4> Mview = impl::to_mdspan(M);
  std::array<std::vector<std::vector<T>>, 4> xbuffer;
  std::array<std::vector<std::vector<T>>, 4> Mbuffer;
  if (discontinuous)
  {
    std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
    std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
    std::tie(xbuffer, xshape, Mbuffer, Mshape)
        = make_discontinuous(xview, Mview, tdim, 1);
    xview = impl::to_mdspan(xbuffer, xshape);
    Mview = impl::to_mdspan(Mbuffer, Mshape);
  }

  sobolev::space space
      = discontinuous ? sobolev::space::L2 : sobolev::space::H1;
  return FiniteElement<T>(
      family::P, celltype, polyset::type::standard, degree, {},
      impl::mdspan_t<T, 2>(math::eye<T>(ndofs).data(), ndofs, ndofs), xview,
      Mview, 0, maps::type::identity, space, discontinuous, degree, degree,
      variant, dpc_variant::unset, dof_ordering);
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T> basix::element::create_iso(cell::type celltype, int degree,
                                            lagrange_variant variant,
                                            bool discontinuous)
{
  if (celltype != cell::type::interval && celltype != cell::type::quadrilateral
      && celltype != cell::type::hexahedron && celltype != cell::type::triangle
      && celltype != cell::type::tetrahedron)
  {
    throw std::runtime_error(
        "Can currently only create iso elements on "
        "intervals, triangles, tetrahedra, quadrilaterals, and hexahedra");
  }

  if (variant == lagrange_variant::unset)
  {
    if (degree < 3)
      variant = element::lagrange_variant::gll_warped;
    else
    {
      throw std::runtime_error(
          "Lagrange elements of degree > 2 need to be given a variant.");
    }
  }

  auto [lattice_type, simplex_method, exterior]
      = variant_to_lattice(celltype, variant);

  if (!exterior)
  {
    // Points used to define this variant are all interior to the cell,
    // so this variant requires that the element is discontinuous
    if (!discontinuous)
    {
      throw std::runtime_error("This variant of Lagrange is only supported for "
                               "discontinuous elements");
    }
    return create_d_iso<T>(celltype, degree, variant, lattice_type,
                           simplex_method);
  }

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t ndofs
      = polyset::dim(celltype, polyset::type::macroedge, degree);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;
  if (degree == 0)
  {
    if (!discontinuous)
    {
      throw std::runtime_error(
          "Cannot create a continuous order 0 Lagrange basis function");
    }

    for (std::size_t i = 0; i < tdim; ++i)
    {
      std::size_t num_entities = cell::num_sub_entities(celltype, i);
      x[i] = std::vector(num_entities, impl::mdarray_t<T, 2>(0, tdim));
      M[i] = std::vector(num_entities, impl::mdarray_t<T, 4>(0, 1, 0, 1));
    }

    const auto [pt, shape]
        = lattice::create<T>(celltype, 0, lattice_type, true, simplex_method);
    x[tdim].emplace_back(std::array<std::size_t, 2>{shape[0], shape[1]}, pt);
    auto& _M = M[tdim].emplace_back(
        std::array<std::size_t, 4>{shape[0], 1, shape[0], 1}, 0);
    for (std::size_t i = 0; i < shape[0]; ++i)
      _M[i, 0, i, 0] = 1;
  }
  else
  {
    // Create points at nodes, ordered by topology (vertices first)
    for (std::size_t dim = 0; dim <= tdim; ++dim)
    {
      // Loop over entities of dimension 'dim'
      for (std::size_t e = 0; e < topology[dim].size(); ++e)
      {
        const auto [entity_x, entity_x_shape]
            = cell::sub_entity_geometry<T>(celltype, dim, e);
        if (dim == 0)
        {
          x[dim].emplace_back(entity_x_shape, entity_x);
          auto& _M = M[dim].emplace_back(
              std::array<std::size_t, 4>{entity_x_shape[0], 1,
                                         entity_x_shape[0], 1},
              0);
          for (std::size_t i = 0; i < entity_x_shape[0]; ++i)
            _M[i, 0, i, 0] = 1;
        }
        else if (dim == tdim)
        {
          const auto [pt, shape] = lattice::create<T>(
              celltype, 2 * degree, lattice_type, false, simplex_method);
          x[dim].emplace_back(shape, pt);
          auto& _M = M[dim].emplace_back(
              std::array<std::size_t, 4>{shape[0], 1, shape[0], 1}, 0);
          for (std::size_t i = 0; i < shape[0]; ++i)
            _M[i, 0, i, 0] = 1;
        }
        else
        {
          cell::type ct = cell::sub_entity_type(celltype, dim, e);
          const auto [pt, shape] = lattice::create<T>(
              ct, 2 * degree, lattice_type, false, simplex_method);
          impl::mdspan_t<const T, 2> lattice(pt.data(), shape);
          std::span<const T> x0(entity_x.data(), entity_x_shape[1]);
          impl::mdspan_t<const T, 2> entity_x_view(entity_x.data(),
                                                   entity_x_shape);

          auto& _x = x[dim].emplace_back(shape[0], entity_x_shape[1]);
          for (std::size_t i = 0; i < shape[0]; ++i)
            for (std::size_t j = 0; j < entity_x_shape[1]; ++j)
              _x[i, j] = x0[j];

          for (std::size_t j = 0; j < shape[0]; ++j)
            for (std::size_t k = 0; k < shape[1]; ++k)
              for (std::size_t q = 0; q < tdim; ++q)
                _x[j, q] += (entity_x_view[k + 1, q] - x0[q]) * lattice[j, k];

          auto& _M = M[dim].emplace_back(shape[0], 1, shape[0], 1);
          std::fill(_M.data(), _M.data() + _M.size(), 0);
          for (std::size_t i = 0; i < shape[0]; ++i)
            _M[i, 0, i, 0] = 1;
        }
      }
    }
  }

  std::array<std::vector<mdspan_t<const T, 2>>, 4> xview = impl::to_mdspan(x);
  std::array<std::vector<mdspan_t<const T, 4>>, 4> Mview = impl::to_mdspan(M);
  std::array<std::vector<std::vector<T>>, 4> xbuffer;
  std::array<std::vector<std::vector<T>>, 4> Mbuffer;
  if (discontinuous)
  {
    std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
    std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
    std::tie(xbuffer, xshape, Mbuffer, Mshape)
        = make_discontinuous(xview, Mview, tdim, 1);
    xview = impl::to_mdspan(xbuffer, xshape);
    Mview = impl::to_mdspan(Mbuffer, Mshape);
  }

  sobolev::space space
      = discontinuous ? sobolev::space::L2 : sobolev::space::H1;
  return FiniteElement<T>(
      family::iso, celltype, polyset::type::macroedge, degree, {},
      impl::mdspan_t<T, 2>(math::eye<T>(ndofs).data(), ndofs, ndofs), xview,
      Mview, 0, maps::type::identity, space, discontinuous, degree, degree,
      variant, dpc_variant::unset);
}
//-----------------------------------------------------------------------------
template FiniteElement<float> element::create_lagrange(cell::type, int,
                                                       lagrange_variant, bool,
                                                       std::vector<int>);
template FiniteElement<double> element::create_lagrange(cell::type, int,
                                                        lagrange_variant, bool,
                                                        std::vector<int>);
template FiniteElement<float> element::create_iso(cell::type, int,
                                                  lagrange_variant, bool);
template FiniteElement<double> element::create_iso(cell::type, int,
                                                   lagrange_variant, bool);
//-----------------------------------------------------------------------------
