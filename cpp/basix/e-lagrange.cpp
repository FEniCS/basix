// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-lagrange.h"
#include "dof-transformations.h"
#include "element-families.h"
#include "log.h"
#include "maps.h"
#include "polyset.h"
#include "quadrature.h"
#include <numeric>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

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
  case element::lagrange_variant::chebyshev_warped:
  {
    if (celltype == cell::type::interval
        or celltype == cell::type::quadrilateral
        or celltype == cell::type::hexahedron)
      return {lattice::type::chebyshev, lattice::simplex_method::none, false};
    // TODO: is this the best thing to do for simplices?
    return {lattice::type::chebyshev_plus_endpoints,
            lattice::simplex_method::warp, false};
  }
  case element::lagrange_variant::chebyshev_isaac:
  {
    if (celltype == cell::type::interval
        or celltype == cell::type::quadrilateral
        or celltype == cell::type::hexahedron)
      return {lattice::type::chebyshev, lattice::simplex_method::none, false};
    // TODO: is this the best thing to do for simplices?
    return {lattice::type::chebyshev_plus_endpoints,
            lattice::simplex_method::isaac, false};
  }
  default:
    throw std::runtime_error("Unsupported variant");
  }
}
//-----------------------------------------------------------------------------
FiniteElement create_d_lagrange(cell::type celltype, int degree,
                                lattice::type lattice_type,
                                lattice::simplex_method simplex_method)
{
  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t ndofs = polyset::dim(celltype, degree);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  if (celltype == cell::type::prism or celltype == cell::type::pyramid)
  {
    throw std::runtime_error(
        "This variant is not yet supported on prisms and pyramids.");
  }

  const int lattice_degree
      = celltype == cell::type::triangle
            ? degree + 3
            : (celltype == cell::type::tetrahedron ? degree + 4 : degree + 2);

  // Create points in interior
  auto pt = lattice::create(celltype, lattice_degree, lattice_type, false,
                            simplex_method);
  x[tdim].push_back(pt);
  const std::size_t num_dofs = pt.shape(0);
  std::array<std::size_t, 3> s = {num_dofs, 1, num_dofs};
  M[tdim].push_back(xt::xtensor<double, 3>(s));
  xt::view(M[tdim][0], xt::all(), 0, xt::all()) = xt::eye<double>(num_dofs);

  std::map<cell::type, xt::xtensor<double, 3>> entity_transformations;

  // Entity transformations for edges
  if (tdim > 1)
  {
    const std::array<std::size_t, 3> shape = {1, 0, 0};
    xt::xtensor<double, 3> et = xt::zeros<double>(shape);
    entity_transformations[cell::type::interval] = et;
  }

  // Entity transformations for triangular faces
  if (celltype == cell::type::tetrahedron or celltype == cell::type::prism
      or celltype == cell::type::pyramid)
  {
    const std::array<std::size_t, 3> shape = {2, 0, 0};
    xt::xtensor<double, 3> ft = xt::zeros<double>(shape);
    entity_transformations[cell::type::triangle] = ft;
  }

  // Entity transformations for quadrilateral faces
  if (celltype == cell::type::hexahedron or celltype == cell::type::prism
      or celltype == cell::type::pyramid)
  {
    const std::array<std::size_t, 3> shape = {2, 0, 0};
    xt::xtensor<double, 3> ft = xt::zeros<double>(shape);
    entity_transformations[cell::type::quadrilateral] = ft;
  }

  xt::xtensor<double, 3> coeffs = compute_expansion_coefficients(
      celltype, xt::eye<double>(ndofs), {M[0], M[1], M[2], M[3]},
      {x[0], x[1], x[2], x[3]}, degree);
  return FiniteElement(element::family::P, celltype, degree, {1}, coeffs,
                       entity_transformations, x, M, maps::type::identity,
                       true);
}
//-----------------------------------------------------------------------------
} // namespace

//----------------------------------------------------------------------------
FiniteElement basix::create_lagrange(cell::type celltype, int degree,
                                     element::lagrange_variant variant,
                                     bool discontinuous)
{
  if (celltype == cell::type::point)
    throw std::runtime_error("Invalid celltype");

  auto [lattice_type, simplex_method, exterior]
      = variant_to_lattice(celltype, variant);

  if (!exterior)
  {
    // Points used to define this variant are all interior to the cell, so this
    // variant required that the element is discontinuous
    if (!discontinuous)
    {
      throw std::runtime_error("This variant of Lagrange is only supported for "
                               "discontinuous elements");
    }
    return create_d_lagrange(celltype, degree, lattice_type, simplex_method);
  }

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t ndofs = polyset::dim(celltype, degree);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  // Create points at nodes, ordered by topology (vertices first)
  if (degree == 0)
  {
    if (!discontinuous)
    {
      throw std::runtime_error(
          "Cannot create a continuous order 0 Lagrange basis function");
    }
    auto pt = lattice::create(celltype, 0, lattice_type, true, simplex_method);
    x[tdim].push_back(pt);
    const std::size_t num_dofs = pt.shape(0);
    std::array<std::size_t, 3> s = {num_dofs, 1, num_dofs};
    M[tdim].push_back(xt::xtensor<double, 3>(s));
    xt::view(M[tdim][0], xt::all(), 0, xt::all()) = xt::eye<double>(num_dofs);
  }
  else
  {
    for (std::size_t dim = 0; dim <= tdim; ++dim)
    {
      M[dim].resize(topology[dim].size());
      x[dim].resize(topology[dim].size());

      // Loop over entities of dimension 'dim'
      for (std::size_t e = 0; e < topology[dim].size(); ++e)
      {
        const xt::xtensor<double, 2> entity_x
            = cell::sub_entity_geometry(celltype, dim, e);
        if (dim == 0)
        {
          x[dim][e] = entity_x;
          const std::size_t num_dofs = entity_x.shape(0);
          M[dim][e] = xt::xtensor<double, 3>(
              {num_dofs, static_cast<std::size_t>(1), num_dofs});
          xt::view(M[dim][e], xt::all(), 0, xt::all())
              = xt::eye<double>(num_dofs);
        }
        else if (dim == tdim)
        {
          x[dim][e] = lattice::create(celltype, degree, lattice_type, false,
                                      simplex_method);
          const std::size_t num_dofs = x[dim][e].shape(0);
          std::array<std::size_t, 3> s = {num_dofs, 1, num_dofs};
          M[dim][e] = xt::xtensor<double, 3>(s);
          xt::view(M[dim][e], xt::all(), 0, xt::all())
              = xt::eye<double>(num_dofs);
        }
        else
        {
          cell::type ct = cell::sub_entity_type(celltype, dim, e);
          const auto lattice = lattice::create(ct, degree, lattice_type, false,
                                               simplex_method);
          const std::size_t num_dofs = lattice.shape(0);
          std::array<std::size_t, 3> s = {num_dofs, 1, num_dofs};
          M[dim][e] = xt::xtensor<double, 3>(s);
          xt::view(M[dim][e], xt::all(), 0, xt::all())
              = xt::eye<double>(num_dofs);

          auto x0s = xt::reshape_view(
              xt::row(entity_x, 0),
              {static_cast<std::size_t>(1), entity_x.shape(1)});
          x[dim][e] = xt::tile(x0s, lattice.shape(0));
          auto x0 = xt::row(entity_x, 0);
          for (std::size_t j = 0; j < lattice.shape(0); ++j)
          {
            for (std::size_t k = 0; k < lattice.shape(1); ++k)
            {
              xt::row(x[dim][e], j)
                  += (xt::row(entity_x, k + 1) - x0) * lattice(j, k);
            }
          }
        }
      }
    }
  }

  std::map<cell::type, xt::xtensor<double, 3>> entity_transformations;

  // Entity transformations for edges
  if (tdim > 1)
  {
    const std::vector<int> edge_ref
        = doftransforms::interval_reflection(degree - 1);
    const std::array<std::size_t, 3> shape
        = {1, edge_ref.size(), edge_ref.size()};
    xt::xtensor<double, 3> et = xt::zeros<double>(shape);
    for (std::size_t i = 0; i < edge_ref.size(); ++i)
      et(0, i, edge_ref[i]) = 1;
    entity_transformations[cell::type::interval] = et;
  }

  // Entity transformations for triangular faces
  if (celltype == cell::type::tetrahedron or celltype == cell::type::prism
      or celltype == cell::type::pyramid)
  {
    const std::vector<int> face_rot
        = doftransforms::triangle_rotation(degree - 2);
    const std::vector<int> face_ref
        = doftransforms::triangle_reflection(degree - 2);
    const std::array<std::size_t, 3> shape
        = {2, face_rot.size(), face_rot.size()};
    xt::xtensor<double, 3> ft = xt::zeros<double>(shape);
    for (std::size_t i = 0; i < face_rot.size(); ++i)
    {
      ft(0, i, face_rot[i]) = 1;
      ft(1, i, face_ref[i]) = 1;
    }
    entity_transformations[cell::type::triangle] = ft;
  }

  // Entity transformations for quadrilateral faces
  if (celltype == cell::type::hexahedron or celltype == cell::type::prism
      or celltype == cell::type::pyramid)
  {
    const std::vector<int> face_rot
        = doftransforms::quadrilateral_rotation(degree - 1);
    const std::vector<int> face_ref
        = doftransforms::quadrilateral_reflection(degree - 1);
    const std::array<std::size_t, 3> shape
        = {2, face_rot.size(), face_rot.size()};
    xt::xtensor<double, 3> ft = xt::zeros<double>(shape);
    for (std::size_t i = 0; i < face_rot.size(); ++i)
    {
      ft(0, i, face_rot[i]) = 1;
      ft(1, i, face_ref[i]) = 1;
    }
    entity_transformations[cell::type::quadrilateral] = ft;
  }

  if (discontinuous)
  {
    std::tie(x, M, entity_transformations)
        = make_discontinuous(x, M, entity_transformations, tdim, 1);
  }

  xt::xtensor<double, 3> coeffs = compute_expansion_coefficients(
      celltype, xt::eye<double>(ndofs), {M[0], M[1], M[2], M[3]},
      {x[0], x[1], x[2], x[3]}, degree);
  return FiniteElement(element::family::P, celltype, degree, {1}, coeffs,
                       entity_transformations, x, M, maps::type::identity,
                       discontinuous);
}
//-----------------------------------------------------------------------------
FiniteElement basix::create_dpc(cell::type celltype, int degree,
                                bool discontinuous)
{
  // Only tabulate for scalar. Vector spaces can easily be built from
  // the scalar space.
  if (!discontinuous)
  {
    throw std::runtime_error("Cannot create a continuous DPC element.");
  }

  cell::type simplex_type;
  switch (celltype)
  {
  case cell::type::interval:
    simplex_type = cell::type::interval;
    break;
  case cell::type::quadrilateral:
    simplex_type = cell::type::triangle;
    break;
  case cell::type::hexahedron:
    simplex_type = cell::type::tetrahedron;
    break;
  default:
    throw std::runtime_error("Invalid cell type");
  }

  const std::size_t ndofs = polyset::dim(simplex_type, degree);
  const std::size_t psize = polyset::dim(celltype, degree);

  auto [pts, _wts]
      = quadrature::make_quadrature("default", celltype, 2 * degree);
  auto wts = xt::adapt(_wts);

  xt::xtensor<double, 2> psi_quad = xt::view(
      polyset::tabulate(celltype, degree, 0, pts), 0, xt::all(), xt::all());
  xt::xtensor<double, 2> psi = xt::view(
      polyset::tabulate(simplex_type, degree, 0, pts), 0, xt::all(), xt::all());

  // Create coefficients for order (degree-1) vector polynomials
  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize});
  for (std::size_t i = 0; i < ndofs; ++i)
  {
    auto p_i = xt::col(psi, i);
    for (std::size_t k = 0; k < psize; ++k)
      wcoeffs(i, k) = xt::sum(wts * p_i * xt::col(psi_quad, k))();
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const std::size_t tdim = topology.size() - 1;

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  M[tdim].push_back(xt::xtensor<double, 3>({ndofs, 1, ndofs}));
  xt::view(M[tdim][0], xt::all(), 0, xt::all()) = xt::eye<double>(ndofs);

  const auto pt
      = lattice::create(simplex_type, degree, lattice::type::equispaced, true);
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;
  x[tdim].push_back(pt);

  std::map<cell::type, xt::xtensor<double, 3>> entity_transformations;
  if (tdim > 1)
  {
    entity_transformations[cell::type::interval]
        = xt::xtensor<double, 3>({1, 0, 0});
  }
  if (tdim == 3)
  {
    entity_transformations[cell::type::quadrilateral]
        = xt::xtensor<double, 3>({2, 0, 0});
  }

  xt::xtensor<double, 3> coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, {M[tdim]}, {x[tdim]}, degree);
  return FiniteElement(element::family::DPC, celltype, degree, {1}, coeffs,
                       entity_transformations, x, M, maps::type::identity,
                       discontinuous);
}
//-----------------------------------------------------------------------------
