// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "lagrange.h"
#include "dof-transformations.h"
#include "element-families.h"
#include "lattice.h"
#include "log.h"
#include "maps.h"
#include "polyset.h"
#include "quadrature.h"
#include <numeric>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

//----------------------------------------------------------------------------
FiniteElement basix::create_lagrange(cell::type celltype, int degree)
{
  if (celltype == cell::type::point)
    throw std::runtime_error("Invalid celltype");

  const std::size_t tdim = cell::topological_dimension(celltype);

  const std::size_t ndofs = polyset::dim(celltype, degree);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;
  std::vector<std::vector<int>> entity_dofs(topology.size());

  // Create points at nodes, ordered by topology (vertices first)
  if (degree == 0)
  {
    auto pt = lattice::create(celltype, 0, lattice::type::equispaced, true);
    for (std::size_t i = 0; i < entity_dofs.size(); ++i)
      entity_dofs[i].resize(topology[i].size(), 0);
    entity_dofs[tdim][0] = 1;

    x[tdim].push_back(pt);
    const std::size_t num_dofs = pt.shape(0);
    std::array<std::size_t, 3> s = {num_dofs, 1, num_dofs};
    M[tdim].push_back(xt::xtensor<double, 3>(s));
    xt::view(M[tdim][0], xt::all(), 0, xt::all()) = xt::eye<double>(num_dofs);
  }
  else
  {
    for (std::size_t dim = 0; dim < topology.size(); ++dim)
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
          entity_dofs[0].push_back(1);
        }
        else if (dim == tdim)
        {
          x[dim][e] = lattice::create(celltype, degree,
                                      lattice::type::equispaced, false);
          const std::size_t num_dofs = x[dim][e].shape(0);
          std::array<std::size_t, 3> s = {num_dofs, 1, num_dofs};
          M[dim][e] = xt::xtensor<double, 3>(s);
          xt::view(M[dim][e], xt::all(), 0, xt::all())
              = xt::eye<double>(num_dofs);
          entity_dofs[dim].push_back(num_dofs);
        }
        else
        {
          cell::type ct = cell::sub_entity_type(celltype, dim, e);
          const auto lattice
              = lattice::create(ct, degree, lattice::type::equispaced, false);

          const std::size_t num_dofs = lattice.shape(0);
          std::array<std::size_t, 3> s = {num_dofs, 1, num_dofs};
          M[dim][e] = xt::xtensor<double, 3>(s);
          xt::view(M[dim][e], xt::all(), 0, xt::all())
              = xt::eye<double>(num_dofs);

          entity_dofs[dim].push_back(lattice.shape(0));

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

  std::size_t transform_count = 0;
  for (std::size_t i = 1; i < topology.size() - 1; ++i)
    transform_count += topology[i].size() * i;
  auto base_transformations
      = xt::tile(xt::expand_dims(xt::eye<double>(ndofs), 0), transform_count);
  switch (celltype)
  {
  case cell::type::interval:
    assert(transform_count == 0);
    break;
  case cell::type::triangle:
  {
    const std::vector<int> edge_ref
        = doftransforms::interval_reflection(degree - 1);
    for (int edge = 0; edge < 3; ++edge)
    {
      auto bt = xt::view(base_transformations, edge, xt::all(), xt::all());
      const int start = 3 + edge_ref.size() * edge;
      for (std::size_t i = 0; i < edge_ref.size(); ++i)
      {
        bt(start + i, start + i) = 0;
        bt(start + i, start + edge_ref[i]) = 1;
      }
    }
    break;
  }
  case cell::type::quadrilateral:
  {
    const std::vector<int> edge_ref
        = doftransforms::interval_reflection(degree - 1);
    for (int edge = 0; edge < 4; ++edge)
    {
      auto bt = xt::view(base_transformations, edge, xt::all(), xt::all());
      const int start = 4 + edge_ref.size() * edge;
      for (std::size_t i = 0; i < edge_ref.size(); ++i)
      {
        bt(start + i, start + i) = 0;
        bt(start + i, start + edge_ref[i]) = 1;
      }
    }
    break;
  }
  case cell::type::tetrahedron:
  {
    const std::vector<int> edge_ref
        = doftransforms::interval_reflection(degree - 1);
    for (int edge = 0; edge < 6; ++edge)
    {
      auto bt = xt::view(base_transformations, edge, xt::all(), xt::all());
      const int start = 4 + edge_ref.size() * edge;
      for (std::size_t i = 0; i < edge_ref.size(); ++i)
      {
        bt(start + i, start + i) = 0;
        bt(start + i, start + edge_ref[i]) = 1;
      }
    }
    const std::vector<int> face_ref
        = doftransforms::triangle_reflection(degree - 2);
    const std::vector<int> face_rot
        = doftransforms::triangle_rotation(degree - 2);
    for (int face = 0; face < 4; ++face)
    {
      auto bt0
          = xt::view(base_transformations, 6 + 2 * face, xt::all(), xt::all());
      auto bt1 = xt::view(base_transformations, 6 + 2 * face + 1, xt::all(),
                          xt::all());
      const int start = 4 + edge_ref.size() * 6 + face_ref.size() * face;
      for (std::size_t i = 0; i < face_rot.size(); ++i)
      {
        bt0(start + i, start + i) = 0;
        bt0(start + i, start + face_rot[i]) = 1;
        bt1(start + i, start + i) = 0;
        bt1(start + i, start + face_ref[i]) = 1;
      }
    }
    break;
  }
  case cell::type::hexahedron:
  {
    const std::vector<int> edge_ref
        = doftransforms::interval_reflection(degree - 1);
    for (int edge = 0; edge < 12; ++edge)
    {
      auto bt = xt::view(base_transformations, edge, xt::all(), xt::all());
      const int start = 8 + edge_ref.size() * edge;
      for (std::size_t i = 0; i < edge_ref.size(); ++i)
      {
        bt(start + i, start + i) = 0;
        bt(start + i, start + edge_ref[i]) = 1;
      }
    }

    const std::vector<int> face_ref
        = doftransforms::quadrilateral_reflection(degree - 1);
    const std::vector<int> face_rot
        = doftransforms::quadrilateral_rotation(degree - 1);
    for (int face = 0; face < 6; ++face)
    {
      auto bt0
          = xt::view(base_transformations, 12 + 2 * face, xt::all(), xt::all());
      auto bt1 = xt::view(base_transformations, 12 + 2 * face + 1, xt::all(),
                          xt::all());
      const int start = 8 + edge_ref.size() * 12 + face_ref.size() * face;
      for (std::size_t i = 0; i < face_rot.size(); ++i)
      {
        bt0(start + i, start + i) = 0;
        bt0(start + i, start + face_rot[i]) = 1;
        bt1(start + i, start + i) = 0;
        bt1(start + i, start + face_ref[i]) = 1;
      }
    }
    break;
  }
  default:
    LOG(WARNING) << "Base transformations not implemented for this cell type.";
  }

  xt::xtensor<double, 3> coeffs = compute_expansion_coefficients(
      celltype, xt::eye<double>(ndofs), {M[0], M[1], M[2], M[3]},
      {x[0], x[1], x[2], x[3]}, degree);
  return FiniteElement(element::family::P, celltype, degree, {1}, coeffs,
                       entity_dofs, base_transformations, x, M,
                       maps::type::identity);
}
//-----------------------------------------------------------------------------
FiniteElement basix::create_dlagrange(cell::type celltype, int degree)
{
  // Only tabulate for scalar. Vector spaces can easily be built from
  // the scalar space.

  const std::size_t ndofs = polyset::dim(celltype, degree);

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t i = 0; i < topology.size(); ++i)
    entity_dofs[i].resize(topology[i].size(), 0);
  entity_dofs[topology.size() - 1][0] = ndofs;

  const std::size_t tdim = topology.size() - 1;

  std::size_t transform_count = 0;
  for (std::size_t i = 1; i < tdim; ++i)
    transform_count += topology[i].size() * i;

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  M[tdim].push_back(xt::xtensor<double, 3>({ndofs, 1, ndofs}));
  xt::view(M[tdim][0], xt::all(), 0, xt::all()) = xt::eye<double>(ndofs);

  const auto pt
      = lattice::create(celltype, degree, lattice::type::equispaced, true);
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;
  x[tdim].push_back(pt);

  auto base_transformations
      = xt::tile(xt::expand_dims(xt::eye<double>(ndofs), 0), transform_count);
  xt::xtensor<double, 3> coeffs = compute_expansion_coefficients(
      celltype, xt::eye<double>(ndofs), {M[tdim]}, {x[tdim]}, degree);

  return FiniteElement(element::family::DP, celltype, degree, {1}, coeffs,
                       entity_dofs, base_transformations, x, M,
                       maps::type::identity);
}
//-----------------------------------------------------------------------------
FiniteElement basix::create_dpc(cell::type celltype, int degree)
{
  // Only tabulate for scalar. Vector spaces can easily be built from
  // the scalar space.

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
  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t i = 0; i < topology.size(); ++i)
    entity_dofs[i].resize(topology[i].size(), 0);
  entity_dofs[tdim][0] = ndofs;

  std::size_t transform_count = 0;
  for (std::size_t i = 1; i < tdim; ++i)
    transform_count += topology[i].size() * i;

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  M[tdim].push_back(xt::xtensor<double, 3>({ndofs, 1, ndofs}));
  xt::view(M[tdim][0], xt::all(), 0, xt::all()) = xt::eye<double>(ndofs);

  const auto pt
      = lattice::create(simplex_type, degree, lattice::type::equispaced, true);
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;
  x[tdim].push_back(pt);

  auto base_transformations
      = xt::tile(xt::expand_dims(xt::eye<double>(ndofs), 0), transform_count);
  xt::xtensor<double, 3> coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, {M[tdim]}, {x[tdim]}, degree);
  return FiniteElement(element::family::DPC, celltype, degree, {1}, coeffs,
                       entity_dofs, base_transformations, x, M,
                       maps::type::identity);
}
//-----------------------------------------------------------------------------
