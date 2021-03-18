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

  const std::size_t ndofs = polyset::dim(celltype, degree);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  std::vector<std::vector<int>> entity_dofs(topology.size());

  // Create points at nodes, ordered by topology (vertices first)
  xt::xtensor<double, 2> pt({ndofs, topology.size() - 1});
  if (degree == 0)
  {
    pt = lattice::create(celltype, 0, lattice::type::equispaced, true);
    for (std::size_t i = 0; i < entity_dofs.size(); ++i)
      entity_dofs[i].resize(topology[i].size(), 0);
    entity_dofs[topology.size() - 1][0] = 1;
  }
  else
  {
    int c = 0;
    for (std::size_t dim = 0; dim < topology.size(); ++dim)
    {
      for (std::size_t i = 0; i < topology[dim].size(); ++i)
      {
        const xt::xtensor<double, 2> entity_geom
            = cell::sub_entity_geometry(celltype, dim, i);
        if (dim == 0)
        {
          xt::row(pt, c++) = xt::row(entity_geom, 0);
          entity_dofs[0].push_back(1);
        }
        else if (dim == topology.size() - 1)
        {
          const auto lattice = lattice::create(
              celltype, degree, lattice::type::equispaced, false);
          for (std::size_t j = 0; j < lattice.shape(0); ++j)
            xt::row(pt, c++) = xt::row(lattice, j);
          entity_dofs[dim].push_back(lattice.shape(0));
        }
        else
        {
          cell::type ct = cell::sub_entity_type(celltype, dim, i);
          const auto lattice
              = lattice::create(ct, degree, lattice::type::equispaced, false);
          entity_dofs[dim].push_back(lattice.shape(0));
          for (std::size_t j = 0; j < lattice.shape(0); ++j)
          {
            xt::row(pt, c) = xt::row(entity_geom, 0);
            for (std::size_t k = 0; k < lattice.shape(1); ++k)
            {
              xt::row(pt, c)
                  += (xt::row(entity_geom, k + 1) - xt::row(entity_geom, 0))
                     * lattice(j, k);
            }
            ++c;
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
  {
    assert(transform_count == 0);
    break;
  }
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

  xt::xtensor<double, 2> coeffs = compute_expansion_coefficients(
      celltype, xt::eye<double>(ndofs), xt::eye<double>(ndofs), pt, degree);
  return FiniteElement(element::family::P, celltype, degree, {1}, coeffs,
                       entity_dofs, base_transformations, pt,
                       xt::eye<double>(ndofs), map::type::identity);
}
//-----------------------------------------------------------------------------
FiniteElement basix::create_dlagrange(cell::type celltype, int degree)
{
  // Only tabulate for scalar. Vector spaces can easily be built from
  // the scalar space.

  const std::size_t ndofs = polyset::dim(celltype, degree);

  std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t i = 0; i < topology.size(); ++i)
    entity_dofs[i].resize(topology[i].size(), 0);
  entity_dofs[topology.size() - 1][0] = ndofs;

  const auto pt
      = lattice::create(celltype, degree, lattice::type::equispaced, true);

  std::size_t transform_count = 0;
  for (std::size_t i = 1; i < topology.size() - 1; ++i)
    transform_count += topology[i].size() * i;

  auto base_transformations
      = xt::tile(xt::expand_dims(xt::eye<double>(ndofs), 0), transform_count);
  xt::xtensor<double, 2> coeffs = compute_expansion_coefficients(
      celltype, xt::eye<double>(ndofs), xt::eye<double>(ndofs), pt, degree);

  return FiniteElement(element::family::DP, celltype, degree, {1}, coeffs,
                       entity_dofs, base_transformations, pt,
                       xt::eye<double>(ndofs), map::type::identity);
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

  auto [Qpts, _Qwts]
      = quadrature::make_quadrature("default", celltype, 2 * degree);
  auto Qwts = xt::adapt(_Qwts);

  xt::xtensor<double, 2> quad_polyset_at_Qpts = xt::view(
      polyset::tabulate(celltype, degree, 0, Qpts), 0, xt::all(), xt::all());
  xt::xtensor<double, 2> polyset_at_Qpts
      = xt::view(polyset::tabulate(simplex_type, degree, 0, Qpts), 0, xt::all(),
                 xt::all());

  // Create coefficients for order (degree-1) vector polynomials
  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize});
  for (std::size_t i = 0; i < ndofs; ++i)
  {
    auto p_i = xt::col(polyset_at_Qpts, i);
    for (std::size_t k = 0; k < psize; ++k)
      wcoeffs(i, k) = xt::sum(Qwts * p_i * xt::col(quad_polyset_at_Qpts, k))();
  }

  std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t i = 0; i < topology.size(); ++i)
    entity_dofs[i].resize(topology[i].size(), 0);
  entity_dofs[topology.size() - 1][0] = ndofs;

  const auto pt
      = lattice::create(simplex_type, degree, lattice::type::equispaced, true);

  std::size_t transform_count = 0;
  for (std::size_t i = 1; i < topology.size() - 1; ++i)
    transform_count += topology[i].size() * i;

  auto base_transformations
      = xt::tile(xt::expand_dims(xt::eye<double>(ndofs), 0), transform_count);
  xt::xtensor<double, 2> coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, xt::eye<double>(ndofs), pt, degree);
  return FiniteElement(element::family::DPC, celltype, degree, {1}, coeffs,
                       entity_dofs, base_transformations, pt,
                       xt::eye<double>(ndofs), map::type::identity);
}
//-----------------------------------------------------------------------------
