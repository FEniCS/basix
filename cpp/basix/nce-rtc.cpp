// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "nce-rtc.h"
#include "element-families.h"
#include "lagrange.h"
#include "log.h"
#include "mappings.h"
#include "moments.h"
#include "polyset.h"
#include "quadrature.h"
#include <numeric>
#include <vector>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

//----------------------------------------------------------------------------
FiniteElement basix::create_rtc(cell::type celltype, int degree)
{
  if (celltype != cell::type::quadrilateral
      and celltype != cell::type::hexahedron)
    throw std::runtime_error("Unsupported cell type");

  if (degree > 4)
  {
    // TODO: suggest alternative with non-uniform points once implemented
    LOG(WARNING) << "RTC spaces with high degree using equally spaced"
                 << " points are unstable.";
  }

  const std::size_t tdim = cell::topological_dimension(celltype);

  const cell::type facettype
      = (tdim == 2) ? cell::type::interval : cell::type::quadrilateral;

  // Evaluate the expansion polynomials at the quadrature points
  auto [Qpts, _Qwts]
      = quadrature::make_quadrature("default", celltype, 2 * degree);
  auto Qwts = xt::adapt(_Qwts);
  xt::xtensor<double, 2> polyset_at_Qpts = xt::view(
      polyset::tabulate(celltype, degree, 0, Qpts), 0, xt::all(), xt::all());

  // The number of order (degree) polynomials
  const std::size_t psize = polyset_at_Qpts.shape()[1];

  const int facet_count = tdim == 2 ? 4 : 6;
  const int facet_dofs = polyset::dim(facettype, degree - 1);
  const int internal_dofs = tdim == 2 ? 2 * degree * (degree - 1)
                                      : 3 * degree * degree * (degree - 1);
  const std::size_t ndofs = facet_count * facet_dofs + internal_dofs;

  // Create coefficients for order (degree-1) vector polynomials
  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize * tdim});
  const int nv_interval = polyset::dim(cell::type::interval, degree);
  const int ns_interval = polyset::dim(cell::type::interval, degree - 1);
  int dof = 0;
  if (tdim == 2)
  {
    for (std::size_t d = 0; d < tdim; ++d)
      for (int i = 0; i < ns_interval; ++i)
        for (int j = 0; j < ns_interval; ++j)
          wcoeffs(dof++, psize * d + i * nv_interval + j) = 1;
  }
  else
  {
    for (std::size_t d = 0; d < tdim; ++d)
      for (int i = 0; i < ns_interval; ++i)
        for (int j = 0; j < ns_interval; ++j)
          for (int k = 0; k < ns_interval; ++k)
            wcoeffs(dof++, psize * d + i * nv_interval * nv_interval
                               + j * nv_interval + k)
                = 1;
  }

  // Create coefficients for additional polynomials in the div space
  for (int i = 0; i < pow(degree, tdim - 1); ++i)
  {
    std::vector<int> indices(tdim - 1);
    if (tdim == 2)
      indices[0] = i;
    else
    {
      indices[0] = i / degree;
      indices[1] = i % degree;
    }
    for (std::size_t d = 0; d < tdim; ++d)
    {
      int n = 0;
      xt::xtensor<double, 1> integrand = xt::pow(xt::col(Qpts, d), degree);
      // for (int j = 1; j < degree; ++j)
      //   integrand *= xt::col(Qpts, d);
      for (std::size_t c = 0; c < tdim; ++c)
      {
        if (c != d)
        {
          for (int j = 0; j < indices[n]; ++j)
            integrand *= xt::col(Qpts, c);
          // integrand *= std::pow(xt::col(Qpts, c), ;
          ++n;
        }
      }
      for (std::size_t k = 0; k < psize; ++k)
      {
        const double w_sum
            = xt::sum(Qwts * integrand * xt::col(polyset_at_Qpts, k))();
        wcoeffs(dof, k + psize * d) = w_sum;
      }
      ++dof;
    }
  }

  // quadrature degree
  int quad_deg = 2 * degree;

  xt::xtensor<double, 2> points_facet, matrix_facet;
  FiniteElement moment_space = create_dlagrange(facettype, degree - 1);
  std::tie(points_facet, matrix_facet) = moments::make_normal_integral_moments(
      moment_space, celltype, tdim, quad_deg);
  xt::xtensor<double, 3> facet_transforms
      = moments::create_normal_moment_dof_transformations(moment_space);

  // Add integral moments on interior
  xt::xtensor<double, 2> points_cell, matrix_cell;
  if (degree > 1)
  {
    // Interior integral moment
    std::tie(points_cell, matrix_cell) = moments::make_dot_integral_moments(
        create_nce(celltype, degree - 1), celltype, tdim, quad_deg);
  }

  // Interpolation points and matrix
  xt::xtensor<double, 2> points, matrix;
  std::tie(points, matrix) = combine_interpolation_data(
      points_facet, points_cell, {}, matrix_facet, matrix_cell, {}, tdim, tdim);

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::size_t transform_count = 0;
  for (std::size_t i = 1; i < tdim; ++i)
    transform_count += topology[i].size() * i;

  xt::xtensor<double, 3> base_transformations
      = xt::zeros<double>({transform_count, ndofs, ndofs});
  for (std::size_t i = 0; i < base_transformations.shape()[0]; ++i)
  {
    xt::view(base_transformations, i, xt::all(), xt::all())
        = xt::eye<double>(ndofs);
  }

  if (tdim == 2)
  {
    for (int edge = 0; edge < facet_count; ++edge)
    {
      const std::size_t start = facet_dofs * edge;
      auto range = xt::range(start, start + facet_dofs);
      xt::view(base_transformations, edge, range, range)
          = xt::view(facet_transforms, 0, xt::all(), xt::all());
    }
  }
  else if (tdim == 3)
  {
    const int edge_count = 12;
    for (int face = 0; face < facet_count; ++face)
    {
      const std::size_t start = facet_dofs * face;
      const std::size_t p = edge_count + 2 * face;
      auto range = xt::range(start, start + facet_dofs);
      xt::view(base_transformations, p, range, range)
          = xt::view(facet_transforms, 0, xt::all(), xt::all());
      xt::view(base_transformations, p + 1, range, range)
          = xt::view(facet_transforms, 1, xt::all(), xt::all());
    }
  }

  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t i = 0; i < tdim - 1; ++i)
    entity_dofs[i].resize(topology[i].size(), 0);
  entity_dofs[tdim - 1].resize(topology[tdim - 1].size(), facet_dofs);
  entity_dofs[tdim] = {internal_dofs};

  xt::xtensor<double, 2> coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, matrix, points, degree);
  return FiniteElement(element::family::RT, celltype, degree, {tdim}, coeffs,
                       entity_dofs, base_transformations, points, matrix,
                       mapping::type::contravariantPiola);
}
//-----------------------------------------------------------------------------
FiniteElement basix::create_nce(cell::type celltype, int degree)
{
  if (celltype != cell::type::quadrilateral
      and celltype != cell::type::hexahedron)
    throw std::runtime_error("Unsupported cell type");

  if (degree > 4)
  {
    // TODO: suggest alternative with non-uniform points once implemented
    LOG(WARNING) << "NC spaces with high degree using equally spaced"
                 << " points are unstable.";
  }

  const std::size_t tdim = cell::topological_dimension(celltype);

  // Evaluate the expansion polynomials at the quadrature points
  auto [Qpts, _Qwts]
      = quadrature::make_quadrature("default", celltype, 2 * degree);
  auto Qwts = xt::adapt(_Qwts);
  xt::xtensor<double, 2> polyset_at_Qpts = xt::view(
      polyset::tabulate(celltype, degree, 0, Qpts), 0, xt::all(), xt::all());

  // The number of order (degree) polynomials
  const int psize = polyset_at_Qpts.shape()[1];

  const int edge_count = tdim == 2 ? 4 : 12;
  const int edge_dofs = polyset::dim(cell::type::interval, degree - 1);
  const int face_count = tdim == 2 ? 1 : 6;
  const int face_dofs = 2 * degree * (degree - 1);
  const int volume_count = tdim == 2 ? 0 : 1;
  const int volume_dofs = 3 * degree * (degree - 1) * (degree - 1);

  const std::size_t ndofs = edge_count * edge_dofs + face_count * face_dofs
                            + volume_count * volume_dofs;

  // Create coefficients for order (degree-1) vector polynomials
  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize * tdim});

  const int nv_interval = polyset::dim(cell::type::interval, degree);
  const int ns_interval = polyset::dim(cell::type::interval, degree - 1);
  int dof = 0;
  if (tdim == 2)
  {
    for (std::size_t d = 0; d < tdim; ++d)
      for (int i = 0; i < ns_interval; ++i)
        for (int j = 0; j < ns_interval; ++j)
          wcoeffs(dof++, psize * d + i * nv_interval + j) = 1;
  }
  else
  {
    for (std::size_t d = 0; d < tdim; ++d)
      for (int i = 0; i < ns_interval; ++i)
        for (int j = 0; j < ns_interval; ++j)
          for (int k = 0; k < ns_interval; ++k)
            wcoeffs(dof++, psize * d + i * nv_interval * nv_interval
                               + j * nv_interval + k)
                = 1;
  }

  // Create coefficients for additional polynomials in the curl space
  xt::xtensor<double, 1> integrand;
  if (tdim == 2)
  {
    for (int i = 0; i < degree; ++i)
    {
      for (std::size_t d = 0; d < tdim; ++d)
      {
        integrand = xt::col(Qpts, 1 - d);
        for (int j = 1; j < degree; ++j)
          integrand *= xt::col(Qpts, 1 - d);
        for (int j = 0; j < i; ++j)
          integrand *= xt::col(Qpts, d);

        for (int k = 0; k < psize; ++k)
        {
          const double w_sum
              = xt::sum(Qwts * integrand * xt::col(polyset_at_Qpts, k))();
          wcoeffs(dof, k + psize * d) = w_sum;
        }
        ++dof;
      }
    }
  }
  else
  {
    for (int i = 0; i < degree; ++i)
    {
      for (int j = 0; j < degree + 1; ++j)
      {
        for (std::size_t c = 0; c < tdim; ++c)
        {
          for (std::size_t d = 0; d < tdim; ++d)
          {
            if (d != c)
            {
              const std::size_t e
                  = (c == 0 || d == 0) ? ((c == 1 || d == 1) ? 2 : 1) : 0;
              if (c < e and j == degree)
                continue;

              integrand = xt::col(Qpts, e);
              for (int k = 1; k < degree; ++k)
                integrand *= xt::col(Qpts, e);
              for (int k = 0; k < i; ++k)
                integrand *= xt::col(Qpts, d);
              for (int k = 0; k < j; ++k)
                integrand *= xt::col(Qpts, c);

              for (int k = 0; k < psize; ++k)
              {
                const double w_sum
                    = xt::sum(Qwts * integrand * xt::col(polyset_at_Qpts, k))();
                wcoeffs(dof, k + psize * d) = w_sum;
              }
              ++dof;
            }
          }
        }
      }
    }
  }

  // quadrature degree
  int quad_deg = 2 * degree;

  xt::xtensor<double, 2> points_1d, matrix_1d;
  FiniteElement edge_moment_space
      = create_dlagrange(cell::type::interval, degree - 1);
  std::tie(points_1d, matrix_1d) = moments::make_tangent_integral_moments(
      edge_moment_space, celltype, tdim, quad_deg);
  xt::xtensor<double, 3> edge_transforms
      = moments::create_tangent_moment_dof_transformations(edge_moment_space);

  // Add integral moments on interior
  xt::xtensor<double, 2> points_2d, matrix_2d, points_3d, matrix_3d;
  xt::xtensor<double, 3> face_transforms;
  if (degree > 1)
  {
    // Face integral moment
    FiniteElement moment_space
        = create_rtc(cell::type::quadrilateral, degree - 1);
    std::tie(points_2d, matrix_2d) = moments::make_dot_integral_moments(
        moment_space, celltype, tdim, quad_deg);

    if (tdim == 3)
    {
      face_transforms
          = moments::create_dot_moment_dof_transformations(moment_space);

      // Interior integral moment
      std::tie(points_3d, matrix_3d) = moments::make_dot_integral_moments(
          create_rtc(cell::type::hexahedron, degree - 1), celltype, tdim,
          quad_deg);
    }
  }

  // Interpolation points and matrix
  xt::xtensor<double, 2> points, matrix;
  std::tie(points, matrix)
      = combine_interpolation_data(points_1d, points_2d, points_3d, matrix_1d,
                                   matrix_2d, matrix_3d, tdim, tdim);

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::size_t transform_count = 0;
  for (std::size_t i = 1; i < tdim; ++i)
    transform_count += topology[i].size() * i;

  xt::xtensor<double, 3> base_transformations
      = xt::zeros<double>({transform_count, ndofs, ndofs});
  for (std::size_t i = 0; i < transform_count; ++i)
  {
    xt::view(base_transformations, i, xt::all(), xt::all())
        = xt::eye<double>(ndofs);
  }

  for (int edge = 0; edge < edge_count; ++edge)
  {
    const std::size_t start = edge_dofs * edge;
    auto range = xt::range(start, start + edge_dofs);
    xt::view(base_transformations, edge, range, range)
        = xt::view(edge_transforms, 0, xt::all(), xt::all());
  }

  if (tdim == 3 and degree > 1)
  {
    for (int face = 0; face < face_count; ++face)
    {
      const std::size_t start = edge_dofs * edge_count + face_dofs * face;
      const std::size_t p = edge_count + 2 * face;
      auto range = xt::range(start, start + face_dofs);
      xt::view(base_transformations, p, range, range)
          = xt::view(face_transforms, 0, xt::all(), xt::all());
      xt::view(base_transformations, p + 1, range, range)
          = xt::view(face_transforms, 1, xt::all(), xt::all());
    }
  }

  std::vector<std::vector<int>> entity_dofs(topology.size());
  entity_dofs[0].resize(topology[0].size(), 0);
  entity_dofs[1].resize(topology[1].size(), edge_dofs);
  entity_dofs[2].resize(topology[2].size(), face_dofs);
  if (tdim == 3)
    entity_dofs[3].resize(topology[3].size(), volume_dofs);

  xt::xtensor<double, 2> coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, matrix, points, degree);

  return FiniteElement(element::family::N1E, celltype, degree, {tdim}, coeffs,
                       entity_dofs, base_transformations, points, matrix,
                       mapping::type::covariantPiola);
}
//-----------------------------------------------------------------------------
