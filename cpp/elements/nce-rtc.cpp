// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "nce-rtc.h"
#include "core/dof-transformations.h"
#include "core/element-families.h"
#include "core/log.h"
#include "core/mappings.h"
#include "core/moments.h"
#include "core/polyset.h"
#include "core/quadrature.h"
#include "lagrange.h"
#include <Eigen/Dense>
#include <numeric>
#include <vector>

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

  const int tdim = cell::topological_dimension(celltype);

  const cell::type facettype
      = (tdim == 2) ? cell::type::interval : cell::type::quadrilateral;

  // Evaluate the expansion polynomials at the quadrature points
  auto [Qpts, Qwts]
      = quadrature::make_quadrature("default", celltype, 2 * degree);
  Eigen::ArrayXXd polyset_at_Qpts
      = polyset::tabulate(celltype, degree, 0, Qpts)[0];

  // The number of order (degree) polynomials
  const int psize = polyset_at_Qpts.cols();

  const int facet_count = tdim == 2 ? 4 : 6;
  const int facet_dofs = polyset::dim(facettype, degree - 1);
  const int internal_dofs = tdim == 2 ? 2 * degree * (degree - 1)
                                      : 3 * degree * degree * (degree - 1);
  const int ndofs = facet_count * facet_dofs + internal_dofs;

  // Create coefficients for order (degree-1) vector polynomials
  Eigen::MatrixXd wcoeffs = Eigen::MatrixXd::Zero(ndofs, psize * tdim);

  const int nv_interval = polyset::dim(cell::type::interval, degree);
  const int ns_interval = polyset::dim(cell::type::interval, degree - 1);
  int dof = 0;
  if (tdim == 2)
  {
    for (int d = 0; d < tdim; ++d)
      for (int i = 0; i < ns_interval; ++i)
        for (int j = 0; j < ns_interval; ++j)
          wcoeffs(dof++, psize * d + i * nv_interval + j) = 1;
  }
  else
  {
    for (int d = 0; d < tdim; ++d)
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
    for (int d = 0; d < tdim; ++d)
    {
      int n = 0;
      Eigen::ArrayXd integrand = Qpts.col(d);
      for (int j = 1; j < degree; ++j)
        integrand *= Qpts.col(d);
      for (int c = 0; c < tdim; ++c)
      {
        if (c != d)
        {
          for (int j = 0; j < indices[n]; ++j)
            integrand *= Qpts.col(c);
          ++n;
        }
      }
      for (int k = 0; k < psize; ++k)
      {
        const double w_sum = (Qwts * integrand * polyset_at_Qpts.col(k)).sum();
        wcoeffs(dof, k + psize * d) = w_sum;
      }
      ++dof;
    }
  }

  // quadrature degree
  int quad_deg = 2 * degree;

  Eigen::ArrayXXd points_facet;
  Eigen::MatrixXd matrix_facet;
  std::tie(points_facet, matrix_facet) = moments::make_normal_integral_moments(
      create_dlagrange(facettype, degree - 1), celltype, tdim, degree,
      quad_deg);

  Eigen::ArrayXXd points_cell(0, tdim);
  Eigen::MatrixXd matrix_cell(0, 0);
  // Add integral moments on interior
  if (degree > 1)
  {
    // Interior integral moment
    std::tie(points_cell, matrix_cell) = moments::make_dot_integral_moments(
        create_nce(celltype, degree - 1), celltype, tdim, degree, quad_deg);
  }

  // Interpolation points and matrix
  Eigen::ArrayXXd points;
  Eigen::MatrixXd matrix;

  std::tie(points, matrix) = combine_interpolation_data(
      points_facet, points_cell, {}, matrix_facet, matrix_cell, {}, tdim, tdim);

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  int transform_count = 0;
  for (int i = 1; i < tdim; ++i)
    transform_count += topology[i].size() * i;

  std::vector<Eigen::MatrixXd> base_transformations(
      transform_count, Eigen::MatrixXd::Identity(ndofs, ndofs));
  if (tdim == 2)
  {
    Eigen::ArrayXi edge_ref = doftransforms::interval_reflection(degree);
    Eigen::ArrayXXd edge_dir
        = doftransforms::interval_reflection_tangent_directions(degree);
    for (int edge = 0; edge < facet_count; ++edge)
    {
      const int start = edge_ref.size() * edge;
      for (int i = 0; i < edge_ref.size(); ++i)
      {
        base_transformations[edge](start + i, start + i) = 0;
        base_transformations[edge](start + i, start + edge_ref[i]) = 1;
      }
      Eigen::MatrixXd directions = Eigen::MatrixXd::Identity(ndofs, ndofs);
      directions.block(edge_dir.rows() * edge, edge_dir.cols() * edge,
                       edge_dir.rows(), edge_dir.cols())
          = edge_dir;
      base_transformations[edge] *= directions;
    }
  }
  else if (tdim == 3)
  {
    const int edge_count = 12;
    std::vector<Eigen::MatrixXd> face_transforms
        = moments::create_moment_dof_transformations(
            create_dlagrange(facettype, degree - 1));

    for (int face = 0; face < facet_count; ++face)
    {
      const int start = face_transforms[0].rows() * face;
      const int p = edge_count + 2 * face;

      base_transformations[p].block(start, start, face_transforms[0].rows(),
                                    face_transforms[0].cols())
          = face_transforms[0];
      base_transformations[p + 1].block(start, start, face_transforms[1].rows(),
                                        face_transforms[1].cols())
          = -face_transforms[1];
    }
  }

  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (int i = 0; i < tdim - 1; ++i)
    entity_dofs[i].resize(topology[i].size(), 0);
  entity_dofs[tdim - 1].resize(topology[tdim - 1].size(), facet_dofs);
  entity_dofs[tdim] = {internal_dofs};

  Eigen::MatrixXd coeffs = compute_expansion_coefficients(
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

  const int tdim = cell::topological_dimension(celltype);

  // Evaluate the expansion polynomials at the quadrature points
  auto [Qpts, Qwts]
      = quadrature::make_quadrature("default", celltype, 2 * degree);
  Eigen::ArrayXXd polyset_at_Qpts
      = polyset::tabulate(celltype, degree, 0, Qpts)[0];

  // The number of order (degree) polynomials
  const int psize = polyset_at_Qpts.cols();

  const int edge_count = tdim == 2 ? 4 : 12;
  const int edge_dofs = polyset::dim(cell::type::interval, degree - 1);
  const int face_count = tdim == 2 ? 1 : 6;
  const int face_dofs = 2 * degree * (degree - 1);
  const int volume_count = tdim == 2 ? 0 : 1;
  const int volume_dofs = 3 * degree * (degree - 1) * (degree - 1);

  const int ndofs = edge_count * edge_dofs + face_count * face_dofs
                    + volume_count * volume_dofs;

  // Create coefficients for order (degree-1) vector polynomials
  Eigen::MatrixXd wcoeffs = Eigen::MatrixXd::Zero(ndofs, psize * tdim);

  const int nv_interval = polyset::dim(cell::type::interval, degree);
  const int ns_interval = polyset::dim(cell::type::interval, degree - 1);
  int dof = 0;
  if (tdim == 2)
  {
    for (int d = 0; d < tdim; ++d)
      for (int i = 0; i < ns_interval; ++i)
        for (int j = 0; j < ns_interval; ++j)
          wcoeffs(dof++, psize * d + i * nv_interval + j) = 1;
  }
  else
  {
    for (int d = 0; d < tdim; ++d)
      for (int i = 0; i < ns_interval; ++i)
        for (int j = 0; j < ns_interval; ++j)
          for (int k = 0; k < ns_interval; ++k)
            wcoeffs(dof++, psize * d + i * nv_interval * nv_interval
                               + j * nv_interval + k)
                = 1;
  }

  // Create coefficients for additional polynomials in the curl space
  if (tdim == 2)
  {
    for (int i = 0; i < degree; ++i)
    {
      for (int d = 0; d < tdim; ++d)
      {
        Eigen::ArrayXd integrand = Qpts.col(1 - d);
        for (int j = 1; j < degree; ++j)
          integrand *= Qpts.col(1 - d);
        for (int j = 0; j < i; ++j)
          integrand *= Qpts.col(d);

        for (int k = 0; k < psize; ++k)
        {
          const double w_sum
              = (Qwts * integrand * polyset_at_Qpts.col(k)).sum();
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
        for (int c = 0; c < tdim; ++c)
        {
          for (int d = 0; d < tdim; ++d)
          {
            if (d != c)
            {
              const int e
                  = (c == 0 || d == 0) ? ((c == 1 || d == 1) ? 2 : 1) : 0;
              if (c < e and j == degree)
                continue;
              Eigen::ArrayXd integrand = Qpts.col(e);
              for (int k = 1; k < degree; ++k)
                integrand *= Qpts.col(e);
              for (int k = 0; k < i; ++k)
                integrand *= Qpts.col(d);
              for (int k = 0; k < j; ++k)
                integrand *= Qpts.col(c);

              for (int k = 0; k < psize; ++k)
              {
                const double w_sum
                    = (Qwts * integrand * polyset_at_Qpts.col(k)).sum();
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

  Eigen::ArrayXXd points_1d;
  Eigen::MatrixXd matrix_1d;
  std::tie(points_1d, matrix_1d) = moments::make_tangent_integral_moments(
      create_dlagrange(cell::type::interval, degree - 1), celltype, tdim,
      degree, quad_deg);

  Eigen::ArrayXXd points_2d(0, tdim);
  Eigen::MatrixXd matrix_2d(0, 0);
  Eigen::ArrayXXd points_3d(0, tdim);
  Eigen::MatrixXd matrix_3d(0, 0);
  // Add integral moments on interior
  if (degree > 1)
  {
    // Face integral moment
    std::tie(points_2d, matrix_2d) = moments::make_dot_integral_moments(
        create_rtc(cell::type::quadrilateral, degree - 1), celltype, tdim,
        degree, quad_deg);

    if (tdim == 3)
    {
      // Interior integral moment
      std::tie(points_3d, matrix_3d) = moments::make_dot_integral_moments(
          create_rtc(cell::type::hexahedron, degree - 1), celltype, tdim,
          degree, quad_deg);
    }
  }

  // Interpolation points and matrix
  Eigen::ArrayXXd points;
  Eigen::MatrixXd matrix;

  std::tie(points, matrix)
      = combine_interpolation_data(points_1d, points_2d, points_3d, matrix_1d,
                                   matrix_2d, matrix_3d, tdim, tdim);

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  int transform_count = 0;
  for (int i = 1; i < tdim; ++i)
    transform_count += topology[i].size() * i;

  std::vector<Eigen::MatrixXd> base_transformations(
      transform_count, Eigen::MatrixXd::Identity(ndofs, ndofs));

  Eigen::ArrayXi edge_ref = doftransforms::interval_reflection(degree);
  Eigen::ArrayXXd edge_dir
      = doftransforms::interval_reflection_tangent_directions(degree);

  for (int edge = 0; edge < edge_count; ++edge)
  {
    const int start = edge_ref.size() * edge;
    for (int i = 0; i < edge_ref.size(); ++i)
    {
      base_transformations[edge](start + i, start + i) = 0;
      base_transformations[edge](start + i, start + edge_ref[i]) = 1;
    }
    Eigen::MatrixXd directions = Eigen::MatrixXd::Identity(ndofs, ndofs);
    directions.block(edge_dir.rows() * edge, edge_dir.cols() * edge,
                     edge_dir.rows(), edge_dir.cols())
        = edge_dir;
    base_transformations[edge] *= directions;
  }

  if (tdim == 3 and degree > 1)
  {
    std::vector<Eigen::MatrixXd> face_transforms
        = moments::create_moment_dof_transformations(
            create_rtc(cell::type::quadrilateral, degree - 1));

    for (int face = 0; face < face_count; ++face)
    {
      const int start
          = edge_ref.size() * edge_count + face_transforms[0].rows() * face;
      const int p = edge_count + 2 * face;

      base_transformations[p].block(start, start, face_transforms[0].rows(),
                                    face_transforms[0].cols())
          = face_transforms[0];
      base_transformations[p + 1].block(start, start, face_transforms[1].rows(),
                                        face_transforms[1].cols())
          = face_transforms[1];
    }
  }

  std::vector<std::vector<int>> entity_dofs(topology.size());
  entity_dofs[0].resize(topology[0].size(), 0);
  entity_dofs[1].resize(topology[1].size(), edge_dofs);
  entity_dofs[2].resize(topology[2].size(), face_dofs);
  if (tdim == 3)
    entity_dofs[3].resize(topology[3].size(), volume_dofs);

  Eigen::MatrixXd coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, matrix, points, degree);

  return FiniteElement(element::family::N1E, celltype, degree, {tdim}, coeffs,
                       entity_dofs, base_transformations, points, matrix,
                       mapping::type::covariantPiola);
}
//-----------------------------------------------------------------------------
