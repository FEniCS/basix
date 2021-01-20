// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "qdivcurl.h"
#include "dof-permutations.h"
#include "lagrange.h"
#include "moments.h"
#include "polyset.h"
#include "quadrature.h"
#include "log.h"
#include <Eigen/Dense>
#include <numeric>
#include <vector>

using namespace basix;

//----------------------------------------------------------------------------
FiniteElement basix::create_qdiv(cell::type celltype, int degree,
                                 const std::string& name)
{
  if (celltype != cell::type::quadrilateral
      and celltype != cell::type::hexahedron)
    throw std::runtime_error("Unsupported cell type");

  if (degree > 4)
  {
    // TODO: suggest alternative with non-uniform points once implemented
    LOG(WARNING) << "Qdiv spaces with high degree using equally spaced"
                 << " points are unstable.".
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

  // Dual space
  Eigen::MatrixXd dual = Eigen::MatrixXd::Zero(ndofs, psize * tdim);

  // quadrature degree
  int quad_deg = 2 * degree;

  // Add rows to dualmat for integral moments on facets
  dual.block(0, 0, facet_count * facet_dofs, psize * tdim)
      = moments::make_normal_integral_moments(
          create_dlagrange(facettype, degree - 1), celltype, tdim, degree,
          quad_deg);

  // Add rows to dualmat for integral moments on interior
  if (degree > 1)
  {
    // Interior integral moment
    dual.block(facet_count * facet_dofs, 0, internal_dofs, psize * tdim)
        = moments::make_dot_integral_moments(create_qcurl(celltype, degree - 1),
                                             celltype, tdim, degree, quad_deg);
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  int perm_count = 0;
  for (int i = 1; i < tdim; ++i)
    perm_count += topology[i].size() * i;

  std::vector<Eigen::MatrixXd> base_permutations(
      perm_count, Eigen::MatrixXd::Identity(ndofs, ndofs));
  if (tdim == 2)
  {
    Eigen::ArrayXi edge_ref = dofperms::interval_reflection(degree);
    Eigen::ArrayXXd edge_dir
        = dofperms::interval_reflection_tangent_directions(degree);
    for (int edge = 0; edge < facet_count; ++edge)
    {
      const int start = edge_ref.size() * edge;
      for (int i = 0; i < edge_ref.size(); ++i)
      {
        base_permutations[edge](start + i, start + i) = 0;
        base_permutations[edge](start + i, start + edge_ref[i]) = 1;
      }
      Eigen::MatrixXd directions = Eigen::MatrixXd::Identity(ndofs, ndofs);
      directions.block(edge_dir.rows() * edge, edge_dir.cols() * edge,
                       edge_dir.rows(), edge_dir.cols())
          = edge_dir;
      base_permutations[edge] *= directions;
    }
  }
  else if (tdim == 3)
  {
    Eigen::ArrayXi face_ref = dofperms::quadrilateral_reflection(degree);
    Eigen::ArrayXi face_rot = dofperms::quadrilateral_rotation(degree);

    for (int face = 0; face < facet_count; ++face)
    {
      const int start = face_ref.size() * face;
      for (int i = 0; i < face_rot.size(); ++i)
      {
        base_permutations[12 + 2 * face](start + i, start + i) = 0;
        base_permutations[12 + 2 * face](start + i, start + face_rot[i]) = 1;
        base_permutations[12 + 2 * face + 1](start + i, start + i) = 0;
        base_permutations[12 + 2 * face + 1](start + i, start + face_ref[i])
            = -1;
      }
    }
  }

  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (int i = 0; i < tdim - 1; ++i)
    entity_dofs[i].resize(topology[i].size(), 0);
  entity_dofs[tdim - 1].resize(topology[tdim - 1].size(), facet_dofs);
  entity_dofs[tdim] = {internal_dofs};

  Eigen::MatrixXd coeffs = compute_expansion_coefficients(wcoeffs, dual);
  return FiniteElement(name, celltype, degree, {tdim}, coeffs, entity_dofs,
                       base_permutations, {}, {}, "contravariant piola");
}
//-----------------------------------------------------------------------------
FiniteElement basix::create_qcurl(cell::type celltype, int degree,
                                  const std::string& name)
{
  if (celltype != cell::type::quadrilateral
      and celltype != cell::type::hexahedron)
    throw std::runtime_error("Unsupported cell type");

  if (degree > 4)
  {
    // TODO: suggest alternative with non-uniform points once implemented
    LOG(WARNING) << "Qcurl spaces with high degree using equally spaced"
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

  // Dual space

  Eigen::MatrixXd dual = Eigen::MatrixXd::Zero(ndofs, psize * tdim);

  // quadrature degree
  int quad_deg = 2 * degree;

  // Add rows to dualmat for integral moments on facets
  dual.block(0, 0, edge_count * edge_dofs, psize * tdim)
      = moments::make_tangent_integral_moments(
          create_dlagrange(cell::type::interval, degree - 1), celltype, tdim,
          degree, quad_deg);

  // Add rows to dualmat for integral moments on interior
  if (degree > 1)
  {
    // Face integral moment
    dual.block(edge_count * edge_dofs, 0, face_count * face_dofs, psize * tdim)
        = moments::make_dot_integral_moments(
            create_qdiv(cell::type::quadrilateral, degree - 1), celltype, tdim,
            degree, quad_deg);

    if (tdim == 3)
    {
      // Interior integral moment
      dual.block(edge_count * edge_dofs + face_count * face_dofs, 0,
                 volume_dofs, psize * tdim)
          = moments::make_dot_integral_moments(
              create_qdiv(cell::type::hexahedron, degree - 1), celltype, tdim,
              degree, quad_deg);
    }
  }
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  int perm_count = 0;
  for (int i = 1; i < tdim; ++i)
    perm_count += topology[i].size() * i;

  std::vector<Eigen::MatrixXd> base_permutations(
      perm_count, Eigen::MatrixXd::Identity(ndofs, ndofs));

  Eigen::ArrayXi edge_ref = dofperms::interval_reflection(degree);
  Eigen::ArrayXXd edge_dir
      = dofperms::interval_reflection_tangent_directions(degree);

  for (int edge = 0; edge < edge_count; ++edge)
  {
    const int start = edge_ref.size() * edge;
    for (int i = 0; i < edge_ref.size(); ++i)
    {
      base_permutations[edge](start + i, start + i) = 0;
      base_permutations[edge](start + i, start + edge_ref[i]) = 1;
    }
    Eigen::MatrixXd directions = Eigen::MatrixXd::Identity(ndofs, ndofs);
    directions.block(edge_dir.rows() * edge, edge_dir.cols() * edge,
                     edge_dir.rows(), edge_dir.cols())
        = edge_dir;
    base_permutations[edge] *= directions;
  }

  if (tdim == 3 and degree > 1)
  {
    Eigen::MatrixXd face_ref
        = dofperms::quadrilateral_qdiv_reflection(degree - 1);
    Eigen::MatrixXd face_rot
        = dofperms::quadrilateral_qdiv_rotation(degree - 1);

    for (int face = 0; face < face_count; ++face)
    {
      const int start = edge_ref.size() * edge_count + face_ref.rows() * face;
      const int p = edge_count + 2 * face;

      base_permutations[p].block(start, start, face_rot.rows(), face_rot.cols())
          = face_rot;
      base_permutations[p + 1].block(start, start, face_ref.rows(),
                                     face_ref.cols())
          = face_ref;
    }
  }

  std::vector<std::vector<int>> entity_dofs(topology.size());
  entity_dofs[0].resize(topology[0].size(), 0);
  entity_dofs[1].resize(topology[1].size(), edge_dofs);
  entity_dofs[2].resize(topology[2].size(), face_dofs);
  if (tdim == 3)
    entity_dofs[3].resize(topology[3].size(), volume_dofs);

  Eigen::MatrixXd coeffs = compute_expansion_coefficients(wcoeffs, dual);
  return FiniteElement(name, celltype, degree, {tdim}, coeffs, entity_dofs,
                       base_permutations, {}, {}, "covariant piola");
}
//-----------------------------------------------------------------------------
Eigen::MatrixXd basix::dofperms::quadrilateral_qdiv_rotation(int degree)
{
  const int n = 2 * degree * (degree + 1);
  Eigen::MatrixXd perm = Eigen::MatrixXd::Zero(n, n);

  // Permute functions on edges
  for (int i = 0; i < degree; ++i)
  {
    perm(i, 2 * degree - 1 - i) = -1;
    perm(degree + i, 3 * degree + i) = 1;
    perm(2 * degree + i, i) = 1;
    perm(3 * degree + i, 3 * degree - 1 - i) = -1;
  }
  if (degree > 1)
    perm.block(4 * degree, 4 * degree, n - 4 * degree, n - 4 * degree)
        = quadrilateral_qcurl_rotation(degree - 1);
  return perm;
}
//-----------------------------------------------------------------------------
Eigen::MatrixXd basix::dofperms::quadrilateral_qcurl_rotation(int degree)
{
  const int n = 2 * degree * (degree + 1);
  Eigen::MatrixXd perm = Eigen::MatrixXd::Zero(n, n);

  // Permute functions on edges
  for (int i = 0; i < degree; ++i)
  {
    perm(i, 2 * degree - 1 - i) = -1;
    perm(degree + i, 3 * degree + i) = 1;
    perm(2 * degree + i, i) = 1;
    perm(3 * degree + i, 3 * degree - 1 - i) = -1;
  }
  if (degree > 1)
    perm.block(4 * degree, 4 * degree, n - 4 * degree, n - 4 * degree)
        = quadrilateral_qdiv_rotation(degree - 1);
  return perm;
}
//-----------------------------------------------------------------------------
Eigen::MatrixXd basix::dofperms::quadrilateral_qdiv_reflection(int degree)
{
  const int n = 2 * degree * (degree + 1);
  Eigen::MatrixXd perm = Eigen::MatrixXd::Zero(n, n);

  // Permute functions on edges
  for (int i = 0; i < degree; ++i)
  {
    perm(i, degree + i) = -1;
    perm(degree + i, i) = -1;
    perm(2 * degree + i, 3 * degree + i) = -1;
    perm(3 * degree + i, 2 * degree + i) = -1;
  }
  if (degree > 1)
    perm.block(4 * degree, 4 * degree, n - 4 * degree, n - 4 * degree)
        = quadrilateral_qcurl_reflection(degree - 1);

  return perm;
}
//-----------------------------------------------------------------------------
Eigen::MatrixXd basix::dofperms::quadrilateral_qcurl_reflection(int degree)
{
  const int n = 2 * degree * (degree + 1);
  Eigen::MatrixXd perm = Eigen::MatrixXd::Zero(n, n);

  // Permute functions on edges
  for (int i = 0; i < degree; ++i)
  {
    perm(i, degree + i) = 1;
    perm(degree + i, i) = 1;
    perm(2 * degree + i, 3 * degree + i) = 1;
    perm(3 * degree + i, 2 * degree + i) = 1;
  }
  if (degree > 1)
    perm.block(4 * degree, 4 * degree, n - 4 * degree, n - 4 * degree)
        = quadrilateral_qdiv_reflection(degree - 1);

  return perm;
}
//-----------------------------------------------------------------------------
