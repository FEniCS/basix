// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-nce-rtc.h"
#include "e-lagrange.h"
#include "element-families.h"
#include "log.h"
#include "maps.h"
#include "moments.h"
#include "polyset.h"
#include "quadrature.h"
#include <numeric>
#include <vector>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

//----------------------------------------------------------------------------
FiniteElement basix::create_rtc(cell::type celltype, int degree,
                                element::variant variant)
{
  if (celltype != cell::type::quadrilateral
      and celltype != cell::type::hexahedron)
  {
    throw std::runtime_error("Unsupported cell type");
  }

  if (degree > 4 and variant == element::variant::EQ)
  {
    // TODO: suggest alternative with non-uniform points once implemented
    LOG(WARNING) << "RTC spaces with high degree using equally spaced"
                 << " points are unstable.";
  }

  const std::size_t tdim = cell::topological_dimension(celltype);

  const cell::type facettype
      = (tdim == 2) ? cell::type::interval : cell::type::quadrilateral;

  // Evaluate the expansion polynomials at the quadrature points
  auto [pts, _wts]
      = quadrature::make_quadrature("default", celltype, 2 * degree);
  auto Qwts = xt::adapt(_wts);
  xt::xtensor<double, 2> phi = xt::view(
      polyset::tabulate(celltype, degree, 0, pts), 0, xt::all(), xt::all());

  // The number of order (degree) polynomials
  const std::size_t psize = phi.shape(1);

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
      xt::xtensor<double, 1> integrand = xt::pow(xt::col(pts, d), degree);
      for (std::size_t c = 0; c < tdim; ++c)
      {
        if (c != d)
        {
          integrand *= xt::pow(xt::col(pts, c), indices[n]);
          ++n;
        }
      }

      for (std::size_t k = 0; k < psize; ++k)
      {
        const double w_sum = xt::sum(Qwts * integrand * xt::col(phi, k))();
        wcoeffs(dof, k + psize * d) = w_sum;
      }
      ++dof;
    }
  }

  // Quadrature degree
  int quad_deg = 2 * degree;

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  FiniteElement moment_space = create_dlagrange(facettype, degree - 1, variant);
  std::tie(x[tdim - 1], M[tdim - 1]) = moments::make_normal_integral_moments(
      moment_space, celltype, tdim, quad_deg);
  xt::xtensor<double, 3> facet_transforms
      = moments::create_normal_moment_dof_transformations(moment_space);

  // Add integral moments on interior
  if (degree > 1)
  {
    std::tie(x[tdim], M[tdim]) = moments::make_dot_integral_moments(
        create_nce(celltype, degree - 1, variant), celltype, tdim, quad_deg);
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::vector<xt::xtensor<double, 2>> entity_transformations;
  if (tdim == 2)
  {
    entity_transformations.push_back(
        xt::view(facet_transforms, 0, xt::all(), xt::all()));
  }
  else if (tdim == 3)
  {
    entity_transformations.push_back(xt::xtensor<double, 2>({0, 0}));
    entity_transformations.push_back(
        xt::view(facet_transforms, 0, xt::all(), xt::all()));
    entity_transformations.push_back(
        xt::view(facet_transforms, 1, xt::all(), xt::all()));
  }

  xt::xtensor<double, 3> coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, {M[tdim - 1], M[tdim]}, {x[tdim - 1], x[tdim]},
      degree);
  return FiniteElement(element::family::RT, celltype, degree, {tdim}, coeffs,
                       entity_transformations, x, M,
                       maps::type::contravariantPiola);
}
//-----------------------------------------------------------------------------
FiniteElement basix::create_nce(cell::type celltype, int degree,
                                element::variant variant)
{
  if (celltype != cell::type::quadrilateral
      and celltype != cell::type::hexahedron)
    throw std::runtime_error("Unsupported cell type");

  if (degree > 4 and variant == element::variant::EQ)
  {
    // TODO: suggest alternative with non-uniform points once implemented
    LOG(WARNING) << "NC spaces with high degree using equally spaced"
                 << " points are unstable.";
  }

  const std::size_t tdim = cell::topological_dimension(celltype);

  // Evaluate the expansion polynomials at the quadrature points
  auto [pts, _wts]
      = quadrature::make_quadrature("default", celltype, 2 * degree);
  auto wts = xt::adapt(_wts);
  xt::xtensor<double, 2> phi = xt::view(
      polyset::tabulate(celltype, degree, 0, pts), 0, xt::all(), xt::all());

  // The number of order (degree) polynomials
  const int psize = phi.shape(1);

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
  switch (tdim)
  {
  case 2:
  {
    for (int i = 0; i < degree; ++i)
    {
      for (std::size_t d = 0; d < tdim; ++d)
      {
        integrand = xt::col(pts, 1 - d);
        for (int j = 1; j < degree; ++j)
          integrand *= xt::col(pts, 1 - d);
        for (int j = 0; j < i; ++j)
          integrand *= xt::col(pts, d);
        for (int k = 0; k < psize; ++k)
        {
          const double w_sum = xt::sum(wts * integrand * xt::col(phi, k))();
          wcoeffs(dof, k + psize * d) = w_sum;
        }
        ++dof;
      }
    }
    break;
  }
  default:
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

              integrand = xt::col(pts, e);
              for (int k = 1; k < degree; ++k)
                integrand *= xt::col(pts, e);
              for (int k = 0; k < i; ++k)
                integrand *= xt::col(pts, d);
              for (int k = 0; k < j; ++k)
                integrand *= xt::col(pts, c);

              for (int k = 0; k < psize; ++k)
              {
                const double w_sum
                    = xt::sum(wts * integrand * xt::col(phi, k))();
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

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  FiniteElement edge_moment_space
      = create_dlagrange(cell::type::interval, degree - 1, variant);
  std::tie(x[1], M[1]) = moments::make_tangent_integral_moments(
      edge_moment_space, celltype, tdim, quad_deg);
  xt::xtensor<double, 3> edge_transforms
      = moments::create_tangent_moment_dof_transformations(edge_moment_space);

  // Add integral moments on interior
  xt::xtensor<double, 3> face_transforms;
  if (degree > 1)
  {
    // Face integral moment
    FiniteElement moment_space
        = create_rtc(cell::type::quadrilateral, degree - 1, variant);
    std::tie(x[2], M[2]) = moments::make_dot_integral_moments(
        moment_space, celltype, tdim, quad_deg);

    if (tdim == 3)
    {
      face_transforms
          = moments::create_dot_moment_dof_transformations(moment_space);

      // Interior integral moment
      std::tie(x[3], M[3]) = moments::make_dot_integral_moments(
          create_rtc(cell::type::hexahedron, degree - 1, variant), celltype,
          tdim, quad_deg);
    }
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::vector<xt::xtensor<double, 2>> entity_transformations;

  entity_transformations.push_back(
      xt::view(edge_transforms, 0, xt::all(), xt::all()));

  if (tdim == 3)
  {
    if (degree == 1)
    {
      entity_transformations.push_back(xt::xtensor<double, 2>({0, 0}));
      entity_transformations.push_back(xt::xtensor<double, 2>({0, 0}));
    }
    else
    {
      entity_transformations.push_back(
          xt::view(face_transforms, 0, xt::all(), xt::all()));
      entity_transformations.push_back(
          xt::view(face_transforms, 1, xt::all(), xt::all()));
    }
  }

  xt::xtensor<double, 3> coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, {M[1], M[2], M[3]}, {x[1], x[2], x[3]}, degree);
  return FiniteElement(element::family::N1E, celltype, degree, {tdim}, coeffs,
                       entity_transformations, x, M,
                       maps::type::covariantPiola);
}
//-----------------------------------------------------------------------------
