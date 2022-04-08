// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-nedelec.h"
#include "e-lagrange.h"
#include "e-raviart-thomas.h"
#include "element-families.h"
#include "maps.h"
#include "moments.h"
#include "polyset.h"
#include "quadrature.h"
#include <array>
#include <vector>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

namespace
{
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_nedelec_2d_space(int degree)
{
  // Number of order (degree) vector polynomials
  const std::size_t nv = degree * (degree + 1) / 2;

  // Number of order (degree-1) vector polynomials
  const std::size_t ns0 = (degree - 1) * degree / 2;

  // Number of additional polynomials in Nedelec set
  const std::size_t ns = degree;

  // Tabulate polynomial set at quadrature points
  const auto [pts, _wts] = quadrature::make_quadrature(
      quadrature::type::Default, cell::type::triangle, 2 * degree);
  const auto wts = xt::adapt(_wts);
  const xt::xtensor<double, 2> phi
      = xt::view(polyset::tabulate(cell::type::triangle, degree, 0, pts), 0,
                 xt::all(), xt::all());

  const std::size_t psize = phi.shape(1);

  // Create coefficients for order (degree-1) vector polynomials
  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({nv * 2 + ns, psize * 2});
  xt::view(wcoeffs, xt::range(0, nv), xt::range(0, nv)) = xt::eye<double>(nv);
  xt::view(wcoeffs, xt::range(nv, 2 * nv), xt::range(psize, psize + nv))
      = xt::eye<double>(nv);

  // Create coefficients for the additional Nedelec polynomials
  for (std::size_t i = 0; i < ns; ++i)
  {
    auto p = xt::col(phi, ns0 + i);
    for (std::size_t k = 0; k < psize; ++k)
    {
      auto pk = xt::col(phi, k);
      wcoeffs(2 * nv + i, k) = xt::sum(wts * p * xt::col(pts, 1) * pk)();
      wcoeffs(2 * nv + i, k + psize)
          = xt::sum(-wts * p * xt::col(pts, 0) * pk)();
    }
  }

  return wcoeffs;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_nedelec_3d_space(int degree)
{
  // Reference tetrahedron
  const std::size_t tdim = 3;

  // Number of order (degree) vector polynomials
  const std::size_t nv = degree * (degree + 1) * (degree + 2) / 6;

  // Number of order (degree-1) vector polynomials
  const std::size_t ns0 = (degree - 1) * degree * (degree + 1) / 6;

  // Number of additional Nedelec polynomials that could be added
  const std::size_t ns = degree * (degree + 1) / 2;

  // Number of polynomials that would be included that are not
  // independent so are removed
  const std::size_t ns_remove = degree * (degree - 1) / 2;

  // Number of dofs in the space, ie size of polynomial set
  const std::size_t ndofs = 6 * degree + 4 * degree * (degree - 1)
                            + (degree - 2) * (degree - 1) * degree / 2;

  // Tabulate polynomial basis at quadrature points
  const auto [pts, _wts] = quadrature::make_quadrature(
      quadrature::type::Default, cell::type::tetrahedron, 2 * degree);
  const auto wts = xt::adapt(_wts);
  xt::xtensor<double, 2> phi
      = xt::view(polyset::tabulate(cell::type::tetrahedron, degree, 0, pts), 0,
                 xt::all(), xt::all());
  const std::size_t psize = phi.shape(1);

  // Create coefficients for order (degree-1) polynomials
  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize * tdim});
  for (std::size_t i = 0; i < tdim; ++i)
  {
    auto range0 = xt::range(nv * i, nv * i + nv);
    auto range1 = xt::range(psize * i, psize * i + nv);
    xt::view(wcoeffs, range0, range1) = xt::eye<double>(nv);
  }

  // Create coefficients for additional Nedelec polynomials
  auto p0 = xt::col(pts, 0);
  auto p1 = xt::col(pts, 1);
  auto p2 = xt::col(pts, 2);
  for (std::size_t i = 0; i < ns; ++i)
  {
    auto p = xt::col(phi, ns0 + i);
    for (std::size_t k = 0; k < psize; ++k)
    {
      const double w = xt::sum(wts * p * p2 * xt::col(phi, k))();

      // Don't include polynomials (*, *, 0) that are dependant
      if (i >= ns_remove)
        wcoeffs(tdim * nv + i - ns_remove, psize + k) = -w;
      wcoeffs(tdim * nv + i + ns - ns_remove, k) = w;
    }
  }

  for (std::size_t i = 0; i < ns; ++i)
  {
    auto p = xt::col(phi, ns0 + i);
    for (std::size_t k = 0; k < psize; ++k)
    {
      const double w = xt::sum(wts * p * p1 * xt::col(phi, k))();
      wcoeffs(tdim * nv + i + ns * 2 - ns_remove, k) = -w;

      // Don't include polynomials (*, *, 0) that are dependant
      if (i >= ns_remove)
        wcoeffs(tdim * nv + i - ns_remove, psize * 2 + k) = w;
    }
  }

  for (std::size_t i = 0; i < ns; ++i)
  {
    auto p = xt::col(phi, ns0 + i);
    for (std::size_t k = 0; k < psize; ++k)
    {
      const double w = xt::sum(wts * p * p0 * xt::col(phi, k))();
      wcoeffs(tdim * nv + i + ns - ns_remove, psize * 2 + k) = -w;
      wcoeffs(tdim * nv + i + ns * 2 - ns_remove, psize + k) = w;
    }
  }

  return wcoeffs;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
FiniteElement basix::element::create_nedelec(cell::type celltype, int degree,
                                             bool discontinuous)
{
  if (degree < 1)
    throw std::runtime_error("Degree must be at least 1");

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;
  xt::xtensor<double, 2> wcoeffs;

  const std::size_t tdim = cell::topological_dimension(celltype);

  x[0] = std::vector<xt::xtensor<double, 2>>(
      cell::num_sub_entities(celltype, 0), xt::xtensor<double, 2>({0, tdim}));
  M[0] = std::vector<xt::xtensor<double, 3>>(
      cell::num_sub_entities(celltype, 0),
      xt::xtensor<double, 3>({0, tdim, 0}));

  switch (celltype)
  {
  case cell::type::triangle:
  {
    wcoeffs = create_nedelec_2d_space(degree);
    break;
  }
  case cell::type::tetrahedron:
  {
    wcoeffs = create_nedelec_3d_space(degree);
    break;
  }
  default:
    throw std::runtime_error("Invalid celltype in Nedelec");
  }

  // Integral representation for the boundary (edge) dofs
  FiniteElement edge_space
      = element::create_lagrange(cell::type::interval, degree - 1,
                                 element::lagrange_variant::equispaced, true);
  std::tie(x[1], M[1]) = moments::make_tangent_integral_moments(
      edge_space, celltype, tdim, 2 * degree - 1);

  // Face dofs
  if (degree > 1)
  {
    FiniteElement face_space
        = element::create_lagrange(cell::type::triangle, degree - 2,
                                   element::lagrange_variant::equispaced, true);
    std::tie(x[2], M[2]) = moments::make_integral_moments(face_space, celltype,
                                                          tdim, 2 * degree - 2);
  }
  else
  {
    x[2] = std::vector<xt::xtensor<double, 2>>(
        cell::num_sub_entities(celltype, 2), xt::xtensor<double, 2>({0, tdim}));
    M[2] = std::vector<xt::xtensor<double, 3>>(
        cell::num_sub_entities(celltype, 2),
        xt::xtensor<double, 3>({0, tdim, 0}));
  }

  // Volume dofs
  if (tdim == 3)
  {
    if (degree > 2 and tdim == 3)
    {
      std::tie(x[3], M[3]) = moments::make_integral_moments(
          element::create_lagrange(cell::type::tetrahedron, degree - 3,
                                   element::lagrange_variant::equispaced, true),
          cell::type::tetrahedron, 3, 2 * degree - 3);
    }
    else
    {
      x[3] = std::vector<xt::xtensor<double, 2>>(
          cell::num_sub_entities(celltype, 3),
          xt::xtensor<double, 2>({0, tdim}));
      M[3] = std::vector<xt::xtensor<double, 3>>(
          cell::num_sub_entities(celltype, 3),
          xt::xtensor<double, 3>({0, tdim, 0}));
    }
  }

  if (discontinuous)
  {
    std::tie(x, M) = element::make_discontinuous(x, M, tdim, tdim);
  }

  return FiniteElement(element::family::N1E, celltype, degree, {tdim}, wcoeffs,
                       x, M, maps::type::covariantPiola, discontinuous,
                       degree - 1);
}
//-----------------------------------------------------------------------------
FiniteElement basix::element::create_nedelec2(cell::type celltype, int degree,
                                              bool discontinuous)
{
  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron)
    throw std::runtime_error("Invalid celltype in Nedelec");

  if (degree < 1)
    throw std::runtime_error("Degree must be at least 1");

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  const std::size_t tdim = cell::topological_dimension(celltype);

  x[0] = std::vector<xt::xtensor<double, 2>>(
      cell::num_sub_entities(celltype, 0), xt::xtensor<double, 2>({0, tdim}));
  M[0] = std::vector<xt::xtensor<double, 3>>(
      cell::num_sub_entities(celltype, 0),
      xt::xtensor<double, 3>({0, tdim, 0}));

  // Integral representation for the edge dofs
  FiniteElement edge_space
      = element::create_lagrange(cell::type::interval, degree,
                                 element::lagrange_variant::equispaced, true);
  std::tie(x[1], M[1]) = moments::make_tangent_integral_moments(
      edge_space, celltype, tdim, 2 * degree);

  if (degree > 1)
  {
    // Integral moments on faces
    FiniteElement face_space
        = element::create_rt(cell::type::triangle, degree - 1, true);
    std::tie(x[2], M[2]) = moments::make_dot_integral_moments(
        face_space, celltype, tdim, 2 * degree - 1);
  }
  else
  {
    x[2] = std::vector<xt::xtensor<double, 2>>(
        cell::num_sub_entities(celltype, 2), xt::xtensor<double, 2>({0, tdim}));
    M[2] = std::vector<xt::xtensor<double, 3>>(
        cell::num_sub_entities(celltype, 2),
        xt::xtensor<double, 3>({0, tdim, 0}));
  }
  if (tdim == 3)
  {
    if (degree > 2)
    {
      // Interior integral moment
      std::tie(x[3], M[3]) = moments::make_dot_integral_moments(
          element::create_rt(cell::type::tetrahedron, degree - 2, true),
          celltype, tdim, 2 * degree - 2);
    }
    else
    {
      x[3] = std::vector<xt::xtensor<double, 2>>(
          cell::num_sub_entities(celltype, 3),
          xt::xtensor<double, 2>({0, tdim}));
      M[3] = std::vector<xt::xtensor<double, 3>>(
          cell::num_sub_entities(celltype, 3),
          xt::xtensor<double, 3>({0, tdim, 0}));
    }
  }

  const std::size_t psize = polyset::dim(celltype, degree);
  xt::xtensor<double, 2> wcoeffs = xt::eye<double>(tdim * psize);

  if (discontinuous)
  {
    std::tie(x, M) = element::make_discontinuous(x, M, tdim, tdim);
  }

  return FiniteElement(element::family::N2E, celltype, degree, {tdim}, wcoeffs,
                       x, M, maps::type::covariantPiola, discontinuous, degree);
}
//-----------------------------------------------------------------------------
