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
impl::mdarray2_t create_nedelec_2d_space(int degree)
{
  // Number of order (degree) vector polynomials
  const std::size_t nv = degree * (degree + 1) / 2;

  // Number of order (degree-1) vector polynomials
  const std::size_t ns0 = (degree - 1) * degree / 2;

  // Number of additional polynomials in Nedelec set
  const std::size_t ns = degree;

  // Tabulate polynomial set at quadrature points
  const auto [pts, wts] = quadrature::make_quadrature(
      quadrature::type::Default, cell::type::triangle, 2 * degree);
  const xt::xtensor<double, 3> phi
      = polyset::tabulate(cell::type::triangle, degree, 0, pts);

  const std::size_t psize = phi.shape(1);

  // Create coefficients for order (degree-1) vector polynomials
  impl::mdarray2_t wcoeffs(nv * 2 + ns, psize * 2);
  for (std::size_t i = 0; i < nv; ++i)
  {
    wcoeffs(i, i) = 1.0;
    wcoeffs(nv + i, psize + i) = 1.0;
  }

  // Create coefficients for the additional Nedelec polynomials
  for (std::size_t i = 0; i < ns; ++i)
  {
    for (std::size_t j = 0; j < psize; ++j)
    {
      wcoeffs(2 * nv + i, j) = 0.0;
      wcoeffs(2 * nv + i, j + psize) = 0.0;
      for (std::size_t k = 0; k < wts.size(); ++k)
      {
        double p = phi(0, ns0 + i, k);
        wcoeffs(2 * nv + i, j) += wts[k] * p * pts(k, 1) * phi(0, j, k);
        wcoeffs(2 * nv + i, j + psize) -= wts[k] * p * pts(k, 0) * phi(0, j, k);
      }
    }
  }

  return wcoeffs;
}
//-----------------------------------------------------------------------------
impl::mdarray2_t create_nedelec_3d_space(int degree)
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
  const auto [pts, wts] = quadrature::make_quadrature(
      quadrature::type::Default, cell::type::tetrahedron, 2 * degree);
  xt::xtensor<double, 3> phi
      = polyset::tabulate(cell::type::tetrahedron, degree, 0, pts);
  const std::size_t psize = phi.shape(1);

  // Create coefficients for order (degree-1) polynomials
  impl::mdarray2_t wcoeffs(ndofs, psize * tdim);
  for (std::size_t i = 0; i < tdim; ++i)
    for (std::size_t j = 0; j < nv; ++j)
      wcoeffs(i * nv + j, i * psize + j) = 1.0;

  // Create coefficients for additional Nedelec polynomials
  for (std::size_t i = 0; i < ns; ++i)
  {
    for (std::size_t j = 0; j < psize; ++j)
    {
      double w = 0.0;
      for (std::size_t k = 0; k < wts.size(); ++k)
        w += wts[k] * phi(0, ns0 + i, k) * pts(k, 2) * phi(0, j, k);

      // Don't include polynomials (*, *, 0) that are dependant
      if (i >= ns_remove)
        wcoeffs(tdim * nv + i - ns_remove, psize + j) = -w;
      wcoeffs(tdim * nv + i + ns - ns_remove, j) = w;
    }
  }

  for (std::size_t i = 0; i < ns; ++i)
  {
    for (std::size_t j = 0; j < psize; ++j)
    {
      double w = 0.0;
      for (std::size_t k = 0; k < wts.size(); ++k)
        w += wts[k] * phi(0, ns0 + i, k) * pts(k, 1) * phi(0, j, k);
      wcoeffs(tdim * nv + i + ns * 2 - ns_remove, j) = -w;

      // Don't include polynomials (*, *, 0) that are dependant
      if (i >= ns_remove)
        wcoeffs(tdim * nv + i - ns_remove, psize * 2 + j) = w;
    }
  }

  for (std::size_t i = 0; i < ns; ++i)
  {
    for (std::size_t j = 0; j < psize; ++j)
    {
      double w = 0.0;
      for (std::size_t k = 0; k < wts.size(); ++k)
        w += wts[k] * phi(0, ns0 + i, k) * pts(k, 0) * phi(0, j, k);

      wcoeffs(tdim * nv + i + ns - ns_remove, psize * 2 + j) = -w;
      wcoeffs(tdim * nv + i + ns * 2 - ns_remove, psize + j) = w;
    }
  }

  return wcoeffs;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
FiniteElement element::create_nedelec(cell::type celltype, int degree,
                                      lagrange_variant lvariant,
                                      bool discontinuous)
{
  if (degree < 1)
    throw std::runtime_error("Degree must be at least 1");

  const std::size_t tdim = cell::topological_dimension(celltype);

  std::array<std::vector<xt::xtensor<double, 4>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  x[0] = std::vector(cell::num_sub_entities(celltype, 0),
                     xt::xtensor<double, 2>({0, tdim}));
  M[0] = std::vector(cell::num_sub_entities(celltype, 0),
                     xt::xtensor<double, 4>({0, tdim, 0, 1}));

  xt::xtensor<double, 2> wcoeffs;
  switch (celltype)
  {
  case cell::type::triangle:
  {
    impl::mdarray2_t _wcoeffs = create_nedelec_2d_space(degree);
    wcoeffs = xt::xtensor<double, 2>({_wcoeffs.extent(0), _wcoeffs.extent(1)});
    std::copy_n(_wcoeffs.data(), _wcoeffs.size(), wcoeffs.data());
    break;
  }
  case cell::type::tetrahedron:
  {
    // wcoeffs = create_nedelec_3d_space(degree);
    impl::mdarray2_t _wcoeffs = create_nedelec_3d_space(degree);
    wcoeffs = xt::xtensor<double, 2>({_wcoeffs.extent(0), _wcoeffs.extent(1)});
    std::copy_n(_wcoeffs.data(), _wcoeffs.size(), wcoeffs.data());
    break;
  }
  default:
    throw std::runtime_error("Invalid celltype in Nedelec");
  }

  // Integral representation for the boundary (edge) dofs
  FiniteElement edge_space = element::create_lagrange(
      cell::type::interval, degree - 1, lvariant, true);
  std::tie(x[1], M[1]) = moments::make_tangent_integral_moments(
      edge_space, celltype, tdim, 2 * degree - 1);

  // Face dofs
  if (degree > 1)
  {
    FiniteElement face_space = element::create_lagrange(
        cell::type::triangle, degree - 2, lvariant, true);
    std::tie(x[2], M[2]) = moments::make_integral_moments(face_space, celltype,
                                                          tdim, 2 * degree - 2);
  }
  else
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, 2);
    x[2] = std::vector(num_ent, xt::xtensor<double, 2>({0, tdim}));
    M[2] = std::vector(num_ent, xt::xtensor<double, 4>({0, tdim, 0, 1}));
  }

  // Volume dofs
  if (tdim == 3)
  {
    if (degree > 2 and tdim == 3)
    {
      std::tie(x[3], M[3]) = moments::make_integral_moments(
          element::create_lagrange(cell::type::tetrahedron, degree - 3,
                                   lvariant, true),
          cell::type::tetrahedron, 3, 2 * degree - 3);
    }
    else
    {
      const std::size_t num_ent = cell::num_sub_entities(celltype, 3);
      x[3] = std::vector(num_ent, xt::xtensor<double, 2>({0, tdim}));
      M[3] = std::vector(num_ent, xt::xtensor<double, 4>({0, tdim, 0, 1}));
    }
  }

  if (discontinuous)
    std::tie(x, M) = element::make_discontinuous(x, M, tdim, tdim);

  return FiniteElement(element::family::N1E, celltype, degree, {tdim}, wcoeffs,
                       x, M, 0, maps::type::covariantPiola, discontinuous,
                       degree - 1, degree, lvariant);
}
//-----------------------------------------------------------------------------
FiniteElement element::create_nedelec2(cell::type celltype, int degree,
                                       lagrange_variant lvariant,
                                       bool discontinuous)
{
  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron)
    throw std::runtime_error("Invalid celltype in Nedelec");

  if (degree < 1)
    throw std::runtime_error("Degree must be at least 1");

  std::array<std::vector<xt::xtensor<double, 4>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  const std::size_t tdim = cell::topological_dimension(celltype);

  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, 0);
    x[0] = std::vector(num_ent, xt::xtensor<double, 2>({0, tdim}));
    M[0] = std::vector(num_ent, xt::xtensor<double, 4>({0, tdim, 0, 1}));
  }

  // Integral representation for the edge dofs
  FiniteElement edge_space
      = element::create_lagrange(cell::type::interval, degree, lvariant, true);
  std::tie(x[1], M[1]) = moments::make_tangent_integral_moments(
      edge_space, celltype, tdim, 2 * degree);

  if (degree > 1)
  {
    // Integral moments on faces
    FiniteElement face_space
        = element::create_rt(cell::type::triangle, degree - 1, lvariant, true);
    std::tie(x[2], M[2]) = moments::make_dot_integral_moments(
        face_space, celltype, tdim, 2 * degree - 1);
  }
  else
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, 2);
    x[2] = std::vector(num_ent, xt::xtensor<double, 2>({0, tdim}));
    M[2] = std::vector(num_ent, xt::xtensor<double, 4>({0, tdim, 0, 1}));
  }
  if (tdim == 3)
  {
    if (degree > 2)
    {
      // Interior integral moment
      std::tie(x[3], M[3]) = moments::make_dot_integral_moments(
          element::create_rt(cell::type::tetrahedron, degree - 2, lvariant,
                             true),
          celltype, tdim, 2 * degree - 2);
    }
    else
    {
      const std::size_t num_ent = cell::num_sub_entities(celltype, 2);
      x[3] = std::vector(num_ent, xt::xtensor<double, 2>({0, tdim}));
      M[3] = std::vector(num_ent, xt::xtensor<double, 4>({0, tdim, 0, 1}));
    }
  }

  const std::size_t psize = polyset::dim(celltype, degree);
  xt::xtensor<double, 2> wcoeffs = xt::eye<double>(tdim * psize);

  if (discontinuous)
    std::tie(x, M) = element::make_discontinuous(x, M, tdim, tdim);

  return FiniteElement(element::family::N2E, celltype, degree, {tdim}, wcoeffs,
                       x, M, 0, maps::type::covariantPiola, discontinuous,
                       degree, degree, lvariant);
}
//-----------------------------------------------------------------------------
