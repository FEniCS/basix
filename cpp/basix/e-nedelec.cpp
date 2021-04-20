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
#include <xtensor/xpad.hpp>
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
      "default", cell::type::triangle, 2 * degree);
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
std::pair<std::array<std::vector<xt::xtensor<double, 2>>, 4>,
          std::array<std::vector<xt::xtensor<double, 3>>, 4>>
create_nedelec_2d_interpolation(int degree, element::variant variant)
{
  const int quad_deg = 5 * degree;

  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;
  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;

  // Integral representation for the boundary (edge) dofs
  std::tie(x[1], M[1]) = moments::make_tangent_integral_moments(
      create_dlagrange(cell::type::interval, degree - 1, variant),
      cell::type::triangle, 2, quad_deg);
  if (degree > 1)
  {
    std::tie(x[2], M[2]) = moments::make_integral_moments(
        create_dlagrange(cell::type::triangle, degree - 2, variant),
        cell::type::triangle, 2, quad_deg);
  }

  return {x, M};
}
//-----------------------------------------------------------------------------
std::vector<xt::xtensor<double, 2>>
create_nedelec_2d_entity_transforms(int degree, element::variant variant)
{
  std::vector<xt::xtensor<double, 2>> entity_transformations;

  xt::xtensor<double, 3> edge_transforms
      = moments::create_tangent_moment_dof_transformations(
          create_dlagrange(cell::type::interval, degree - 1, variant));
  entity_transformations.push_back(
      xt::view(edge_transforms, 0, xt::all(), xt::all()));

  return entity_transformations;
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
      "default", cell::type::tetrahedron, 2 * degree);
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
std::pair<std::array<std::vector<xt::xtensor<double, 2>>, 4>,
          std::array<std::vector<xt::xtensor<double, 3>>, 4>>
create_nedelec_3d_interpolation(int degree, element::variant variant)
{
  // Number of dofs and interpolation points
  const int quad_deg = 5 * degree;

  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;
  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;

  std::tie(x[1], M[1]) = moments::make_tangent_integral_moments(
      create_dlagrange(cell::type::interval, degree - 1, variant),
      cell::type::tetrahedron, 3, quad_deg);

  if (degree > 1)
  {
    std::tie(x[2], M[2]) = moments::make_integral_moments(
        create_dlagrange(cell::type::triangle, degree - 2, variant),
        cell::type::tetrahedron, 3, quad_deg);
  }

  if (degree > 2)
  {
    std::tie(x[3], M[3]) = moments::make_integral_moments(
        create_dlagrange(cell::type::tetrahedron, degree - 3, variant),
        cell::type::tetrahedron, 3, quad_deg);
  }

  return {x, M};
}
//-----------------------------------------------------------------------------
std::vector<xt::xtensor<double, 2>>
create_nedelec_3d_entity_transforms(int degree, element::variant variant)
{
  std::vector<xt::xtensor<double, 2>> entity_transformations(3);

  const xt::xtensor<double, 3> edge_transforms
      = moments::create_tangent_moment_dof_transformations(
          create_dlagrange(cell::type::interval, degree - 1, variant));
  entity_transformations[0]
      = xt::view(edge_transforms, 0, xt::all(), xt::all());

  // Faces
  if (degree > 1)
  {
    xt::xtensor<double, 3> face_transforms
        = moments::create_moment_dof_transformations(
            create_dlagrange(cell::type::triangle, degree - 2, variant));

    entity_transformations[1]
        = xt::view(face_transforms, 0, xt::all(), xt::all());
    entity_transformations[2]
        = xt::view(face_transforms, 1, xt::all(), xt::all());
  }
  return entity_transformations;
}
//-----------------------------------------------------------------------------
std::pair<std::array<std::vector<xt::xtensor<double, 2>>, 4>,
          std::array<std::vector<xt::xtensor<double, 3>>, 4>>
create_nedelec2_2d_interpolation(int degree, element::variant variant)
{
  const int quad_deg = 5 * degree;

  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;
  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;

  std::tie(x[1], M[1]) = moments::make_tangent_integral_moments(
      create_dlagrange(cell::type::interval, degree, variant),
      cell::type::triangle, 2, quad_deg);
  if (degree > 1)
  {
    std::tie(x[2], M[2]) = moments::make_dot_integral_moments(
        create_rt(cell::type::triangle, degree - 1, variant),
        cell::type::triangle, 2, quad_deg);
  }

  return {x, M};
}
//-----------------------------------------------------------------------------
std::vector<xt::xtensor<double, 2>>
create_nedelec2_2d_entity_transformations(int degree, element::variant variant)
{
  std::vector<xt::xtensor<double, 2>> entity_transformations;

  xt::xtensor<double, 3> edge_transforms
      = moments::create_tangent_moment_dof_transformations(
          create_dlagrange(cell::type::interval, degree, variant));
  entity_transformations.push_back(
      xt::view(edge_transforms, 0, xt::all(), xt::all()));

  return entity_transformations;
}
//-----------------------------------------------------------------------------
std::pair<std::array<std::vector<xt::xtensor<double, 2>>, 4>,
          std::array<std::vector<xt::xtensor<double, 3>>, 4>>
create_nedelec2_3d_interpolation(int degree, element::variant variant)
{
  // Create quadrature scheme on the edge
  const int quad_deg = 5 * degree;

  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;
  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;

  // Integral representation for the boundary (edge) dofs
  std::tie(x[1], M[1]) = moments::make_tangent_integral_moments(
      create_dlagrange(cell::type::interval, degree, variant),
      cell::type::tetrahedron, 3, quad_deg);

  if (degree > 1)
  {
    // Integral moments on faces
    std::tie(x[2], M[2]) = moments::make_dot_integral_moments(
        create_rt(cell::type::triangle, degree - 1, variant),
        cell::type::tetrahedron, 3, quad_deg);
  }

  if (degree > 2)
  {
    // Interior integral moment
    std::tie(x[3], M[3]) = moments::make_dot_integral_moments(
        create_rt(cell::type::tetrahedron, degree - 2, variant),
        cell::type::tetrahedron, 3, quad_deg);
  }

  return {x, M};
}
//-----------------------------------------------------------------------------
std::vector<xt::xtensor<double, 2>>
create_nedelec2_3d_entity_transformations(int degree, element::variant variant)
{
  std::vector<xt::xtensor<double, 2>> entity_transformations;

  const xt::xtensor<double, 3> edge_transforms
      = moments::create_tangent_moment_dof_transformations(
          create_dlagrange(cell::type::interval, degree, variant));
  entity_transformations.push_back(
      xt::view(edge_transforms, 0, xt::all(), xt::all()));

  // Faces
  if (degree == 1)
  {
    entity_transformations.push_back(xt::xtensor<double, 2>({0, 0}));
    entity_transformations.push_back(xt::xtensor<double, 2>({0, 0}));
  }
  else
  {
    const xt::xtensor<double, 3> face_transforms
        = moments::create_dot_moment_dof_transformations(
            create_rt(cell::type::triangle, degree - 1, variant));
    entity_transformations.push_back(
        xt::view(face_transforms, 0, xt::all(), xt::all()));
    entity_transformations.push_back(
        xt::view(face_transforms, 1, xt::all(), xt::all()));
  }

  return entity_transformations;
}

} // namespace

//-----------------------------------------------------------------------------
FiniteElement basix::create_nedelec(cell::type celltype, int degree,
                                    element::variant variant)
{
  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;
  xt::xtensor<double, 2> wcoeffs;
  std::vector<xt::xtensor<double, 2>> transforms;
  switch (celltype)
  {
  case cell::type::triangle:
  {
    wcoeffs = create_nedelec_2d_space(degree);
    transforms = create_nedelec_2d_entity_transforms(degree, variant);
    std::tie(x, M) = create_nedelec_2d_interpolation(degree, variant);
    break;
  }
  case cell::type::tetrahedron:
  {
    wcoeffs = create_nedelec_3d_space(degree);
    transforms = create_nedelec_3d_entity_transforms(degree, variant);
    std::tie(x, M) = create_nedelec_3d_interpolation(degree, variant);
    break;
  }
  default:
    throw std::runtime_error("Invalid celltype in Nedelec");
  }

  const std::size_t tdim = cell::topological_dimension(celltype);
  const xt::xtensor<double, 3> coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, {M[1], M[2], M[3]}, {x[1], x[2], x[3]}, degree);
  return FiniteElement(element::family::N1E, celltype, degree, {tdim}, coeffs,
                       transforms, x, M, maps::type::covariantPiola);
}
//-----------------------------------------------------------------------------
FiniteElement basix::create_nedelec2(cell::type celltype, int degree,
                                     element::variant variant)
{
  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;
  std::vector<xt::xtensor<double, 2>> entity_transformations;
  switch (celltype)
  {
  case cell::type::triangle:
  {
    std::tie(x, M) = create_nedelec2_2d_interpolation(degree, variant);
    entity_transformations
        = create_nedelec2_2d_entity_transformations(degree, variant);
    break;
  }
  case cell::type::tetrahedron:
  {
    std::tie(x, M) = create_nedelec2_3d_interpolation(degree, variant);
    entity_transformations
        = create_nedelec2_3d_entity_transformations(degree, variant);
    break;
  }
  default:
    throw std::runtime_error("Invalid celltype in Nedelec");
  }

  const std::size_t tdim = cell::topological_dimension(celltype);

  const std::size_t psize = polyset::dim(celltype, degree);
  xt::xtensor<double, 2> wcoeffs = xt::eye<double>(tdim * psize);
  const xt::xtensor<double, 3> coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, {M[1], M[2], M[3]}, {x[1], x[2], x[3]}, degree);
  return FiniteElement(element::family::N2E, celltype, degree, {tdim}, coeffs,
                       entity_transformations, x, M,
                       maps::type::covariantPiola);
}
//-----------------------------------------------------------------------------
