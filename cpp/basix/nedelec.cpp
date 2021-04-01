// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "nedelec.h"
#include "element-families.h"
#include "lagrange.h"
#include "maps.h"
#include "moments.h"
#include "polyset.h"
#include "quadrature.h"
#include "raviart-thomas.h"
#include <numeric>
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
  auto [Qpts, _Qwts] = quadrature::make_quadrature(
      "default", cell::type::triangle, 2 * degree);
  auto Qwts = xt::adapt(_Qwts);
  xt::xtensor<double, 2> Pkp1_at_Qpts
      = xt::view(polyset::tabulate(cell::type::triangle, degree, 0, Qpts), 0,
                 xt::all(), xt::all());

  const std::size_t psize = Pkp1_at_Qpts.shape(1);

  // Create coefficients for order (degree-1) vector polynomials
  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({nv * 2 + ns, psize * 2});
  xt::view(wcoeffs, xt::range(0, nv), xt::range(0, nv)) = xt::eye<double>(nv);
  xt::view(wcoeffs, xt::range(nv, 2 * nv), xt::range(psize, psize + nv))
      = xt::eye<double>(nv);

  // Create coefficients for the additional Nedelec polynomials
  for (std::size_t i = 0; i < ns; ++i)
  {
    auto p = xt::col(Pkp1_at_Qpts, ns0 + i);
    for (std::size_t k = 0; k < psize; ++k)
    {
      auto pk = xt::col(Pkp1_at_Qpts, k);
      wcoeffs(2 * nv + i, k) = xt::sum(Qwts * p * xt::col(Qpts, 1) * pk)();
      wcoeffs(2 * nv + i, k + psize)
          = xt::sum(-Qwts * p * xt::col(Qpts, 0) * pk)();
    }
  }

  return wcoeffs;
}
//-----------------------------------------------------------------------------
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
create_nedelec_2d_interpolation(int degree)
{
  // dof counter
  const int quad_deg = 5 * degree;

  // Integral representation for the boundary (edge) dofs
  xt::xtensor<double, 2> points_1d;
  xt::xtensor<double, 2> matrix_1d;
  std::tie(points_1d, matrix_1d) = moments::make_tangent_integral_moments(
      create_dlagrange(cell::type::interval, degree - 1), cell::type::triangle,
      2, quad_deg);

  xt::xtensor<double, 2> points_2d, matrix_2d;
  if (degree > 1)
  {
    // Interior integral moment
    std::tie(points_2d, matrix_2d) = moments::make_integral_moments(
        create_dlagrange(cell::type::triangle, degree - 2),
        cell::type::triangle, 2, quad_deg);
  }

  return combine_interpolation_data(points_1d, points_2d, {}, matrix_1d,
                                    matrix_2d, {}, 2, 2);
}
//-----------------------------------------------------------------------------
std::pair<std::vector<xt::xtensor<double, 3>>,
          std::vector<xt::xtensor<double, 4>>>
create_nedelec_2d_interpolation_new(int degree)
{
  // dof counter
  const int quad_deg = 5 * degree;

  std::vector<xt::xtensor<double, 3>> x(2);
  std::vector<xt::xtensor<double, 4>> M(2);

  // Integral representation for the boundary (edge) dofs
  std::tie(x[0], M[0]) = moments::make_tangent_integral_moments_new(
      create_dlagrange(cell::type::interval, degree - 1), cell::type::triangle,
      2, quad_deg);

  if (degree > 1)
  {
    std::tie(x[1], M[1]) = moments::make_integral_moments_new(
        create_dlagrange(cell::type::triangle, degree - 2),
        cell::type::triangle, 2, quad_deg);
  }

  return {x, M};
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 3> create_nedelec_2d_base_transforms(int degree)
{
  const std::size_t ndofs = degree * (degree + 2);
  auto base_transformations
      = xt::tile(xt::expand_dims(xt::eye<double>(ndofs), 0), 3);

  xt::xtensor<double, 3> edge_transforms
      = moments::create_tangent_moment_dof_transformations(
          create_dlagrange(cell::type::interval, degree - 1));
  const std::size_t edge_dofs = edge_transforms.shape(1);
  for (std::size_t edge = 0; edge < 3; ++edge)
  {
    const std::size_t start = edge_dofs * edge;
    auto range = xt::range(start, start + edge_dofs);
    xt::view(base_transformations, edge, range, range)
        = xt::view(edge_transforms, 0, xt::all(), xt::all());
  }

  return base_transformations;
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
  auto [Qpts, _Qwts] = quadrature::make_quadrature(
      "default", cell::type::tetrahedron, 2 * degree);
  auto Qwts = xt::adapt(_Qwts);
  xt::xtensor<double, 2> Pkp1_at_Qpts
      = xt::view(polyset::tabulate(cell::type::tetrahedron, degree, 0, Qpts), 0,
                 xt::all(), xt::all());
  const std::size_t psize = Pkp1_at_Qpts.shape(1);

  // Create coefficients for order (degree-1) polynomials
  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize * tdim});
  for (std::size_t i = 0; i < tdim; ++i)
  {
    auto range0 = xt::range(nv * i, nv * i + nv);
    auto range1 = xt::range(psize * i, psize * i + nv);
    xt::view(wcoeffs, range0, range1) = xt::eye<double>(nv);
  }

  // Create coefficients for additional Nedelec polynomials
  auto p0 = xt::col(Qpts, 0);
  auto p1 = xt::col(Qpts, 1);
  auto p2 = xt::col(Qpts, 2);
  for (std::size_t i = 0; i < ns; ++i)
  {
    auto p = xt::col(Pkp1_at_Qpts, ns0 + i);
    for (std::size_t k = 0; k < psize; ++k)
    {
      const double w = xt::sum(Qwts * p * p2 * xt::col(Pkp1_at_Qpts, k))();

      // Don't include polynomials (*, *, 0) that are dependant
      if (i >= ns_remove)
        wcoeffs(tdim * nv + i - ns_remove, psize + k) = -w;
      wcoeffs(tdim * nv + i + ns - ns_remove, k) = w;
    }
  }

  for (std::size_t i = 0; i < ns; ++i)
  {
    auto p = xt::col(Pkp1_at_Qpts, ns0 + i);
    for (std::size_t k = 0; k < psize; ++k)
    {
      const double w = xt::sum(Qwts * p * p1 * xt::col(Pkp1_at_Qpts, k))();
      wcoeffs(tdim * nv + i + ns * 2 - ns_remove, k) = -w;

      // Don't include polynomials (*, *, 0) that are dependant
      if (i >= ns_remove)
        wcoeffs(tdim * nv + i - ns_remove, psize * 2 + k) = w;
    }
  }

  for (std::size_t i = 0; i < ns; ++i)
  {
    auto p = xt::col(Pkp1_at_Qpts, ns0 + i);
    for (std::size_t k = 0; k < psize; ++k)
    {
      const double w = xt::sum(Qwts * p * p0 * xt::col(Pkp1_at_Qpts, k))();
      wcoeffs(tdim * nv + i + ns - ns_remove, psize * 2 + k) = -w;
      wcoeffs(tdim * nv + i + ns * 2 - ns_remove, psize + k) = w;
    }
  }

  return wcoeffs;
}
//-----------------------------------------------------------------------------
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
create_nedelec_3d_interpolation(int degree)
{
  // Number of dofs and interpolation points
  int quad_deg = 5 * degree;

  xt::xtensor<double, 2> points_1d, matrix_1d;
  std::tie(points_1d, matrix_1d) = moments::make_tangent_integral_moments(
      create_dlagrange(cell::type::interval, degree - 1),
      cell::type::tetrahedron, 3, quad_deg);

  xt::xtensor<double, 2> points_2d, matrix_2d;
  if (degree > 1)
  {
    std::tie(points_2d, matrix_2d) = moments::make_integral_moments(
        create_dlagrange(cell::type::triangle, degree - 2),
        cell::type::tetrahedron, 3, quad_deg);
  }

  xt::xtensor<double, 2> points_3d, matrix_3d;
  if (degree > 2)
  {
    std::tie(points_3d, matrix_3d) = moments::make_integral_moments(
        create_dlagrange(cell::type::tetrahedron, degree - 3),
        cell::type::tetrahedron, 3, quad_deg);
  }

  return combine_interpolation_data(points_1d, points_2d, points_3d, matrix_1d,
                                    matrix_2d, matrix_3d, 3, 3);
}
//-----------------------------------------------------------------------------
std::pair<std::vector<xt::xtensor<double, 3>>,
          std::vector<xt::xtensor<double, 4>>>
create_nedelec_3d_interpolation_new(int degree)
{
  // Number of dofs and interpolation points
  int quad_deg = 5 * degree;

  std::vector<xt::xtensor<double, 3>> x(3);
  std::vector<xt::xtensor<double, 4>> M(3);

  std::tie(x[0], M[0]) = moments::make_tangent_integral_moments_new(
      create_dlagrange(cell::type::interval, degree - 1),
      cell::type::tetrahedron, 3, quad_deg);

  if (degree > 1)
  {
    std::tie(x[1], M[1]) = moments::make_integral_moments_new(
        create_dlagrange(cell::type::triangle, degree - 2),
        cell::type::tetrahedron, 3, quad_deg);
  }

  if (degree > 2)
  {
    std::tie(x[2], M[2]) = moments::make_integral_moments_new(
        create_dlagrange(cell::type::tetrahedron, degree - 3),
        cell::type::tetrahedron, 3, quad_deg);
  }

  return {x, M};
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 3> create_nedelec_3d_base_transforms(int degree)
{
  const std::size_t ndofs = 6 * degree + 4 * degree * (degree - 1)
                            + (degree - 2) * (degree - 1) * degree / 2;
  auto base_transformations
      = xt::tile(xt::expand_dims(xt::eye<double>(ndofs), 0), 14);

  xt::xtensor<double, 3> edge_transforms
      = moments::create_tangent_moment_dof_transformations(
          create_dlagrange(cell::type::interval, degree - 1));
  const std::size_t edge_dofs = edge_transforms.shape(1);
  for (std::size_t edge = 0; edge < 6; ++edge)
  {
    const std::size_t start = edge_dofs * edge;
    auto range = xt::range(start, start + edge_dofs);
    xt::view(base_transformations, edge, range, range)
        = xt::view(edge_transforms, 0, xt::all(), xt::all());
  }

  // Faces
  if (degree > 1)
  {
    xt::xtensor<double, 3> face_transforms
        = moments::create_moment_dof_transformations(
            create_dlagrange(cell::type::triangle, degree - 2));

    const std::size_t face_dofs = face_transforms.shape(1);
    for (std::size_t face = 0; face < 4; ++face)
    {
      const std::size_t start = edge_dofs * 6 + face_dofs * face;
      auto range = xt::range(start, start + face_dofs);
      xt::view(base_transformations, 6 + 2 * face, range, range)
          = xt::view(face_transforms, 0, xt::all(), xt::all());
      xt::view(base_transformations, 6 + 2 * face + 1, range, range)
          = xt::view(face_transforms, 1, xt::all(), xt::all());
    }
  }

  return base_transformations;
}
//-----------------------------------------------------------------------------
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
create_nedelec2_2d_interpolation(int degree)
{
  // Number of dofs and interpolation points
  int quad_deg = 5 * degree;
  xt::xtensor<double, 2> points_1d, matrix_1d;
  std::tie(points_1d, matrix_1d) = moments::make_tangent_integral_moments(
      create_dlagrange(cell::type::interval, degree), cell::type::triangle, 2,
      quad_deg);

  xt::xtensor<double, 2> points_2d, matrix_2d;
  if (degree > 1)
  {
    std::tie(points_2d, matrix_2d) = moments::make_dot_integral_moments(
        create_rt(cell::type::triangle, degree - 1), cell::type::triangle, 2,
        quad_deg);
  }

  return combine_interpolation_data(points_1d, points_2d, {}, matrix_1d,
                                    matrix_2d, {}, 2, 2);
}
//-----------------------------------------------------------------------------
std::pair<std::vector<xt::xtensor<double, 3>>,
          std::vector<xt::xtensor<double, 4>>>
create_nedelec2_2d_interpolation_new(int degree)
{
  int quad_deg = 5 * degree;

  std::vector<xt::xtensor<double, 3>> x(2);
  std::vector<xt::xtensor<double, 4>> M(2);

  std::tie(x[0], M[0]) = moments::make_tangent_integral_moments_new(
      create_dlagrange(cell::type::interval, degree), cell::type::triangle, 2,
      quad_deg);
  if (degree > 1)
  {
    std::tie(x[1], M[1]) = moments::make_dot_integral_moments_new(
        create_rt(cell::type::triangle, degree - 1), cell::type::triangle, 2,
        quad_deg);
  }

  return {x, M};
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 3> create_nedelec2_2d_base_transformations(int degree)
{
  const std::size_t ndofs = (degree + 1) * (degree + 2);
  xt::xtensor<double, 3> base_transformations
      = xt::zeros<double>({static_cast<std::size_t>(3), ndofs, ndofs});
  for (std::size_t i = 0; i < base_transformations.shape(0); ++i)
  {
    xt::view(base_transformations, i, xt::all(), xt::all())
        = xt::eye<double>(ndofs);
  }

  xt::xtensor<double, 3> edge_transforms
      = moments::create_tangent_moment_dof_transformations(
          create_dlagrange(cell::type::interval, degree));
  const std::size_t edge_dofs = edge_transforms.shape(1);
  for (std::size_t edge = 0; edge < 3; ++edge)
  {
    const std::size_t start = edge_dofs * edge;
    auto range = xt::range(start, start + edge_dofs);
    xt::view(base_transformations, edge, range, range)
        = xt::view(edge_transforms, 0, xt::all(), xt::all());
  }

  return base_transformations;
}
//-----------------------------------------------------------------------------
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
create_nedelec2_3d_interpolation(int degree)
{
  // Create quadrature scheme on the edge
  int quad_deg = 5 * degree;

  // Integral representation for the boundary (edge) dofs
  xt::xtensor<double, 2> points_1d, matrix_1d;
  std::tie(points_1d, matrix_1d) = moments::make_tangent_integral_moments(
      create_dlagrange(cell::type::interval, degree), cell::type::tetrahedron,
      3, quad_deg);

  xt::xtensor<double, 2> points_2d, matrix_2d;
  if (degree > 1)
  {
    // Integral moments on faces
    std::tie(points_2d, matrix_2d) = moments::make_dot_integral_moments(
        create_rt(cell::type::triangle, degree - 1), cell::type::tetrahedron, 3,
        quad_deg);
  }

  xt::xtensor<double, 2> points_3d, matrix_3d;
  if (degree > 2)
  {
    // Interior integral moment
    std::tie(points_3d, matrix_3d) = moments::make_dot_integral_moments(
        create_rt(cell::type::tetrahedron, degree - 2), cell::type::tetrahedron,
        3, quad_deg);
  }

  return combine_interpolation_data(points_1d, points_2d, points_3d, matrix_1d,
                                    matrix_2d, matrix_3d, 3, 3);
}
//-----------------------------------------------------------------------------
std::pair<std::vector<xt::xtensor<double, 3>>,
          std::vector<xt::xtensor<double, 4>>>
create_nedelec2_3d_interpolation_new(int degree)
{
  // Create quadrature scheme on the edge
  int quad_deg = 5 * degree;

  std::vector<xt::xtensor<double, 3>> x(3);
  std::vector<xt::xtensor<double, 4>> M(3);

  // Integral representation for the boundary (edge) dofs
  std::tie(x[0], M[0]) = moments::make_tangent_integral_moments_new(
      create_dlagrange(cell::type::interval, degree), cell::type::tetrahedron,
      3, quad_deg);

  if (degree > 1)
  {
    // Integral moments on faces
    std::tie(x[1], M[1]) = moments::make_dot_integral_moments_new(
        create_rt(cell::type::triangle, degree - 1), cell::type::tetrahedron, 3,
        quad_deg);
  }

  if (degree > 2)
  {
    // Interior integral moment
    std::tie(x[2], M[2]) = moments::make_dot_integral_moments_new(
        create_rt(cell::type::tetrahedron, degree - 2), cell::type::tetrahedron,
        3, quad_deg);
  }

  return {x, M};
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 3> create_nedelec2_3d_base_transformations(int degree)
{
  const std::size_t ndofs = (degree + 1) * (degree + 2) * (degree + 3) / 2;
  xt::xtensor<double, 3> base_transformations
      = xt::zeros<double>({static_cast<std::size_t>(14), ndofs, ndofs});
  for (std::size_t i = 0; i < base_transformations.shape(0); ++i)
  {
    xt::view(base_transformations, i, xt::all(), xt::all())
        = xt::eye<double>(ndofs);
  }

  xt::xtensor<double, 3> edge_transforms
      = moments::create_tangent_moment_dof_transformations(
          create_dlagrange(cell::type::interval, degree));
  const std::size_t edge_dofs = edge_transforms.shape(1);
  for (std::size_t edge = 0; edge < 6; ++edge)
  {
    const std::size_t start = edge_dofs * edge;
    auto range = xt::range(start, start + edge_dofs);
    xt::view(base_transformations, edge, range, range)
        = xt::view(edge_transforms, 0, xt::all(), xt::all());
  }

  // Faces
  if (degree > 1)
  {
    xt::xtensor<double, 3> face_transforms
        = moments::create_dot_moment_dof_transformations(
            create_rt(cell::type::triangle, degree - 1));
    const std::size_t face_dofs = face_transforms.shape(1);
    for (std::size_t face = 0; face < 4; ++face)
    {
      const std::size_t start = edge_dofs * 6 + face_dofs * face;
      auto range = xt::range(start, start + face_dofs);
      xt::view(base_transformations, 6 + 2 * face, range, range)
          = xt::view(face_transforms, 0, xt::all(), xt::all());
      xt::view(base_transformations, 6 + 2 * face + 1, range, range)
          = xt::view(face_transforms, 1, xt::all(), xt::all());
    }
  }

  return base_transformations;
}

} // namespace

//-----------------------------------------------------------------------------
FiniteElement basix::create_nedelec(cell::type celltype, int degree)
{
  std::vector<xt::xtensor<double, 3>> x;
  std::vector<xt::xtensor<double, 4>> M;

  xt::xtensor<double, 2> wcoeffs, points, interp_matrix;
  xt::xtensor<double, 3> transforms;
  switch (celltype)
  {
  case cell::type::triangle:
    wcoeffs = create_nedelec_2d_space(degree);
    std::tie(points, interp_matrix) = create_nedelec_2d_interpolation(degree);
    transforms = create_nedelec_2d_base_transforms(degree);
    std::tie(x, M) = create_nedelec_2d_interpolation_new(degree);
    break;
  case cell::type::tetrahedron:
    wcoeffs = create_nedelec_3d_space(degree);
    std::tie(points, interp_matrix) = create_nedelec_3d_interpolation(degree);
    transforms = create_nedelec_3d_base_transforms(degree);
    std::tie(x, M) = create_nedelec_3d_interpolation_new(degree);
    break;
  default:
    throw std::runtime_error("Invalid celltype in Nedelec");
  }

  // Nedelec has d dofs on each edge, d(d-1) on each face
  // and d(d-1)(d-2)/2 on the interior in 3D
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  std::vector<std::vector<int>> entity_dofs(topology.size());
  entity_dofs[0].resize(topology[0].size(), 0);
  entity_dofs[1].resize(topology[1].size(), degree);
  entity_dofs[2].resize(topology[2].size(), degree * (degree - 1));
  const std::size_t tdim = cell::topological_dimension(celltype);
  if (tdim > 2)
    entity_dofs[3] = {degree * (degree - 1) * (degree - 2) / 2};

  const xt::xtensor<double, 3> coeffs
      = compute_expansion_coefficients(celltype, wcoeffs, M, x, degree);
  return FiniteElement(element::family::N1E, celltype, degree, {tdim}, coeffs,
                       entity_dofs, transforms, points, interp_matrix,
                       maps::type::covariantPiola);
}
//-----------------------------------------------------------------------------
FiniteElement basix::create_nedelec2(cell::type celltype, int degree)
{
  std::vector<xt::xtensor<double, 3>> x;
  std::vector<xt::xtensor<double, 4>> M;

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t psize = polyset::dim(celltype, degree);
  xt::xtensor<double, 2> points, interp_matrix;
  xt::xtensor<double, 3> base_transformations;
  switch (celltype)
  {
  case cell::type::triangle:
    std::tie(points, interp_matrix) = create_nedelec2_2d_interpolation(degree);
    std::tie(x, M) = create_nedelec2_2d_interpolation_new(degree);
    base_transformations = create_nedelec2_2d_base_transformations(degree);
    break;
  case cell::type::tetrahedron:
    std::tie(points, interp_matrix) = create_nedelec2_3d_interpolation(degree);
    std::tie(x, M) = create_nedelec2_3d_interpolation_new(degree);
    base_transformations = create_nedelec2_3d_base_transformations(degree);
    break;
  default:
    throw std::runtime_error("Invalid celltype in Nedelec");
  }

  // Nedelec(2nd kind) has (d + 1) dofs on each edge, (d + 1)(d - 1) on
  // each face and (d - 2)(d - 1)(d + 1)/2 on the interior in 3D
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  std::vector<std::vector<int>> entity_dofs(topology.size());
  entity_dofs[0].resize(topology[0].size(), 0);
  entity_dofs[1].resize(topology[1].size(), degree + 1);
  entity_dofs[2].resize(topology[2].size(), (degree + 1) * (degree - 1));
  if (tdim > 2)
    entity_dofs[3] = {(degree - 2) * (degree - 1) * (degree + 1) / 2};

  xt::xtensor<double, 2> wcoeffs = xt::eye<double>(tdim * psize);
  const xt::xtensor<double, 3> coeffs
      = compute_expansion_coefficients(celltype, wcoeffs, M, x, degree);
  return FiniteElement(element::family::N2E, celltype, degree, {tdim}, coeffs,
                       entity_dofs, base_transformations, points, interp_matrix,
                       maps::type::covariantPiola);
}
//-----------------------------------------------------------------------------
