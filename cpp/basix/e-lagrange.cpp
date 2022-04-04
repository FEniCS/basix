// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-lagrange.h"
#include "dof-transformations.h"
#include "lattice.h"
#include "maps.h"
#include "moments.h"
#include "polynomials.h"
#include "polyset.h"
#include "quadrature.h"
#include <xtensor/xbuilder.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

namespace
{
//----------------------------------------------------------------------------
std::tuple<lattice::type, lattice::simplex_method, bool>
variant_to_lattice(cell::type celltype, element::lagrange_variant variant)
{
  switch (variant)
  {
  case element::lagrange_variant::equispaced:
    return {lattice::type::equispaced, lattice::simplex_method::none, true};
  case element::lagrange_variant::gll_warped:
    return {lattice::type::gll, lattice::simplex_method::warp, true};
  case element::lagrange_variant::gll_isaac:
    return {lattice::type::gll, lattice::simplex_method::isaac, true};
  case element::lagrange_variant::gll_centroid:
    return {lattice::type::gll, lattice::simplex_method::centroid, true};
  case element::lagrange_variant::chebyshev_warped:
  {
    if (celltype == cell::type::interval
        or celltype == cell::type::quadrilateral
        or celltype == cell::type::hexahedron)
      return {lattice::type::chebyshev, lattice::simplex_method::none, false};
    // TODO: is this the best thing to do for simplices?
    return {lattice::type::chebyshev_plus_endpoints,
            lattice::simplex_method::warp, false};
  }
  case element::lagrange_variant::chebyshev_isaac:
  {
    if (celltype == cell::type::interval
        or celltype == cell::type::quadrilateral
        or celltype == cell::type::hexahedron)
      return {lattice::type::chebyshev, lattice::simplex_method::none, false};
    // TODO: is this the best thing to do for simplices?
    return {lattice::type::chebyshev_plus_endpoints,
            lattice::simplex_method::isaac, false};
  }
  case element::lagrange_variant::chebyshev_centroid:
    return {lattice::type::chebyshev, lattice::simplex_method::centroid, false};
  case element::lagrange_variant::gl_warped:
  {
    if (celltype == cell::type::interval
        or celltype == cell::type::quadrilateral
        or celltype == cell::type::hexahedron)
      return {lattice::type::gl, lattice::simplex_method::none, false};
    // TODO: is this the best thing to do for simplices?
    return {lattice::type::gl_plus_endpoints, lattice::simplex_method::warp,
            false};
  }
  case element::lagrange_variant::gl_isaac:
  {
    if (celltype == cell::type::interval
        or celltype == cell::type::quadrilateral
        or celltype == cell::type::hexahedron)
      return {lattice::type::gl, lattice::simplex_method::none, false};
    // TODO: is this the best thing to do for simplices?
    return {lattice::type::gl_plus_endpoints, lattice::simplex_method::isaac,
            false};
  }
  case element::lagrange_variant::gl_centroid:
    return {lattice::type::gl, lattice::simplex_method::centroid, false};
  default:
    throw std::runtime_error("Unsupported variant");
  }
}
//-----------------------------------------------------------------------------
FiniteElement create_d_lagrange(cell::type celltype, int degree,
                                element::lagrange_variant variant,
                                lattice::type lattice_type,
                                lattice::simplex_method simplex_method)
{
  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t ndofs = polyset::dim(celltype, degree);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  if (celltype == cell::type::prism or celltype == cell::type::pyramid)
  {
    throw std::runtime_error(
        "This variant is not yet supported on prisms and pyramids.");
  }

  const int lattice_degree
      = celltype == cell::type::triangle
            ? degree + 3
            : (celltype == cell::type::tetrahedron ? degree + 4 : degree + 2);

  // Create points in interior
  auto pt = lattice::create(celltype, lattice_degree, lattice_type, false,
                            simplex_method);
  x[tdim].push_back(pt);
  const std::size_t num_dofs = pt.shape(0);
  std::array<std::size_t, 3> s = {num_dofs, 1, num_dofs};
  M[tdim].push_back(xt::xtensor<double, 3>(s));
  xt::view(M[tdim][0], xt::all(), 0, xt::all()) = xt::eye<double>(num_dofs);

  std::map<cell::type, xt::xtensor<double, 3>> entity_transformations;

  // Entity transformations for edges
  if (tdim > 1)
  {
    const std::array<std::size_t, 3> shape = {1, 0, 0};
    xt::xtensor<double, 3> et = xt::zeros<double>(shape);
    entity_transformations[cell::type::interval] = et;
  }

  // Entity transformations for triangular faces
  if (celltype == cell::type::tetrahedron or celltype == cell::type::prism
      or celltype == cell::type::pyramid)
  {
    const std::array<std::size_t, 3> shape = {2, 0, 0};
    xt::xtensor<double, 3> ft = xt::zeros<double>(shape);
    entity_transformations[cell::type::triangle] = ft;
  }

  // Entity transformations for quadrilateral faces
  if (celltype == cell::type::hexahedron or celltype == cell::type::prism
      or celltype == cell::type::pyramid)
  {
    const std::array<std::size_t, 3> shape = {2, 0, 0};
    xt::xtensor<double, 3> ft = xt::zeros<double>(shape);
    entity_transformations[cell::type::quadrilateral] = ft;
  }

  return FiniteElement(element::family::P, celltype, degree, {},
                       xt::eye<double>(ndofs), entity_transformations, x, M,
                       maps::type::identity, true, degree, degree, {}, variant);
}
//----------------------------------------------------------------------------
std::vector<std::tuple<std::vector<FiniteElement>, std::vector<int>>>
create_tensor_product_factors(cell::type celltype, int degree,
                              element::lagrange_variant variant)
{
  if (celltype == cell::type::quadrilateral)
  {
    FiniteElement sub_element
        = element::create_lagrange(cell::type::interval, degree, variant, true);
    std::vector<int> perm((degree + 1) * (degree + 1));
    if (degree == 0)
      perm[0] = 0;
    else
    {
      int p = 0;
      int n = degree - 1;
      perm[p++] = 0;
      perm[p++] = 2;
      for (int i = 0; i < n; ++i)
        perm[p++] = 4 + n + i;
      perm[p++] = 1;
      perm[p++] = 3;
      for (int i = 0; i < n; ++i)
        perm[p++] = 4 + 2 * n + i;
      for (int i = 0; i < n; ++i)
      {
        perm[p++] = 4 + i;
        perm[p++] = 4 + 3 * n + i;
        for (int j = 0; j < n; ++j)
          perm[p++] = 4 + i + (4 + j) * n;
      }
    }
    return {{{sub_element, sub_element}, perm}};
  }
  if (celltype == cell::type::hexahedron)
  {
    FiniteElement sub_element
        = element::create_lagrange(cell::type::interval, degree, variant, true);
    std::vector<int> perm((degree + 1) * (degree + 1) * (degree + 1));
    if (degree == 0)
      perm[0] = 0;
    else
    {
      int p = 0;
      int n = degree - 1;
      perm[p++] = 0;
      perm[p++] = 4;
      for (int i = 0; i < n; ++i)
        perm[p++] = 8 + 2 * n + i;
      perm[p++] = 2;
      perm[p++] = 6;
      for (int i = 0; i < n; ++i)
        perm[p++] = 8 + 6 * n + i;
      for (int i = 0; i < n; ++i)
      {
        perm[p++] = 8 + n + i;
        perm[p++] = 8 + 9 * n + i;
        for (int j = 0; j < n; ++j)
          perm[p++] = 8 + 12 * n + 2 * n * n + i + n * j;
      }
      perm[p++] = 1;
      perm[p++] = 5;
      for (int i = 0; i < n; ++i)
        perm[p++] = 8 + 4 * n + i;
      perm[p++] = 3;
      perm[p++] = 7;
      for (int i = 0; i < n; ++i)
        perm[p++] = 8 + 7 * n + i;
      for (int i = 0; i < n; ++i)
      {
        perm[p++] = 8 + 3 * n + i;
        perm[p++] = 8 + 10 * n + i;
        for (int j = 0; j < n; ++j)
          perm[p++] = 8 + 12 * n + 3 * n * n + i + n * j;
      }
      for (int i = 0; i < n; ++i)
      {
        perm[p++] = 8 + i;
        perm[p++] = 8 + 8 * n + i;
        for (int j = 0; j < n; ++j)
          perm[p++] = 8 + 12 * n + n * n + i + n * j;
        perm[p++] = 8 + 5 * n + i;
        perm[p++] = 8 + 11 * n + i;
        for (int j = 0; j < n; ++j)
          perm[p++] = 8 + 12 * n + 4 * n * n + i + n * j;
        for (int j = 0; j < n; ++j)
        {
          perm[p++] = 8 + 12 * n + i + n * j;
          perm[p++] = 8 + 12 * n + 5 * n * n + i + n * j;
          for (int k = 0; k < n; ++k)
          {
            perm[p++] = 8 + 12 * n + 6 * n * n + i + n * j + n * n * k;
          }
        }
      }
    }
    return {{{sub_element, sub_element, sub_element}, perm}};
  }
  return {};
}
//----------------------------------------------------------------------------
xt::xtensor<double, 2> vtk_triangle_points(int degree)
{
  const double d = static_cast<double>(1) / static_cast<double>(degree + 3);
  if (degree == 0)
    return {{d, d}};

  const std::size_t npoints = polyset::dim(cell::type::triangle, degree);
  xt::xtensor<double, 2> out({npoints, 2});

  out(0, 0) = d;
  out(0, 1) = d;
  out(1, 0) = 1 - 2 * d;
  out(1, 1) = d;
  out(2, 0) = d;
  out(2, 1) = 1 - 2 * d;
  int n = 3;
  if (degree >= 2)
  {
    for (int i = 1; i < degree; ++i)
    {
      out(n, 0) = d + ((1 - 3 * d) * i) / (degree);
      out(n, 1) = d;
      ++n;
    }
    for (int i = 1; i < degree; ++i)
    {
      out(n, 0) = d + ((1 - 3 * d) * (degree - i)) / (degree);
      out(n, 1) = d + ((1 - 3 * d) * i) / (degree);
      ++n;
    }
    for (int i = 1; i < degree; ++i)
    {
      out(n, 0) = d;
      out(n, 1) = d + ((1 - 3 * d) * (degree - i)) / (degree);
      ++n;
    }
  }
  if (degree >= 3)
  {
    xt::xtensor<double, 2> pts = vtk_triangle_points(degree - 3);
    for (std::size_t i = 0; i < pts.shape(0); ++i)
    {
      for (std::size_t j = 0; j < pts.shape(1); ++j)
        out(n, j) = d + (1 - 3 * d) * pts(i, j);
      ++n;
    }
  }

  return out;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> vtk_tetrahedron_points(int degree)
{
  const double d = static_cast<double>(1) / static_cast<double>(degree + 4);

  if (degree == 0)
    return {{d, d, d}};

  const std::size_t npoints = polyset::dim(cell::type::tetrahedron, degree);
  xt::xtensor<double, 2> out({npoints, 3});

  out(0, 0) = d;
  out(0, 1) = d;
  out(0, 2) = d;
  out(1, 0) = 1 - 3 * d;
  out(1, 1) = d;
  out(1, 2) = d;
  out(2, 0) = d;
  out(2, 1) = 1 - 3 * d;
  out(2, 2) = d;
  out(3, 0) = d;
  out(3, 1) = d;
  out(3, 2) = 1 - 3 * d;
  int n = 4;
  if (degree >= 2)
  {
    for (int i = 1; i < degree; ++i)
    {
      out(n, 0) = d + ((1 - 4 * d) * i) / (degree);
      out(n, 1) = d;
      out(n, 2) = d;
      ++n;
    }
    for (int i = 1; i < degree; ++i)
    {
      out(n, 0) = d + ((1 - 4 * d) * (degree - i)) / (degree);
      out(n, 1) = d + ((1 - 4 * d) * i) / (degree);
      out(n, 2) = d;
      ++n;
    }
    for (int i = 1; i < degree; ++i)
    {
      out(n, 0) = d;
      out(n, 1) = d + ((1 - 4 * d) * (degree - i)) / (degree);
      out(n, 2) = d;
      ++n;
    }
    for (int i = 1; i < degree; ++i)
    {
      out(n, 0) = d;
      out(n, 1) = d;
      out(n, 2) = d + ((1 - 4 * d) * i) / (degree);
      ++n;
    }
    for (int i = 1; i < degree; ++i)
    {
      out(n, 0) = d + ((1 - 4 * d) * (degree - i)) / (degree);
      out(n, 1) = d;
      out(n, 2) = d + ((1 - 4 * d) * i) / (degree);
      ++n;
    }
    for (int i = 1; i < degree; ++i)
    {
      out(n, 0) = d;
      out(n, 1) = d + ((1 - 4 * d) * (degree - i)) / (degree);
      out(n, 2) = d + ((1 - 4 * d) * i) / (degree);
      ++n;
    }
  }
  if (degree >= 3)
  {
    xt::xtensor<double, 2> pts = vtk_triangle_points(degree - 3);
    for (std::size_t i = 0; i < pts.shape(0); ++i)
    {
      out(n, 0) = d + pts(i, 0) * (1 - 4 * d);
      out(n, 1) = d;
      out(n, 2) = d + pts(i, 1) * (1 - 4 * d);
      ++n;
    }
    for (std::size_t i = 0; i < pts.shape(0); ++i)
    {
      out(n, 0) = 1 - 3 * d - (pts(i, 0) + pts(i, 1)) * (1 - 4 * d);
      out(n, 1) = d + pts(i, 0) * (1 - 4 * d);
      out(n, 2) = d + pts(i, 1) * (1 - 4 * d);
      ++n;
    }
    for (std::size_t i = 0; i < pts.shape(0); ++i)
    {
      out(n, 0) = d;
      out(n, 1) = d + pts(i, 0) * (1 - 4 * d);
      out(n, 2) = d + pts(i, 1) * (1 - 4 * d);
      ++n;
    }
    for (std::size_t i = 0; i < pts.shape(0); ++i)
    {
      out(n, 0) = d + pts(i, 0) * (1 - 4 * d);
      out(n, 1) = d + pts(i, 1) * (1 - 4 * d);
      out(n, 2) = d;
      ++n;
    }
  }
  if (degree >= 4)
  {
    xt::view(out, xt::range(n, npoints), xt::all())
        = vtk_tetrahedron_points(degree - 4);

    xt::xtensor<double, 2> pts = vtk_tetrahedron_points(degree - 4);
    for (std::size_t i = 0; i < pts.shape(0); ++i)
    {
      for (std::size_t j = 0; j < pts.shape(1); ++j)
        out(n, j) = d + (1 - 4 * d) * pts(i, j);
      ++n;
    }
  }

  return out;
}
//-----------------------------------------------------------------------------
FiniteElement create_vtk_element(cell::type celltype, int degree,
                                 bool discontinuous)
{
  if (celltype == cell::type::point)
    throw std::runtime_error("Invalid celltype");

  if (degree == 0)
  {
    throw std::runtime_error("Cannot create an order 0 VTK element.");
  }

  // DOF transformation don't yet work on this element, so throw runtime error
  // is trying to make continuous version
  if (!discontinuous)
  {
    throw std::runtime_error("Continuous VTK element not yet supported.");
  }

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t ndofs = polyset::dim(celltype, degree);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  for (std::size_t dim = 0; dim <= tdim; ++dim)
  {
    M[dim].resize(topology[dim].size());
    x[dim].resize(topology[dim].size());
  }

  switch (celltype)
  {
  case cell::type::interval:
  {
    // Points at vertices
    x[0][0] = {{0.}};
    x[0][1] = {{1.}};
    for (int i = 0; i < 2; ++i)
      M[0][i] = {{{1.}}};

    // Points on interval
    x[1][0] = xt::xtensor<double, 2>(
        {static_cast<std::size_t>(degree - 1), static_cast<std::size_t>(1)});
    for (int i = 1; i < degree; ++i)
      x[1][0](i - 1, 0) = static_cast<double>(i) / static_cast<double>(degree);

    M[1][0] = xt::xtensor<double, 3>({static_cast<std::size_t>(degree - 1), 1,
                                      static_cast<std::size_t>(degree - 1)});
    xt::view(M[1][0], xt::all(), 0, xt::all()) = xt::eye<double>(degree - 1);

    break;
  }
  case cell::type::triangle:
  {
    // Points at vertices
    x[0][0] = {{0., 0.}};
    x[0][1] = {{1., 0.}};
    x[0][2] = {{0., 1.}};
    for (int i = 0; i < 3; ++i)
      M[0][i] = {{{1.}}};

    // Points on edges
    std::array<std::size_t, 2> s
        = {static_cast<std::size_t>(degree - 1), static_cast<std::size_t>(2)};
    for (int i = 0; i < 3; ++i)
      x[1][i] = xt::xtensor<double, 2>(s);

    for (int i = 1; i < degree; ++i)
    {
      x[1][0](i - 1, 0) = static_cast<double>(i) / static_cast<double>(degree);
      x[1][0](i - 1, 1) = 0;

      x[1][1](i - 1, 0)
          = static_cast<double>(degree - i) / static_cast<double>(degree);
      x[1][1](i - 1, 1) = static_cast<double>(i) / static_cast<double>(degree);

      x[1][2](i - 1, 0) = 0;
      x[1][2](i - 1, 1)
          = static_cast<double>(degree - i) / static_cast<double>(degree);
    }

    for (int i = 0; i < 3; ++i)
    {
      M[1][i] = xt::xtensor<double, 3>({static_cast<std::size_t>(degree - 1), 1,
                                        static_cast<std::size_t>(degree - 1)});
      xt::view(M[1][i], xt::all(), 0, xt::all()) = xt::eye<double>(degree - 1);
    }

    // Points in triangle
    if (degree >= 3)
    {
      x[2][0] = vtk_triangle_points(degree - 3);
      M[2][0] = xt::xtensor<double, 3>({x[2][0].shape(0), 1, x[2][0].shape(0)});
      xt::view(M[2][0], xt::all(), 0, xt::all())
          = xt::eye<double>(x[2][0].shape(0));
    }
    else
    {
      x[2][0] = xt::xtensor<double, 2>({0, 2});
      M[2][0] = xt::xtensor<double, 3>({0, 1, 0});
    }

    break;
  }
  case cell::type::tetrahedron:
  {
    // Points at vertices
    x[0][0] = {{0., 0., 0.}};
    x[0][1] = {{1., 0., 0.}};
    x[0][2] = {{0., 1., 0.}};
    x[0][3] = {{0., 0., 1.}};
    for (int i = 0; i < 4; ++i)
      M[0][i] = {{{1.}}};

    // Points on edges
    std::array<std::size_t, 2> s
        = {static_cast<std::size_t>(degree - 1), static_cast<std::size_t>(3)};
    for (int i = 0; i < 6; ++i)
      x[1][i] = xt::xtensor<double, 2>(s);
    for (int i = 1; i < degree; ++i)
    {
      x[1][0](i - 1, 0) = static_cast<double>(i) / static_cast<double>(degree);
      x[1][0](i - 1, 1) = 0;
      x[1][0](i - 1, 2) = 0;

      x[1][1](i - 1, 0)
          = static_cast<double>(degree - i) / static_cast<double>(degree);
      x[1][1](i - 1, 1) = static_cast<double>(i) / static_cast<double>(degree);
      x[1][1](i - 1, 2) = 0;

      x[1][2](i - 1, 0) = 0;
      x[1][2](i - 1, 1)
          = static_cast<double>(degree - i) / static_cast<double>(degree);
      x[1][2](i - 1, 2) = 0;

      x[1][3](i - 1, 0) = 0;
      x[1][3](i - 1, 1) = 0;
      x[1][3](i - 1, 2) = static_cast<double>(i) / static_cast<double>(degree);

      x[1][4](i - 1, 0)
          = static_cast<double>(degree - i) / static_cast<double>(degree);
      x[1][4](i - 1, 1) = 0;
      x[1][4](i - 1, 2) = static_cast<double>(i) / static_cast<double>(degree);

      x[1][5](i - 1, 0) = 0;
      x[1][5](i - 1, 1)
          = static_cast<double>(degree - i) / static_cast<double>(degree);
      x[1][5](i - 1, 2) = static_cast<double>(i) / static_cast<double>(degree);
    }
    for (int i = 0; i < 6; ++i)
    {
      M[1][i] = xt::xtensor<double, 3>({static_cast<std::size_t>(degree - 1), 1,
                                        static_cast<std::size_t>(degree - 1)});
      xt::view(M[1][i], xt::all(), 0, xt::all()) = xt::eye<double>(degree - 1);
    }

    // Points on faces
    if (degree >= 3)
    {
      xt::xtensor<double, 2> pts = vtk_triangle_points(degree - 3);

      std::array<std::size_t, 2> s
          = {pts.shape(0), static_cast<std::size_t>(3)};
      for (int i = 0; i < 4; ++i)
        x[2][i] = xt::xtensor<double, 2>(s);

      for (std::size_t i = 0; i < pts.shape(0); ++i)
      {
        const double x0 = pts(i, 0);
        const double x1 = pts(i, 1);

        x[2][0](i, 0) = x0;
        x[2][0](i, 1) = 0;
        x[2][0](i, 2) = x1;

        x[2][1](i, 0) = 1 - x0 - x1;
        x[2][1](i, 1) = x0;
        x[2][1](i, 2) = x1;

        x[2][2](i, 0) = 0;
        x[2][2](i, 1) = x0;
        x[2][2](i, 2) = x1;

        x[2][3](i, 0) = x0;
        x[2][3](i, 1) = x1;
        x[2][3](i, 2) = 0;
      }

      for (int i = 0; i < 4; ++i)
      {
        M[2][i]
            = xt::xtensor<double, 3>({x[2][0].shape(0), 1, x[2][0].shape(0)});
        xt::view(M[2][i], xt::all(), 0, xt::all())
            = xt::eye<double>(x[2][0].shape(0));
      }
    }
    else
    {
      for (int i = 0; i < 4; ++i)
      {
        x[2][i] = xt::xtensor<double, 2>({0, 3});
        M[2][i] = xt::xtensor<double, 3>({0, 1, 0});
      }
    }

    if (degree >= 4)
    {
      x[3][0] = vtk_tetrahedron_points(degree - 4);
      M[3][0] = xt::xtensor<double, 3>({x[3][0].shape(0), 1, x[3][0].shape(0)});
      xt::view(M[3][0], xt::all(), 0, xt::all())
          = xt::eye<double>(x[3][0].shape(0));
    }
    else
    {
      x[3][0] = xt::xtensor<double, 2>({0, 3});
      M[3][0] = xt::xtensor<double, 3>({0, 1, 0});
    }

    break;
  }
  case cell::type::quadrilateral:
  {
    // Points at vertices
    x[0][0] = {{0., 0.}};
    x[0][1] = {{1., 0.}};
    x[0][2] = {{1., 1.}};
    x[0][3] = {{0., 1.}};
    for (int i = 0; i < 4; ++i)
      M[0][i] = {{{1.}}};

    // Points on edges
    std::array<std::size_t, 2> s = {static_cast<std::size_t>(degree - 1), 2};
    for (int i = 0; i < 4; ++i)
      x[1][i] = xt::xtensor<double, 2>(s);

    for (int i = 1; i < degree; ++i)
    {
      x[1][0](i - 1, 0) = static_cast<double>(i) / static_cast<double>(degree);
      x[1][0](i - 1, 1) = 0;

      x[1][1](i - 1, 0) = 1;
      x[1][1](i - 1, 1) = static_cast<double>(i) / static_cast<double>(degree);

      x[1][2](i - 1, 0) = static_cast<double>(i) / static_cast<double>(degree);
      x[1][2](i - 1, 1) = 1;

      x[1][3](i - 1, 0) = 0;
      x[1][3](i - 1, 1) = static_cast<double>(i) / static_cast<double>(degree);
    }

    for (int i = 0; i < 4; ++i)
    {
      M[1][i] = xt::xtensor<double, 3>({static_cast<std::size_t>(degree - 1), 1,
                                        static_cast<std::size_t>(degree - 1)});
      xt::view(M[1][i], xt::all(), 0, xt::all()) = xt::eye<double>(degree - 1);
    }

    // Points in quadrilateral
    x[2][0] = xt::xtensor<double, 2>(
        {static_cast<std::size_t>((degree - 1) * (degree - 1)), 2});

    int n = 0;
    for (int j = 1; j < degree; ++j)
      for (int i = 1; i < degree; ++i)
      {
        x[2][0](n, 0) = static_cast<double>(i) / static_cast<double>(degree);
        x[2][0](n, 1) = static_cast<double>(j) / static_cast<double>(degree);
        ++n;
      }

    M[2][0] = xt::xtensor<double, 3>({x[2][0].shape(0), 1, x[2][0].shape(0)});
    xt::view(M[2][0], xt::all(), 0, xt::all())
        = xt::eye<double>(x[2][0].shape(0));

    break;
  }
  case cell::type::hexahedron:
  {
    // Points at vertices
    x[0][0] = {{0., 0., 0.}};
    x[0][1] = {{1., 0., 0.}};
    x[0][2] = {{1., 1., 0.}};
    x[0][3] = {{0., 1., 0.}};
    x[0][4] = {{0., 0., 1.}};
    x[0][5] = {{1., 0., 1.}};
    x[0][6] = {{1., 1., 1.}};
    x[0][7] = {{0., 1., 1.}};
    for (int i = 0; i < 8; ++i)
      M[0][i] = {{{1.}}};

    // Points on edges
    std::array<std::size_t, 2> s = {static_cast<std::size_t>(degree - 1), 3};
    for (int i = 0; i < 12; ++i)
      x[1][i] = xt::xtensor<double, 2>(s);
    for (int i = 1; i < degree; ++i)
    {
      x[1][0](i - 1, 0) = static_cast<double>(i) / static_cast<double>(degree);
      x[1][0](i - 1, 1) = 0;
      x[1][0](i - 1, 2) = 0;

      x[1][1](i - 1, 0) = 1;
      x[1][1](i - 1, 1) = static_cast<double>(i) / static_cast<double>(degree);
      x[1][1](i - 1, 2) = 0;

      x[1][2](i - 1, 0) = static_cast<double>(i) / static_cast<double>(degree);
      x[1][2](i - 1, 1) = 1;
      x[1][2](i - 1, 2) = 0;

      x[1][3](i - 1, 0) = 0;
      x[1][3](i - 1, 1) = static_cast<double>(i) / static_cast<double>(degree);
      x[1][3](i - 1, 2) = 0;

      x[1][4](i - 1, 0) = static_cast<double>(i) / static_cast<double>(degree);
      x[1][4](i - 1, 1) = 0;
      x[1][4](i - 1, 2) = 1;

      x[1][5](i - 1, 0) = 1;
      x[1][5](i - 1, 1) = static_cast<double>(i) / static_cast<double>(degree);
      x[1][5](i - 1, 2) = 1;

      x[1][6](i - 1, 0) = static_cast<double>(i) / static_cast<double>(degree);
      x[1][6](i - 1, 1) = 1;
      x[1][6](i - 1, 2) = 1;

      x[1][7](i - 1, 0) = 0;
      x[1][7](i - 1, 1) = static_cast<double>(i) / static_cast<double>(degree);
      x[1][7](i - 1, 2) = 1;

      x[1][8](i - 1, 0) = 0;
      x[1][8](i - 1, 1) = 0;
      x[1][8](i - 1, 2) = static_cast<double>(i) / static_cast<double>(degree);

      x[1][9](i - 1, 0) = 1;
      x[1][9](i - 1, 1) = 0;
      x[1][9](i - 1, 2) = static_cast<double>(i) / static_cast<double>(degree);

      x[1][10](i - 1, 0) = 1;
      x[1][10](i - 1, 1) = 1;
      x[1][10](i - 1, 2) = static_cast<double>(i) / static_cast<double>(degree);

      x[1][11](i - 1, 0) = 0;
      x[1][11](i - 1, 1) = 1;
      x[1][11](i - 1, 2) = static_cast<double>(i) / static_cast<double>(degree);
    }
    for (int i = 0; i < 12; ++i)
    {
      M[1][i] = xt::xtensor<double, 3>({static_cast<std::size_t>(degree - 1), 1,
                                        static_cast<std::size_t>(degree - 1)});
      xt::view(M[1][i], xt::all(), 0, xt::all()) = xt::eye<double>(degree - 1);
    }

    // Points on faces
    std::array<std::size_t, 2> s2
        = {static_cast<std::size_t>((degree - 1) * (degree - 1)), 3};
    for (int i = 0; i < 6; ++i)
      x[2][i] = xt::xtensor<double, 2>(s2);

    int n = 0;
    for (int j = 1; j < degree; ++j)
      for (int i = 1; i < degree; ++i)
      {
        x[2][0](n, 0) = 0;
        x[2][0](n, 1) = static_cast<double>(i) / static_cast<double>(degree);
        x[2][0](n, 2) = static_cast<double>(j) / static_cast<double>(degree);

        x[2][1](n, 0) = 1;
        x[2][1](n, 1) = static_cast<double>(i) / static_cast<double>(degree);
        x[2][1](n, 2) = static_cast<double>(j) / static_cast<double>(degree);

        x[2][2](n, 0) = static_cast<double>(i) / static_cast<double>(degree);
        x[2][2](n, 1) = 0;
        x[2][2](n, 2) = static_cast<double>(j) / static_cast<double>(degree);

        x[2][3](n, 0) = static_cast<double>(i) / static_cast<double>(degree);
        x[2][3](n, 1) = 1;
        x[2][3](n, 2) = static_cast<double>(j) / static_cast<double>(degree);

        x[2][4](n, 0) = static_cast<double>(i) / static_cast<double>(degree);
        x[2][4](n, 1) = static_cast<double>(j) / static_cast<double>(degree);
        x[2][4](n, 2) = 0;

        x[2][5](n, 0) = static_cast<double>(i) / static_cast<double>(degree);
        x[2][5](n, 1) = static_cast<double>(j) / static_cast<double>(degree);
        x[2][5](n, 2) = 1;

        ++n;
      }

    for (int i = 0; i < 6; ++i)
    {
      M[2][i] = xt::xtensor<double, 3>({x[2][0].shape(0), 1, x[2][0].shape(0)});
      xt::view(M[2][i], xt::all(), 0, xt::all())
          = xt::eye<double>(x[2][0].shape(0));
    }

    // Points in hexahedron
    x[3][0] = xt::xtensor<double, 2>(
        {static_cast<std::size_t>((degree - 1) * (degree - 1) * (degree - 1)),
         3});

    n = 0;
    for (int k = 1; k < degree; ++k)
      for (int j = 1; j < degree; ++j)
        for (int i = 1; i < degree; ++i)
        {
          x[3][0](n, 0) = static_cast<double>(i) / static_cast<double>(degree);
          x[3][0](n, 1) = static_cast<double>(j) / static_cast<double>(degree);
          x[3][0](n, 2) = static_cast<double>(k) / static_cast<double>(degree);
          ++n;
        }

    M[3][0] = xt::xtensor<double, 3>({x[3][0].shape(0), 1, x[3][0].shape(0)});
    xt::view(M[3][0], xt::all(), 0, xt::all())
        = xt::eye<double>(x[3][0].shape(0));

    break;
  }
  default:
  {
    throw std::runtime_error("Unsupported cell type.");
  }
  }

  // Initialise empty transformations, as these will be removed anyway when the
  // discontinuous element is made
  std::map<cell::type, xt::xtensor<double, 3>> entity_transformations;
  // Entity transformations for edges
  if (tdim > 1)
  {
    entity_transformations[cell::type::interval]
        = xt::xtensor<double, 3>({1, 0, 0});
  }

  // Entity transformations for triangular faces
  if (celltype == cell::type::tetrahedron or celltype == cell::type::prism
      or celltype == cell::type::pyramid)
  {
    entity_transformations[cell::type::triangle]
        = xt::xtensor<double, 3>({2, 0, 0});
  }

  // Entity transformations for quadrilateral faces
  if (celltype == cell::type::hexahedron or celltype == cell::type::prism
      or celltype == cell::type::pyramid)
  {
    entity_transformations[cell::type::quadrilateral]
        = xt::xtensor<double, 3>({2, 0, 0});
  }

  if (discontinuous)
  {
    std::tie(x, M, entity_transformations)
        = element::make_discontinuous(x, M, entity_transformations, tdim, 1);
  }

  return FiniteElement(element::family::P, celltype, degree, {},
                       xt::eye<double>(ndofs), entity_transformations, x, M,
                       maps::type::identity, discontinuous, degree, degree);
}
//-----------------------------------------------------------------------------
FiniteElement create_legendre(cell::type celltype, int degree,
                              bool discontinuous)
{
  if (!discontinuous)
    throw std::runtime_error("Legendre variant must be discontinuous");

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t ndofs = polyset::dim(celltype, degree);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;
  std::map<cell::type, xt::xtensor<double, 3>> entity_transformations;

  auto [pts, _wts] = quadrature::make_quadrature(quadrature::type::Default,
                                                 celltype, degree * 2);
  auto wts = xt::adapt(_wts);

  // Evaluate moment space at quadrature points
  const xt::xtensor<double, 2> phi = polynomials::tabulate(
      polynomials::type::legendre, celltype, degree, pts);

  for (std::size_t dim = 0; dim <= tdim; ++dim)
  {
    M[dim].resize(topology[dim].size());
    x[dim].resize(topology[dim].size());
    if (dim < tdim)
    {
      if (dim > 0)
        entity_transformations[cell::sub_entity_type(celltype, dim, 0)]
            = xt::xtensor<double, 3>({dim, 0, 0});
      for (std::size_t e = 0; e < topology[dim].size(); ++e)
      {
        x[dim][e] = xt::xtensor<double, 2>({0, tdim});
        M[dim][e] = xt::xtensor<double, 3>({0, 1, 0});
      }
    }
  }
  x[tdim][0] = pts.dimension() == 1 ? pts.reshape({pts.shape(0), 1}) : pts;
  M[tdim][0] = xt::xtensor<double, 3>({ndofs, 1, pts.shape(0)});
  for (std::size_t i = 0; i < ndofs; ++i)
    xt::view(M[tdim][0], i, 0, xt::all()) = xt::col(phi, i) * wts;

  return FiniteElement(element::family::P, celltype, degree, {},
                       xt::eye<double>(ndofs), entity_transformations, x, M,
                       maps::type::identity, discontinuous, degree, degree, {},
                       element::lagrange_variant::legendre);
}
//-----------------------------------------------------------------------------
FiniteElement create_legendre_dpc(cell::type celltype, int degree,
                                  bool discontinuous)
{
  if (!discontinuous)
    throw std::runtime_error("Legendre variant must be discontinuous");

  cell::type simplex_type;
  switch (celltype)
  {
  case cell::type::quadrilateral:
    simplex_type = cell::type::triangle;
    break;
  case cell::type::hexahedron:
    simplex_type = cell::type::tetrahedron;
    break;
  default:
    throw std::runtime_error("Invalid cell type");
  }

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t psize = polyset::dim(celltype, degree);
  const std::size_t ndofs = polyset::dim(simplex_type, degree);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;
  std::map<cell::type, xt::xtensor<double, 3>> entity_transformations;

  auto [pts, _wts] = quadrature::make_quadrature(quadrature::type::Default,
                                                 celltype, degree * 2);
  auto wts = xt::adapt(_wts);

  // Evaluate moment space at quadrature points
  const xt::xtensor<double, 2> phi = polynomials::tabulate(
      polynomials::type::legendre, celltype, degree, pts);

  for (std::size_t dim = 0; dim <= tdim; ++dim)
  {
    M[dim].resize(topology[dim].size());
    x[dim].resize(topology[dim].size());
    if (dim < tdim)
    {
      if (dim > 0)
        entity_transformations[cell::sub_entity_type(celltype, dim, 0)]
            = xt::xtensor<double, 3>({dim, 0, 0});
      for (std::size_t e = 0; e < topology[dim].size(); ++e)
      {
        x[dim][e] = xt::xtensor<double, 2>({0, tdim});
        M[dim][e] = xt::xtensor<double, 3>({0, 1, 0});
      }
    }
  }
  x[tdim][0] = pts.dimension() == 1 ? pts.reshape({pts.shape(0), 1}) : pts;
  M[tdim][0] = xt::xtensor<double, 3>({ndofs, 1, pts.shape(0)});

  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize});

  if (celltype == cell::type::quadrilateral)
  {
    int row_n = 0;
    for (int i = 0; i <= degree; ++i)
    {
      for (int j = 0; j <= degree - i; ++j)
      {
        xt::view(M[tdim][0], row_n, 0, xt::all())
            = xt::col(phi, i * (degree + 1) + j) * wts;
        wcoeffs(row_n, i * (degree + 1) + j) = 1;
        ++row_n;
      }
    }
  }
  else
  {
    int row_n = 0;
    for (int i = 0; i <= degree; ++i)
    {
      for (int j = 0; j <= degree - i; ++j)
      {
        for (int k = 0; k <= degree - i - j; ++k)
        {
          xt::view(M[tdim][0], row_n, 0, xt::all())
              = xt::col(phi,
                        i * (degree + 1) * (degree + 1) + j * (degree + 1) + k)
                * wts;
          wcoeffs(row_n, i * (degree + 1) * (degree + 1) + j * (degree + 1) + k)
              = 1;
          ++row_n;
        }
      }
    }
  }

  return FiniteElement(element::family::DPC, celltype, degree, {}, wcoeffs,
                       entity_transformations, x, M, maps::type::identity,
                       discontinuous, degree, degree, {},
                       element::lagrange_variant::unset,
                       element::dpc_variant::legendre);
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> make_dpc_points(cell::type celltype, int degree,
                                       element::dpc_variant variant)
{
  if (degree == 0)
    return lattice::create(celltype, 0, lattice::type::equispaced, true);

  if (variant == element::dpc_variant::simplex_equispaced
      or variant == element::dpc_variant::simplex_gll)
  {
    lattice::type latticetype;
    lattice::simplex_method latticesm = lattice::simplex_method::isaac;
    if (variant == element::dpc_variant::simplex_equispaced)
      latticetype = lattice::type::equispaced;
    else if (variant == element::dpc_variant::simplex_gll)
      latticetype = lattice::type::gll;

    switch (celltype)
    {
    case cell::type::quadrilateral:
      return lattice::create(cell::type::triangle, degree, latticetype, true,
                             latticesm);
    case cell::type::hexahedron:
      return lattice::create(cell::type::tetrahedron, degree, latticetype, true,
                             latticesm);
    default:
      throw std::runtime_error("Invalid cell type");
    }
  }
  else if (variant == element::dpc_variant::horizontal_equispaced
           or variant == element::dpc_variant::horizontal_gll)
  {
    lattice::type latticetype;
    if (variant == element::dpc_variant::horizontal_equispaced)
      latticetype = lattice::type::equispaced;
    else if (variant == element::dpc_variant::horizontal_gll)
      latticetype = lattice::type::gll;

    switch (celltype)
    {
    case cell::type::quadrilateral:
    {
      xt::xtensor<double, 2> pts(
          {static_cast<std::size_t>((degree + 2) * (degree + 1) / 2), 2});
      std::size_t n = 0;
      for (int j = 0; j <= degree; ++j)
      {
        const auto interval_pts = lattice::create(
            cell::type::interval, degree - j, latticetype, true);
        for (int i = 0; i <= degree - j; ++i)
        {
          pts(n, 0) = interval_pts(i, 0);
          pts(n, 1) = j % 2 == 0
                          ? static_cast<double>(j / 2) / degree
                          : 1 - static_cast<double>((j - 1) / 2) / degree;
          ++n;
        }
      }
      return pts;
    }
    case cell::type::hexahedron:
    {
      xt::xtensor<double, 2> pts(
          {static_cast<std::size_t>((degree + 3) * (degree + 2) * (degree + 1)
                                    / 6),
           3});
      std::size_t n = 0;
      for (int k = 0; k <= degree; ++k)
      {
        for (int j = 0; j <= degree - k; ++j)
        {
          const auto interval_pts = lattice::create(
              cell::type::interval, degree - j - k, latticetype, true);
          for (int i = 0; i <= degree - j - k; ++i)
          {
            pts(n, 0) = interval_pts(i, 0);
            pts(n, 1)
                = degree - k == 0
                      ? 0.5
                      : (j % 2 == 0 ? static_cast<double>(j / 2) / (degree - k)
                                    : 1
                                          - static_cast<double>((j - 1) / 2)
                                                / (degree - k));
            pts(n, 2) = k % 2 == 0
                            ? static_cast<double>(k / 2) / degree
                            : 1 - static_cast<double>((k - 1) / 2) / degree;
            ++n;
          }
        }
      }
      return pts;
    }
    default:
      throw std::runtime_error("Invalid cell type");
    }
  }
  else if (variant == element::dpc_variant::diagonal_equispaced
           or variant == element::dpc_variant::diagonal_gll)
  {
    lattice::type latticetype;
    lattice::simplex_method latticesm = lattice::simplex_method::isaac;
    if (variant == element::dpc_variant::diagonal_equispaced)
      latticetype = lattice::type::equispaced;
    else if (variant == element::dpc_variant::diagonal_gll)
      latticetype = lattice::type::gll;

    switch (celltype)
    {
    case cell::type::quadrilateral:
    {
      xt::xtensor<double, 2> pts(
          {static_cast<std::size_t>((degree + 2) * (degree + 1) / 2), 2});

      const double gap = static_cast<double>(2 * (degree + 1))
                         / (degree * degree + degree + 1);

      std::size_t n = 0;
      for (int j = 0; j <= degree; ++j)
      {
        const auto interval_pts
            = lattice::create(cell::type::interval, j, latticetype, true);
        const double y = gap * (j % 2 == 0 ? j / 2 : degree - (j - 1) / 2);
        const double coord0 = y < 1 ? y : y - 1;
        const double coord1 = y < 1 ? 0 : 1;
        for (int i = 0; i <= j; ++i)
        {
          const double x = interval_pts(i, 0);
          pts(n, 0) = coord0 * (1 - x) + coord1 * x;
          pts(n, 1) = coord1 * (1 - x) + coord0 * x;
          ++n;
        }
      }
      return pts;
    }
    case cell::type::hexahedron:
    {
      xt::xtensor<double, 2> pts(
          {static_cast<std::size_t>((degree + 3) * (degree + 2) * (degree + 1)
                                    / 6),
           3});

      const double gap
          = static_cast<double>(3 * degree) / (degree * degree + 1);

      std::size_t n = 0;
      for (int k = 0; k <= degree; ++k)
      {
        const double z = gap * (k % 2 == 0 ? k / 2 : degree - (k - 1) / 2);
        const auto triangle_pts = lattice::create(cell::type::triangle, k,
                                                  latticetype, true, latticesm);
        if (z < 1)
          for (std::size_t p = 0; p < triangle_pts.shape(0); ++p)
          {
            const double coord0 = triangle_pts(p, 0);
            const double coord1 = triangle_pts(p, 1);
            pts(n, 0) = coord0 * z;
            pts(n, 1) = coord1 * z;
            pts(n, 2) = (1 - coord0 - coord1) * z;
            ++n;
          }
        else if (z > 2)
          for (std::size_t p = 0; p < triangle_pts.shape(0); ++p)
          {
            const double coord0 = triangle_pts(p, 0);
            const double coord1 = triangle_pts(p, 1);
            pts(n, 0) = 1 - (3 - z) * coord0;
            pts(n, 1) = 1 - (3 - z) * coord1;
            pts(n, 2) = 1 - (3 - z) * (1 - coord0 - coord1);
            ++n;
          }
        else
        {
          for (std::size_t p = 0; p < triangle_pts.shape(0); ++p)
          {
            const double coord0 = triangle_pts(p, 0);
            const double coord1 = triangle_pts(p, 1);
            pts(n, 0) = 1 - (2 - z) * coord0 - coord1;
            pts(n, 1) = coord0 + (z - 1) * coord1;
            pts(n, 2) = z - 1 - (z - 1) * coord0 + (2 - z) * coord1;
            ++n;
          }
        }
      }
      return pts;
    }
    default:
      throw std::runtime_error("Invalid cell type");
    }
  }
  else
    throw std::runtime_error("Unsupported_variant");
}
//----------------------------------------------------------------------------
} // namespace

//----------------------------------------------------------------------------
FiniteElement basix::element::create_lagrange(cell::type celltype, int degree,
                                              element::lagrange_variant variant,
                                              bool discontinuous)
{
  if (celltype == cell::type::point)
  {
    if (degree != 0)
      throw std::runtime_error("Can only create order 0 Lagrange on a point");

    std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
    std::array<std::vector<xt::xtensor<double, 2>>, 4> x;
    x[0].push_back(xt::zeros<double>({1, 0}));
    M[0].push_back({{{1}}});
    std::map<cell::type, xt::xtensor<double, 3>> entity_transformations;
    xt::xtensor<double, 2> wcoeffs = {{1}};

    return FiniteElement(element::family::P, cell::type::point, 0, {}, wcoeffs,
                         entity_transformations, x, M, maps::type::identity,
                         discontinuous, degree, degree);
  }

  if (variant == element::lagrange_variant::unset)
  {
    if (degree < 3)
      variant = element::lagrange_variant::equispaced;
    else
      throw std::runtime_error(
          "Lagrange elements of degree > 2 need to be given a variant.");
  }

  if (variant == element::lagrange_variant::vtk)
    return create_vtk_element(celltype, degree, discontinuous);

  if (variant == element::lagrange_variant::legendre)
    return create_legendre(celltype, degree, discontinuous);

  auto [lattice_type, simplex_method, exterior]
      = variant_to_lattice(celltype, variant);

  if (!exterior)
  {
    // Points used to define this variant are all interior to the cell, so this
    // variant requires that the element is discontinuous
    if (!discontinuous)
    {
      throw std::runtime_error("This variant of Lagrange is only supported for "
                               "discontinuous elements");
    }
    return create_d_lagrange(celltype, degree, variant, lattice_type,
                             simplex_method);
  }

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t ndofs = polyset::dim(celltype, degree);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  // Create points at nodes, ordered by topology (vertices first)
  if (degree == 0)
  {
    if (!discontinuous)
    {
      throw std::runtime_error(
          "Cannot create a continuous order 0 Lagrange basis function");
    }
    auto pt = lattice::create(celltype, 0, lattice_type, true, simplex_method);
    x[tdim].push_back(pt);
    const std::size_t num_dofs = pt.shape(0);
    std::array<std::size_t, 3> s = {num_dofs, 1, num_dofs};
    M[tdim].push_back(xt::xtensor<double, 3>(s));
    xt::view(M[tdim][0], xt::all(), 0, xt::all()) = xt::eye<double>(num_dofs);
  }
  else
  {
    for (std::size_t dim = 0; dim <= tdim; ++dim)
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
        }
        else if (dim == tdim)
        {
          x[dim][e] = lattice::create(celltype, degree, lattice_type, false,
                                      simplex_method);
          const std::size_t num_dofs = x[dim][e].shape(0);
          std::array<std::size_t, 3> s = {num_dofs, 1, num_dofs};
          M[dim][e] = xt::xtensor<double, 3>(s);
          xt::view(M[dim][e], xt::all(), 0, xt::all())
              = xt::eye<double>(num_dofs);
        }
        else
        {
          cell::type ct = cell::sub_entity_type(celltype, dim, e);
          const auto lattice = lattice::create(ct, degree, lattice_type, false,
                                               simplex_method);
          const std::size_t num_dofs = lattice.shape(0);
          std::array<std::size_t, 3> s = {num_dofs, 1, num_dofs};
          M[dim][e] = xt::xtensor<double, 3>(s);
          xt::view(M[dim][e], xt::all(), 0, xt::all())
              = xt::eye<double>(num_dofs);

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

  std::map<cell::type, xt::xtensor<double, 3>> entity_transformations;

  // Entity transformations for edges
  if (tdim > 1)
  {
    const std::vector<int> edge_ref
        = doftransforms::interval_reflection(degree - 1);
    const std::array<std::size_t, 3> shape
        = {1, edge_ref.size(), edge_ref.size()};
    xt::xtensor<double, 3> et = xt::zeros<double>(shape);
    for (std::size_t i = 0; i < edge_ref.size(); ++i)
      et(0, i, edge_ref[i]) = 1;
    entity_transformations[cell::type::interval] = et;
  }

  // Entity transformations for triangular faces
  if (celltype == cell::type::tetrahedron or celltype == cell::type::prism
      or celltype == cell::type::pyramid)
  {
    const std::vector<int> face_rot
        = doftransforms::triangle_rotation(degree - 2);
    const std::vector<int> face_ref
        = doftransforms::triangle_reflection(degree - 2);
    const std::array<std::size_t, 3> shape
        = {2, face_rot.size(), face_rot.size()};
    xt::xtensor<double, 3> ft = xt::zeros<double>(shape);
    for (std::size_t i = 0; i < face_rot.size(); ++i)
    {
      ft(0, i, face_rot[i]) = 1;
      ft(1, i, face_ref[i]) = 1;
    }
    entity_transformations[cell::type::triangle] = ft;
  }

  // Entity transformations for quadrilateral faces
  if (celltype == cell::type::hexahedron or celltype == cell::type::prism
      or celltype == cell::type::pyramid)
  {
    const std::vector<int> face_rot
        = doftransforms::quadrilateral_rotation(degree - 1);
    const std::vector<int> face_ref
        = doftransforms::quadrilateral_reflection(degree - 1);
    const std::array<std::size_t, 3> shape
        = {2, face_rot.size(), face_rot.size()};
    xt::xtensor<double, 3> ft = xt::zeros<double>(shape);
    for (std::size_t i = 0; i < face_rot.size(); ++i)
    {
      ft(0, i, face_rot[i]) = 1;
      ft(1, i, face_ref[i]) = 1;
    }
    entity_transformations[cell::type::quadrilateral] = ft;
  }

  if (discontinuous)
  {
    std::tie(x, M, entity_transformations)
        = element::make_discontinuous(x, M, entity_transformations, tdim, 1);
  }

  auto tensor_factors
      = create_tensor_product_factors(celltype, degree, variant);

  return FiniteElement(element::family::P, celltype, degree, {},
                       xt::eye<double>(ndofs), entity_transformations, x, M,
                       maps::type::identity, discontinuous, degree, degree,
                       tensor_factors, variant);
}
//-----------------------------------------------------------------------------
FiniteElement basix::element::create_dpc(cell::type celltype, int degree,
                                         element::dpc_variant variant,
                                         bool discontinuous)
{
  // Only tabulate for scalar. Vector spaces can easily be built from
  // the scalar space.
  if (!discontinuous)
  {
    throw std::runtime_error("Cannot create a continuous DPC element.");
  }

  if (variant == element::dpc_variant::unset)
  {
    if (degree == 0)
      variant = element::dpc_variant::simplex_equispaced;
    else
      throw std::runtime_error(
          "DPC elements of degree > 0 need to be given a variant.");
  }

  cell::type simplex_type;
  switch (celltype)
  {
  case cell::type::quadrilateral:
    simplex_type = cell::type::triangle;
    break;
  case cell::type::hexahedron:
    simplex_type = cell::type::tetrahedron;
    break;
  default:
    throw std::runtime_error("Invalid cell type");
  }

  if (variant == element::dpc_variant::legendre)
    return create_legendre_dpc(celltype, degree, discontinuous);

  const std::size_t ndofs = polyset::dim(simplex_type, degree);
  const std::size_t psize = polyset::dim(celltype, degree);

  auto [pts, _wts] = quadrature::make_quadrature(quadrature::type::Default,
                                                 celltype, 2 * degree);
  auto wts = xt::adapt(_wts);

  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize});

  if (celltype == cell::type::quadrilateral)
  {
    int row_n = 0;
    for (int i = 0; i <= degree; ++i)
    {
      for (int j = 0; j <= degree - i; ++j)
      {
        wcoeffs(row_n++, i * (degree + 1) + j) = 1;
      }
    }
  }
  else
  {
    int row_n = 0;
    for (int i = 0; i <= degree; ++i)
    {
      for (int j = 0; j <= degree - i; ++j)
      {
        for (int k = 0; k <= degree - i - j; ++k)
        {
          wcoeffs(row_n++,
                  i * (degree + 1) * (degree + 1) + j * (degree + 1) + k)
              = 1;
        }
      }
    }
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const std::size_t tdim = topology.size() - 1;

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  M[tdim].push_back(xt::xtensor<double, 3>({ndofs, 1, ndofs}));
  xt::view(M[tdim][0], xt::all(), 0, xt::all()) = xt::eye<double>(ndofs);

  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;
  const auto pt = make_dpc_points(celltype, degree, variant);
  x[tdim].push_back(pt);

  std::map<cell::type, xt::xtensor<double, 3>> entity_transformations;
  entity_transformations[cell::type::interval]
      = xt::xtensor<double, 3>({1, 0, 0});
  if (tdim == 3)
  {
    entity_transformations[cell::type::quadrilateral]
        = xt::xtensor<double, 3>({2, 0, 0});
  }

  return FiniteElement(element::family::DPC, celltype, degree, {}, wcoeffs,
                       entity_transformations, x, M, maps::type::identity,
                       discontinuous, degree, degree, {},
                       element::lagrange_variant::unset, variant);
}
//-----------------------------------------------------------------------------
