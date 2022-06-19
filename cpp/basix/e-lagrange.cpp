// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-lagrange.h"
#include "lattice.h"
#include "maps.h"
#include "mdspan.hpp"
#include "moments.h"
#include "polynomials.h"
#include "polyset.h"
#include "quadrature.h"
#include <xtensor/xbuilder.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xview.hpp>

namespace stdex = std::experimental;

using namespace basix;

namespace
{
using mdspan2_t = stdex::mdspan<double, stdex::dextents<2>>;
using mdspan4_t = stdex::mdspan<double, stdex::dextents<4>>;

//----------------------------------------------------------------------------
template <typename U>
xt::xtensor<typename U::value_type, 2> mdspan_to_xtensor2(const U& x)
{
  auto e = x.extents();
  xt::xtensor<typename U::value_type, 2> y({e.extent(0), e.extent(1)});
  for (std::size_t k0 = 0; k0 < e.extent(0); ++k0)
    for (std::size_t k1 = 0; k1 < e.extent(1); ++k1)
      y(k0, k1) = x(k0, k1);
  return y;
}
//----------------------------------------------------------------------------
template <typename U>
xt::xtensor<double, 4> mdspan_to_xtensor4(const U& x)
{
  auto e = x.extents();
  std::array<std::size_t, 4> shape
      = {e.extent(0), e.extent(1), e.extent(2), e.extent(3)};
  xt::xtensor<double, 4> y(shape);
  for (std::size_t k0 = 0; k0 < e.extent(0); ++k0)
    for (std::size_t k1 = 0; k1 < e.extent(1); ++k1)
      for (std::size_t k2 = 0; k2 < e.extent(2); ++k2)
        for (std::size_t k3 = 0; k3 < e.extent(3); ++k3)
          y(k0, k1, k2, k3) = x(k0, k1, k2, k3);

  return y;
}
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
  if (celltype == cell::type::prism or celltype == cell::type::pyramid)
  {
    throw std::runtime_error(
        "This variant is not yet supported on prisms and pyramids.");
  }

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t ndofs = polyset::dim(celltype, degree);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<std::vector<double>>, 4> x;
  std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
  std::array<std::vector<std::vector<double>>, 4> M;
  std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
  for (std::size_t i = 0; i < tdim; ++i)
  {
    std::size_t num_ent = cell::num_sub_entities(celltype, i);
    x[i] = std::vector<std::vector<double>>(num_ent, std::vector<double>(0));
    xshape[i] = std::vector<std::array<std::size_t, 2>>(num_ent, {0, tdim});
    M[i] = std::vector<std::vector<double>>(num_ent, std::vector<double>(0));
    Mshape[i] = std::vector<std::array<std::size_t, 4>>(num_ent, {0, 1, 0, 1});
  }

  const int lattice_degree
      = celltype == cell::type::triangle
            ? degree + 3
            : (celltype == cell::type::tetrahedron ? degree + 4 : degree + 2);

  // Create points in interior
  const xt::xtensor<double, 2> pt = lattice::create(
      celltype, lattice_degree, lattice_type, false, simplex_method);
  x[tdim].emplace_back(pt.data(), pt.data() + pt.size());
  xshape[tdim].push_back({pt.shape(0), pt.shape(1)});

  const std::size_t num_dofs = pt.shape(0);
  M[tdim].emplace_back(num_dofs * num_dofs);
  Mshape[tdim].push_back({num_dofs, 1, num_dofs, 1});
  mdspan4_t Mview(M[tdim].back().data(), num_dofs, 1, num_dofs, 1);
  for (std::size_t i = 0; i < num_dofs; ++i)
    Mview(i, 0, i, 0) = 1.0;

  // Convert data to xtensor
  std::array<std::vector<xt::xtensor<double, 2>>, 4> _x;
  std::array<std::vector<xt::xtensor<double, 4>>, 4> _M;
  for (std::size_t i = 0; i < x.size(); ++i)
  {
    for (std::size_t j = 0; j < x[i].size(); ++j)
      _x[i].push_back(
          mdspan_to_xtensor2(mdspan2_t(x[i][j].data(), xshape[i][j])));
  }
  for (std::size_t i = 0; i < M.size(); ++i)
  {
    for (std::size_t j = 0; j < M[i].size(); ++j)
      _M[i].push_back(
          mdspan_to_xtensor4(mdspan4_t(M[i][j].data(), Mshape[i][j])));
  }
  return FiniteElement(element::family::P, celltype, degree, {},
                       xt::eye<double>(ndofs), _x, _M, 0, maps::type::identity,
                       true, degree, degree, variant);
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
std::vector<double> vtk_triangle_points(int degree)
{
  const double d = static_cast<double>(1) / static_cast<double>(degree + 3);
  if (degree == 0)
    return {d, d};

  const std::size_t npoints = polyset::dim(cell::type::triangle, degree);
  std::vector<double> outdata(npoints * 2);
  stdex::mdspan<double, stdex::extents<stdex::dynamic_extent, 2>> out(
      outdata.data(), npoints, 2);

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
    std::vector<double> pts_data = vtk_triangle_points(degree - 3);
    stdex::mdspan<double, stdex::extents<stdex::dynamic_extent, 2>> pts(
        pts_data.data(), pts_data.size() / 2, 2);
    for (std::size_t i = 0; i < pts.extent(0); ++i)
    {
      for (std::size_t j = 0; j < pts.extent(1); ++j)
        out(n, j) = d + (1 - 3 * d) * pts(i, j);
      ++n;
    }
  }

  return outdata;
}
//-----------------------------------------------------------------------------
std::vector<double> vtk_tetrahedron_points(int degree)
{
  const double d = static_cast<double>(1) / static_cast<double>(degree + 4);
  if (degree == 0)
    return {d, d, d};

  const std::size_t npoints = polyset::dim(cell::type::tetrahedron, degree);
  std::vector<double> outdata(npoints * 3);
  stdex::mdspan<double, stdex::extents<stdex::dynamic_extent, 3>> out(
      outdata.data(), npoints, 3);

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
    std::vector<double> pts_data = vtk_triangle_points(degree - 3);
    stdex::mdspan<double, stdex::extents<stdex::dynamic_extent, 2>> pts(
        pts_data.data(), pts_data.size() / 2, 2);

    for (std::size_t i = 0; i < pts.extent(0); ++i)
    {
      out(n, 0) = d + pts(i, 0) * (1 - 4 * d);
      out(n, 1) = d;
      out(n, 2) = d + pts(i, 1) * (1 - 4 * d);
      ++n;
    }
    for (std::size_t i = 0; i < pts.extent(0); ++i)
    {
      out(n, 0) = 1 - 3 * d - (pts(i, 0) + pts(i, 1)) * (1 - 4 * d);
      out(n, 1) = d + pts(i, 0) * (1 - 4 * d);
      out(n, 2) = d + pts(i, 1) * (1 - 4 * d);
      ++n;
    }
    for (std::size_t i = 0; i < pts.extent(0); ++i)
    {
      out(n, 0) = d;
      out(n, 1) = d + pts(i, 0) * (1 - 4 * d);
      out(n, 2) = d + pts(i, 1) * (1 - 4 * d);
      ++n;
    }
    for (std::size_t i = 0; i < pts.extent(0); ++i)
    {
      out(n, 0) = d + pts(i, 0) * (1 - 4 * d);
      out(n, 1) = d + pts(i, 1) * (1 - 4 * d);
      out(n, 2) = d;
      ++n;
    }
  }

  if (degree >= 4)
  {
    std::vector<double> pts_data = vtk_tetrahedron_points(degree - 4);
    stdex::mdspan<double, stdex::extents<stdex::dynamic_extent, 3>> pts(
        pts_data.data(), pts_data.size() / 3, 3);

    auto out_view = stdex::submdspan(out, std::pair<int, int>{n, npoints},
                                     stdex::full_extent);
    for (std::size_t i = 0; i < out_view.extent(0); ++i)
      for (std::size_t j = 0; j < out_view.extent(1); ++j)
        out_view(i, j) = pts(i, j);

    // xt::view(out, xt::range(n, npoints), xt::all())
    //     = vtk_tetrahedron_points(degree - 4);

    // xt::xtensor<double, 2> pts = vtk_tetrahedron_points(degree - 4);
    // std::vector<double> pts_data = vtk_tetrahedron_points(degree - 4);
    // stdex::mdspan<double, stdex::extents<stdex::dynamic_extent, 3>> pts(
    //     pts_data.data(), pts_data.size() / 3, 3);

    for (std::size_t i = 0; i < pts.extent(0); ++i)
    {
      for (std::size_t j = 0; j < pts.extent(1); ++j)
        out(n, j) = d + (1 - 4 * d) * pts(i, j);
      ++n;
    }
  }

  return outdata;
}
//-----------------------------------------------------------------------------
FiniteElement create_vtk_element(cell::type celltype, int degree,
                                 bool discontinuous)
{
  if (celltype == cell::type::point)
    throw std::runtime_error("Invalid celltype");

  if (degree == 0)
    throw std::runtime_error("Cannot create an order 0 VTK element.");

  // DOF transformation don't yet work on this element, so throw runtime
  // error if trying to make continuous version
  if (!discontinuous)
    throw std::runtime_error("Continuous VTK element not yet supported.");

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t ndofs = polyset::dim(celltype, degree);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<xt::xtensor<double, 4>>, 4> M;
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
      M[0][i] = {{{{1.}}}};

    // Points on interval
    x[1][0] = xt::xtensor<double, 2>(
        {static_cast<std::size_t>(degree - 1), static_cast<std::size_t>(1)});
    for (int i = 1; i < degree; ++i)
      x[1][0](i - 1, 0) = static_cast<double>(i) / static_cast<double>(degree);

    M[1][0] = xt::xtensor<double, 4>({static_cast<std::size_t>(degree - 1), 1,
                                      static_cast<std::size_t>(degree - 1), 1});
    xt::view(M[1][0], xt::all(), 0, xt::all(), 0) = xt::eye<double>(degree - 1);

    break;
  }
  case cell::type::triangle:
  {
    // Points at vertices
    x[0][0] = {{0., 0.}};
    x[0][1] = {{1., 0.}};
    x[0][2] = {{0., 1.}};
    for (int i = 0; i < 3; ++i)
      M[0][i] = {{{{1.}}}};

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
      M[1][i]
          = xt::xtensor<double, 4>({static_cast<std::size_t>(degree - 1), 1,
                                    static_cast<std::size_t>(degree - 1), 1});
      xt::view(M[1][i], xt::all(), 0, xt::all(), 0)
          = xt::eye<double>(degree - 1);
    }

    // Points in triangle
    if (degree >= 3)
    {
      std::vector<double> pts_data = vtk_triangle_points(degree - 3);
      auto pts = xt::adapt(pts_data,
                           std::vector<std::size_t>{pts_data.size() / 2, 2});
      x[2][0] = pts;
      // x[2][0] = vtk_triangle_points(degree - 3);
      M[2][0]
          = xt::xtensor<double, 4>({x[2][0].shape(0), 1, x[2][0].shape(0), 1});
      xt::view(M[2][0], xt::all(), 0, xt::all(), 0)
          = xt::eye<double>(x[2][0].shape(0));
    }
    else
    {
      x[2][0] = xt::xtensor<double, 2>({0, 2});
      M[2][0] = xt::xtensor<double, 4>({0, 1, 0, 1});
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
      M[0][i] = {{{{1.}}}};

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
      M[1][i]
          = xt::xtensor<double, 4>({static_cast<std::size_t>(degree - 1), 1,
                                    static_cast<std::size_t>(degree - 1), 1});
      xt::view(M[1][i], xt::all(), 0, xt::all(), 0)
          = xt::eye<double>(degree - 1);
    }

    // Points on faces
    if (degree >= 3)
    {
      std::vector<double> pts_data = vtk_triangle_points(degree - 3);

      xt::xtensor<double, 2> pts = xt::adapt(
          pts_data, std::vector<std::size_t>{pts_data.size() / 2, 2});

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
        M[2][i] = xt::xtensor<double, 4>(
            {x[2][0].shape(0), 1, x[2][0].shape(0), 1});
        xt::view(M[2][i], xt::all(), 0, xt::all(), 0)
            = xt::eye<double>(x[2][0].shape(0));
      }
    }
    else
    {
      for (int i = 0; i < 4; ++i)
      {
        x[2][i] = xt::xtensor<double, 2>({0, 3});
        M[2][i] = xt::xtensor<double, 4>({0, 1, 0, 1});
      }
    }

    if (degree >= 4)
    {
      std::vector<double> pts_data = vtk_tetrahedron_points(degree - 4);
      xt::xtensor<double, 2> pts = xt::adapt(
          pts_data, std::vector<std::size_t>{pts_data.size() / 3, 3});

      x[3][0] = pts;
      M[3][0]
          = xt::xtensor<double, 4>({x[3][0].shape(0), 1, x[3][0].shape(0), 1});
      xt::view(M[3][0], xt::all(), 0, xt::all(), 0)
          = xt::eye<double>(x[3][0].shape(0));
    }
    else
    {
      x[3][0] = xt::xtensor<double, 2>({0, 3});
      M[3][0] = xt::xtensor<double, 4>({0, 1, 0, 1});
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
      M[0][i] = {{{{1.}}}};

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
      M[1][i]
          = xt::xtensor<double, 4>({static_cast<std::size_t>(degree - 1), 1,
                                    static_cast<std::size_t>(degree - 1), 1});
      xt::view(M[1][i], xt::all(), 0, xt::all(), 0)
          = xt::eye<double>(degree - 1);
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

    M[2][0]
        = xt::xtensor<double, 4>({x[2][0].shape(0), 1, x[2][0].shape(0), 1});
    xt::view(M[2][0], xt::all(), 0, xt::all(), 0)
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
      M[0][i] = {{{{1.}}}};

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
      M[1][i]
          = xt::xtensor<double, 4>({static_cast<std::size_t>(degree - 1), 1,
                                    static_cast<std::size_t>(degree - 1), 1});
      xt::view(M[1][i], xt::all(), 0, xt::all(), 0)
          = xt::eye<double>(degree - 1);
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
      M[2][i]
          = xt::xtensor<double, 4>({x[2][0].shape(0), 1, x[2][0].shape(0), 1});
      xt::view(M[2][i], xt::all(), 0, xt::all(), 0)
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

    M[3][0]
        = xt::xtensor<double, 4>({x[3][0].shape(0), 1, x[3][0].shape(0), 1});
    xt::view(M[3][0], xt::all(), 0, xt::all(), 0)
        = xt::eye<double>(x[3][0].shape(0));

    break;
  }
  default:
    throw std::runtime_error("Unsupported cell type.");
  }

  if (discontinuous)
    std::tie(x, M) = element::make_discontinuous(x, M, tdim, 1);

  return FiniteElement(element::family::P, celltype, degree, {},
                       xt::eye<double>(ndofs), x, M, 0, maps::type::identity,
                       discontinuous, degree, degree,
                       element::lagrange_variant::vtk);
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

  std::array<std::vector<std::vector<double>>, 4> xbuffer;
  std::array<std::vector<mdspan2_t>, 4> x;
  std::array<std::vector<std::vector<double>>, 4> Mbuffer;
  std::array<std::vector<mdspan4_t>, 4> M;
  for (std::size_t i = 0; i < tdim; ++i)
  {
    std::size_t num_entities = cell::num_sub_entities(celltype, i);
    x[i] = std::vector<mdspan2_t>(num_entities, mdspan2_t(nullptr, 0, tdim));
    Mbuffer[i] = std::vector<std::vector<double>>(num_entities,
                                                  std::vector<double>(0));
    for (std::size_t j = 0; j < num_entities; ++j)
      M[i].push_back(mdspan4_t(Mbuffer[i][j].data(), 0, 1, 0, 1));
  }

  for (std::size_t dim = 0; dim <= tdim; ++dim)
  {
    xbuffer[dim].resize(topology[dim].size());
    x[dim].resize(topology[dim].size());
    Mbuffer[dim].resize(topology[dim].size());
    M[dim].resize(topology[dim].size());
    if (dim < tdim)
    {
      for (std::size_t e = 0; e < topology[dim].size(); ++e)
      {
        xbuffer[dim][e] = {};
        x[dim][e] = mdspan2_t(xbuffer[dim][e].data(), 0, tdim);
        Mbuffer[dim][e] = {};
        M[dim][e] = mdspan4_t(Mbuffer[dim][e].data(), 0, 1, 0, 1);
      }
    }
  }

  auto [pts, _wts] = quadrature::make_quadrature(quadrature::type::Default,
                                                 celltype, degree * 2);
  auto wts = xt::adapt(_wts);

  // Evaluate moment space at quadrature points
  const xt::xtensor<double, 2> phi = polynomials::tabulate(
      polynomials::type::legendre, celltype, degree, pts);

  for (std::size_t dim = 0; dim <= tdim; ++dim)
  {
    xbuffer[dim].resize(topology[dim].size());
    x[dim].resize(topology[dim].size());
    Mbuffer[dim].resize(topology[dim].size());
    M[dim].resize(topology[dim].size());
    if (dim < tdim)
    {
      for (std::size_t e = 0; e < topology[dim].size(); ++e)
      {
        x[dim][e] = mdspan2_t(nullptr, 0, tdim);
        M[dim][e] = mdspan4_t(nullptr, 0, 1, 0, 1);
      }
    }
  }

  xbuffer[tdim][0] = std::vector<double>(pts.data(), pts.data() + pts.size());
  x[tdim][0] = mdspan2_t(xbuffer[tdim][0].data(), pts.shape(0), pts.shape(1));
  // x[tdim][0] = pts;
  Mbuffer[tdim][0] = std::vector<double>(ndofs * pts.shape(0));
  M[tdim][0] = mdspan4_t(Mbuffer[tdim][0].data(), ndofs, 1, pts.shape(0), 1);
  // M[tdim][0] = xt::xtensor<double, 4>({ndofs, 1, pts.shape(0), 1});
  // for (std::size_t i = 0; i < ndofs; ++i)
  //   xt::view(M[tdim][0], i, 0, xt::all(), 0) = xt::row(phi, i) * wts;

  for (std::size_t i = 0; i < ndofs; ++i)
    for (std::size_t j = 0; j < pts.shape(0); ++j)
      M[tdim][0](i, 0, j, 0) = phi(i, j) * wts(j);

  // Convert data to xtensor
  std::array<std::vector<xt::xtensor<double, 2>>, 4> _x;
  std::array<std::vector<xt::xtensor<double, 4>>, 4> _M;
  for (std::size_t i = 0; i < x.size(); ++i)
  {
    _x[i].resize(x[i].size());
    for (std::size_t j = 0; j < x[i].size(); ++j)
      _x[i][j] = mdspan_to_xtensor2(x[i][j]);
  }
  for (std::size_t i = 0; i < M.size(); ++i)
  {
    _M[i].resize(M[i].size());
    for (std::size_t j = 0; j < M[i].size(); ++j)
      _M[i][j] = mdspan_to_xtensor4(M[i][j]);
  }
  return FiniteElement(element::family::P, celltype, degree, {},
                       xt::eye<double>(ndofs), _x, _M, 0, maps::type::identity,
                       discontinuous, degree, degree,
                       element::lagrange_variant::legendre);
}
//-----------------------------------------------------------------------------
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

    std::array<std::vector<xt::xtensor<double, 4>>, 4> M;
    std::array<std::vector<xt::xtensor<double, 2>>, 4> x;
    x[0].push_back(xt::zeros<double>({1, 0}));
    M[0].push_back({{{{1.}}}});
    xt::xtensor<double, 2> wcoeffs = {{1}};
    return FiniteElement(element::family::P, cell::type::point, 0, {}, wcoeffs,
                         x, M, 0, maps::type::identity, discontinuous, degree,
                         degree);
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
    // Points used to define this variant are all interior to the cell,
    // so this variant requires that the element is discontinuous
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

  std::array<std::vector<std::vector<double>>, 4> Mbuffer;
  std::array<std::vector<mdspan4_t>, 4> M;

  std::array<std::vector<std::vector<double>>, 4> xbuffer;
  std::array<std::vector<mdspan2_t>, 4> x;

  if (degree == 0)
  {
    if (!discontinuous)
    {
      throw std::runtime_error(
          "Cannot create a continuous order 0 Lagrange basis function");
    }

    for (std::size_t i = 0; i < tdim; ++i)
    {
      std::size_t num_entities = cell::num_sub_entities(celltype, i);
      x[i] = std::vector<mdspan2_t>(num_entities, mdspan2_t(nullptr, 0, tdim));
      Mbuffer[i] = std::vector<std::vector<double>>(num_entities,
                                                    std::vector<double>(0));
      for (std::size_t j = 0; j < num_entities; ++j)
        M[i].push_back(mdspan4_t(Mbuffer[i][j].data(), 0, 1, 0, 1));
    }

    const xt::xtensor<double, 2> pt
        = lattice::create(celltype, 0, lattice_type, true, simplex_method);
    xbuffer[tdim].push_back(std::vector<double>(pt.begin(), pt.end()));
    x[tdim].push_back(
        mdspan2_t(xbuffer[tdim].back().data(), pt.shape(0), pt.shape(1)));

    const std::size_t num_dofs = pt.shape(0);
    std::array<std::size_t, 4> s = {num_dofs, 1, num_dofs, 1};

    Mbuffer[tdim].push_back(std::vector<double>(num_dofs * num_dofs));
    M[tdim].push_back(
        mdspan4_t(Mbuffer[tdim].back().data(), num_dofs, 1, num_dofs, 1));
    auto Mview = stdex::submdspan(M[tdim].back(), stdex::full_extent, 0,
                                  stdex::full_extent, 0);
    for (std::size_t i = 0; i < num_dofs; ++i)
      Mview(i, i) = 1;
  }
  else
  {
    // Create points at nodes, ordered by topology (vertices first)
    for (std::size_t dim = 0; dim <= tdim; ++dim)
    {
      Mbuffer[dim].resize(topology[dim].size());
      M[dim].resize(topology[dim].size());

      xbuffer[dim].resize(topology[dim].size());
      x[dim].resize(topology[dim].size());

      // Loop over entities of dimension 'dim'
      for (std::size_t e = 0; e < topology[dim].size(); ++e)
      {
        const xt::xtensor<double, 2> entity_x
            = cell::sub_entity_geometry(celltype, dim, e);
        if (dim == 0)
        {
          xbuffer[dim][e] = std::vector<double>(
              entity_x.data(), entity_x.data() + entity_x.size());
          x[dim][e] = mdspan2_t(xbuffer[dim][e].data(), entity_x.shape(0),
                                entity_x.shape(1));

          const std::size_t num_dofs = entity_x.shape(0);
          Mbuffer[dim][e] = std::vector<double>(num_dofs * num_dofs);
          M[dim][e]
              = mdspan4_t(Mbuffer[dim][e].data(), num_dofs, 1, num_dofs, 1);
          auto Mview = stdex::submdspan(M[dim][e], stdex::full_extent, 0,
                                        stdex::full_extent, 0);
          for (std::size_t i = 0; i < num_dofs; ++i)
            Mview(i, i) = 1;
        }
        else if (dim == tdim)
        {
          const xt::xtensor<double, 2> pt = lattice::create(
              celltype, degree, lattice_type, false, simplex_method);
          xbuffer[dim][e] = std::vector<double>(pt.begin(), pt.end());
          x[dim][e]
              = mdspan2_t(xbuffer[dim][e].data(), pt.shape(0), pt.shape(1));

          const std::size_t num_dofs = pt.shape(0);
          Mbuffer[dim][e] = std::vector<double>(num_dofs * num_dofs);
          M[dim][e]
              = mdspan4_t(Mbuffer[dim][e].data(), num_dofs, 1, num_dofs, 1);
          auto Mview = stdex::submdspan(M[dim][e], stdex::full_extent, 0,
                                        stdex::full_extent, 0);
          for (std::size_t i = 0; i < num_dofs; ++i)
            Mview(i, i) = 1;
        }
        else
        {
          cell::type ct = cell::sub_entity_type(celltype, dim, e);
          const xt::xtensor<double, 2> lattice = lattice::create(
              ct, degree, lattice_type, false, simplex_method);
          const std::size_t num_dofs = lattice.shape(0);

          Mbuffer[dim][e] = std::vector<double>(num_dofs * num_dofs);
          M[dim][e]
              = mdspan4_t(Mbuffer[dim][e].data(), num_dofs, 1, num_dofs, 1);
          auto Mview = stdex::submdspan(M[dim][e], stdex::full_extent, 0,
                                        stdex::full_extent, 0);
          for (std::size_t i = 0; i < num_dofs; ++i)
            Mview(i, i) = 1;

          xtl::span<const double> x0(entity_x.data(), entity_x.shape(1));
          for (std::size_t i = 0; i < lattice.shape(0); ++i)
            xbuffer[dim][e].insert(xbuffer[dim][e].end(), x0.begin(), x0.end());
          x[dim][e] = mdspan2_t(xbuffer[dim][e].data(), lattice.shape(0),
                                entity_x.shape(1));

          mdspan2_t& _x = x[dim][e];
          for (std::size_t j = 0; j < lattice.shape(0); ++j)
            for (std::size_t k = 0; k < lattice.shape(1); ++k)
              for (std::size_t q = 0; q < tdim; ++q)
                _x(j, q) += (entity_x(k + 1, q) - x0[q]) * lattice(j, k);
        }
      }
    }
  }

  // Convert data to xtensor
  std::array<std::vector<xt::xtensor<double, 2>>, 4> _x;
  std::array<std::vector<xt::xtensor<double, 4>>, 4> _M;
  for (std::size_t i = 0; i < x.size(); ++i)
  {
    _x[i].resize(x[i].size());
    for (std::size_t j = 0; j < x[i].size(); ++j)
      _x[i][j] = mdspan_to_xtensor2(x[i][j]);
  }
  for (std::size_t i = 0; i < M.size(); ++i)
  {
    _M[i].resize(M[i].size());
    for (std::size_t j = 0; j < M[i].size(); ++j)
      _M[i][j] = mdspan_to_xtensor4(M[i][j]);
  }

  if (discontinuous)
    std::tie(_x, _M) = element::make_discontinuous(_x, _M, tdim, 1);

  auto tensor_factors
      = create_tensor_product_factors(celltype, degree, variant);
  return FiniteElement(element::family::P, celltype, degree, {},
                       xt::eye<double>(ndofs), _x, _M, 0, maps::type::identity,
                       discontinuous, degree, degree, variant, tensor_factors);
}
//-----------------------------------------------------------------------------
