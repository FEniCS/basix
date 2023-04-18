// Copyright (c) 2020-2022 Chris Richardson, Matthew Scroggs and Garth N. Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-lagrange.h"
#include "lattice.h"
#include "maps.h"
#include "math.h"
#include "mdspan.hpp"
#include "moments.h"
#include "polynomials.h"
#include "polyset.h"
#include "quadrature.h"
#include "sobolev-spaces.h"
#include <concepts>

using namespace basix;
namespace stdex = std::experimental;

namespace
{
//----------------------------------------------------------------------------
template <std::floating_point T>
impl::mdarray_t<T, 2> vtk_triangle_points(std::size_t degree)
{
  const T d = 1 / static_cast<T>(degree + 3);
  if (degree == 0)
    return basix::impl::mdarray_t<T, 2>({d, d}, 1, 2);

  const std::size_t npoints = polyset::dim(cell::type::triangle, degree);
  impl::mdarray_t<T, 2> out(npoints, 2);

  out(0, 0) = d;
  out(0, 1) = d;
  out(1, 0) = 1 - 2 * d;
  out(1, 1) = d;
  out(2, 0) = d;
  out(2, 1) = 1 - 2 * d;
  int n = 3;
  if (degree >= 2)
  {
    for (std::size_t i = 1; i < degree; ++i)
    {
      out(n, 0) = d + ((1 - 3 * d) * i) / degree;
      out(n, 1) = d;
      ++n;
    }
    for (std::size_t i = 1; i < degree; ++i)
    {
      out(n, 0) = d + ((1 - 3 * d) * (degree - i)) / degree;
      out(n, 1) = d + ((1 - 3 * d) * i) / degree;
      ++n;
    }
    for (std::size_t i = 1; i < degree; ++i)
    {
      out(n, 0) = d;
      out(n, 1) = d + ((1 - 3 * d) * (degree - i)) / degree;
      ++n;
    }
  }
  if (degree >= 3)
  {
    const auto pts = vtk_triangle_points<T>(degree - 3);
    for (std::size_t i = 0; i < pts.extent(0); ++i)
    {
      for (std::size_t j = 0; j < pts.extent(1); ++j)
        out(n, j) = d + (1 - 3 * d) * pts(i, j);
      ++n;
    }
  }

  return out;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
stdex::mdarray<T, stdex::extents<std::size_t, stdex::dynamic_extent, 3>>
vtk_tetrahedron_points(std::size_t degree)
{
  const T d = 1 / static_cast<T>(degree + 4);
  if (degree == 0)
  {
    return stdex::mdarray<
        T, stdex::extents<std::size_t, stdex::dynamic_extent, 3>>({d, d, d}, 1,
                                                                  2);
  }

  const std::size_t npoints = polyset::dim(cell::type::tetrahedron, degree);
  stdex::mdarray<T, stdex::extents<std::size_t, stdex::dynamic_extent, 3>> out(
      npoints, 3);

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
    for (std::size_t i = 1; i < degree; ++i)
    {
      out(n, 0) = d + ((1 - 4 * d) * i) / (degree);
      out(n, 1) = d;
      out(n, 2) = d;
      ++n;
    }
    for (std::size_t i = 1; i < degree; ++i)
    {
      out(n, 0) = d + ((1 - 4 * d) * (degree - i)) / degree;
      out(n, 1) = d + ((1 - 4 * d) * i) / degree;
      out(n, 2) = d;
      ++n;
    }
    for (std::size_t i = 1; i < degree; ++i)
    {
      out(n, 0) = d;
      out(n, 1) = d + ((1 - 4 * d) * (degree - i)) / degree;
      out(n, 2) = d;
      ++n;
    }
    for (std::size_t i = 1; i < degree; ++i)
    {
      out(n, 0) = d;
      out(n, 1) = d;
      out(n, 2) = d + ((1 - 4 * d) * i) / degree;
      ++n;
    }
    for (std::size_t i = 1; i < degree; ++i)
    {
      out(n, 0) = d + ((1 - 4 * d) * (degree - i)) / degree;
      out(n, 1) = d;
      out(n, 2) = d + ((1 - 4 * d) * i) / degree;
      ++n;
    }
    for (std::size_t i = 1; i < degree; ++i)
    {
      out(n, 0) = d;
      out(n, 1) = d + ((1 - 4 * d) * (degree - i)) / degree;
      out(n, 2) = d + ((1 - 4 * d) * i) / degree;
      ++n;
    }
  }

  if (degree >= 3)
  {
    const auto pts = vtk_triangle_points<T>(degree - 3);
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
    const auto pts = vtk_tetrahedron_points<T>(degree - 4);
    auto _out = impl::mdspan_t<T, 2>(out.data(), out.extents());
    auto out_view = stdex::submdspan(_out, std::pair<int, int>{n, npoints},
                                     stdex::full_extent);
    for (std::size_t i = 0; i < out_view.extent(0); ++i)
      for (std::size_t j = 0; j < out_view.extent(1); ++j)
        out_view(i, j) = pts(i, j);

    for (std::size_t i = 0; i < pts.extent(0); ++i)
    {
      for (std::size_t j = 0; j < pts.extent(1); ++j)
        out(n, j) = d + (1 - 4 * d) * pts(i, j);
      ++n;
    }
  }

  return out;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::array<std::vector<impl::mdarray_t<T, 2>>, 4>,
          std::array<std::vector<impl::mdarray_t<T, 4>>, 4>>
vtk_data_interval(std::size_t degree)
{
  // constexpr std::size_t tdim = 1;
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(cell::type::interval);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;

  // Points at vertices
  x[0].emplace_back(std::vector<T>{0.0}, 1, 1);
  x[0].emplace_back(std::vector<T>{1.0}, 1, 1);
  for (int i = 0; i < 2; ++i)
    M[0].emplace_back(std::vector<T>{1.0}, 1, 1, 1, 1);

  // Points on interval
  auto& _x = x[1].emplace_back(degree - 1, 1);
  for (std::size_t i = 1; i < degree; ++i)
    _x(i - 1, 0) = i / static_cast<T>(degree);

  auto& _M = M[1].emplace_back(degree - 1, 1, degree - 1, 1);
  for (std::size_t i = 0; i < degree - 1; ++i)
    _M(i, 0, i, 0) = 1.0;

  return {std::move(x), std::move(M)};
}
//----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::array<std::vector<impl::mdarray_t<T, 2>>, 4>,
          std::array<std::vector<impl::mdarray_t<T, 4>>, 4>>
vtk_data_triangle(std::size_t degree)
{
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(cell::type::triangle);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;

  // Points at vertices
  x[0].emplace_back(std::vector<T>{0., 0.}, 1, 2);
  x[0].emplace_back(std::vector<T>{1., 0.}, 1, 2);
  x[0].emplace_back(std::vector<T>{0., 1.}, 1, 2);
  for (int i = 0; i < 3; ++i)
    M[0].emplace_back(std::vector<T>{1.0}, 1, 1, 1, 1);

  // Points on edges
  {
    std::array<impl::mdspan_t<T, 2>, 3> xview;
    for (int i = 0; i < 3; ++i)
    {
      auto& _x = x[1].emplace_back(degree - 1, 2);
      xview[i] = impl::mdspan_t<T, 2>(_x.data(), _x.extents());
    }

    for (std::size_t i = 1; i < degree; ++i)
    {
      xview[0](i - 1, 0) = i / static_cast<T>(degree);
      xview[0](i - 1, 1) = 0;

      xview[1](i - 1, 0) = (degree - i) / static_cast<T>(degree);
      xview[1](i - 1, 1) = i / static_cast<T>(degree);

      xview[2](i - 1, 0) = 0;
      xview[2](i - 1, 1) = (degree - i) / static_cast<T>(degree);
    }

    for (int i = 0; i < 3; ++i)
    {
      auto& _M = M[1].emplace_back(degree - 1, 1, degree - 1, 1);
      for (std::size_t k = 0; k < degree - 1; ++k)
        _M(k, 0, k, 0) = 1.0;
    }
  }

  // Interior points
  if (degree >= 3)
  {
    auto& _x = x[2].emplace_back(vtk_triangle_points<T>(degree - 3));
    auto& _M = M[2].emplace_back(_x.extent(0), 1, _x.extent(0), 1);
    for (std::size_t k = 0; k < _M.extent(0); ++k)
      _M(k, 0, k, 0) = 1.0;
  }
  else
  {
    x[2].emplace_back(0, 2);
    M[2].emplace_back(0, 1, 0, 1);
  }

  return {std::move(x), std::move(M)};
}
//----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::array<std::vector<impl::mdarray_t<T, 2>>, 4>,
          std::array<std::vector<impl::mdarray_t<T, 4>>, 4>>
vtk_data_quadrilateral(std::size_t degree)
{
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(cell::type::quadrilateral);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;

  // Points at vertices
  x[0].emplace_back(std::vector<T>{0., 0.}, 1, 2);
  x[0].emplace_back(std::vector<T>{1., 0.}, 1, 2);
  x[0].emplace_back(std::vector<T>{1., 1.}, 1, 2);
  x[0].emplace_back(std::vector<T>{0., 1.}, 1, 2);
  for (int i = 0; i < 4; ++i)
    M[0].emplace_back(std::vector<T>{1.0}, 1, 1, 1, 1);

  // Points on edges
  {
    std::array<impl::mdspan_t<T, 2>, 4> xview;
    for (int i = 0; i < 4; ++i)
    {
      auto& _x = x[1].emplace_back(degree - 1, 2);
      xview[i] = impl::mdspan_t<T, 2>(_x.data(), _x.extents());
    }

    for (std::size_t i = 1; i < degree; ++i)
    {
      xview[0](i - 1, 0) = i / static_cast<T>(degree);
      xview[0](i - 1, 1) = 0;

      xview[1](i - 1, 0) = 1;
      xview[1](i - 1, 1) = i / static_cast<T>(degree);

      xview[2](i - 1, 0) = i / static_cast<T>(degree);
      xview[2](i - 1, 1) = 1;

      xview[3](i - 1, 0) = 0;
      xview[3](i - 1, 1) = i / static_cast<T>(degree);
    }

    for (int i = 0; i < 4; ++i)
    {
      auto& _M = M[1].emplace_back(degree - 1, 1, degree - 1, 1);
      for (std::size_t k = 0; k < degree - 1; ++k)
        _M(k, 0, k, 0) = 1.0;
    }
  }

  // Interior points
  {
    auto& _x = x[2].emplace_back((degree - 1) * (degree - 1), 2);
    int n = 0;
    for (std::size_t j = 1; j < degree; ++j)
    {
      for (std::size_t i = 1; i < degree; ++i)
      {
        _x(n, 0) = i / static_cast<T>(degree);
        _x(n, 1) = j / static_cast<T>(degree);
        ++n;
      }
    }

    auto& _M = M[2].emplace_back(_x.extent(0), 1, _x.extent(0), 1);
    for (std::size_t k = 0; k < _x.extent(0); ++k)
      _M(k, 0, k, 0) = 1.0;
  }

  return {std::move(x), std::move(M)};
}
//----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::array<std::vector<impl::mdarray_t<T, 2>>, 4>,
          std::array<std::vector<impl::mdarray_t<T, 4>>, 4>>
vtk_data_tetrahedron(std::size_t degree)
{
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(cell::type::tetrahedron);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;

  // Points at vertices
  x[0].emplace_back(std::vector<T>{0., 0., 0.}, 1, 3);
  x[0].emplace_back(std::vector<T>{1., 0., 0.}, 1, 3);
  x[0].emplace_back(std::vector<T>{0., 1., 0.}, 1, 3);
  x[0].emplace_back(std::vector<T>{0., 0., 1.}, 1, 3);
  for (int i = 0; i < 4; ++i)
    M[0].emplace_back(std::vector<T>{1.0}, 1, 1, 1, 1);

  // Points on edges
  {
    std::array<impl::mdspan_t<T, 2>, 6> xview;
    for (int i = 0; i < 6; ++i)
    {
      auto& _x = x[1].emplace_back(degree - 1, 3);
      xview[i] = impl::mdspan_t<T, 2>(_x.data(), _x.extents());
    }

    for (std::size_t i = 1; i < degree; ++i)
    {
      xview[0](i - 1, 0) = i / static_cast<T>(degree);
      xview[0](i - 1, 1) = 0;
      xview[0](i - 1, 2) = 0;

      xview[1](i - 1, 0) = (degree - i) / static_cast<T>(degree);
      xview[1](i - 1, 1) = i / static_cast<T>(degree);
      xview[1](i - 1, 2) = 0;

      xview[2](i - 1, 0) = 0;
      xview[2](i - 1, 1) = (degree - i) / static_cast<T>(degree);
      xview[2](i - 1, 2) = 0;

      xview[3](i - 1, 0) = 0;
      xview[3](i - 1, 1) = 0;
      xview[3](i - 1, 2) = i / static_cast<T>(degree);

      xview[4](i - 1, 0) = (degree - i) / static_cast<T>(degree);
      xview[4](i - 1, 1) = 0;
      xview[4](i - 1, 2) = i / static_cast<T>(degree);

      xview[5](i - 1, 0) = 0;
      xview[5](i - 1, 1) = (degree - i) / static_cast<T>(degree);
      xview[5](i - 1, 2) = i / static_cast<T>(degree);
    }

    for (int i = 0; i < 6; ++i)
    {
      auto& _M = M[1].emplace_back(degree - 1, 1, degree - 1, 1);
      for (std::size_t k = 0; k < degree - 1; ++k)
        _M(k, 0, k, 0) = 1.0;
    }
  }

  // Points on faces
  if (degree >= 3)
  {
    const auto pts = vtk_triangle_points<T>(degree - 3);
    std::array<impl::mdspan_t<T, 2>, 4> xview;
    for (int i = 0; i < 4; ++i)
    {
      auto& _x = x[2].emplace_back(pts.extent(0), 3);
      xview[i] = impl::mdspan_t<T, 2>(_x.data(), _x.extents());
    }

    for (std::size_t i = 0; i < pts.extent(0); ++i)
    {
      const T x0 = pts(i, 0);
      const T x1 = pts(i, 1);

      xview[0](i, 0) = x0;
      xview[0](i, 1) = 0;
      xview[0](i, 2) = x1;

      xview[1](i, 0) = 1 - x0 - x1;
      xview[1](i, 1) = x0;
      xview[1](i, 2) = x1;

      xview[2](i, 0) = 0;
      xview[2](i, 1) = x0;
      xview[2](i, 2) = x1;

      xview[3](i, 0) = x0;
      xview[3](i, 1) = x1;
      xview[3](i, 2) = 0;
    }

    for (int i = 0; i < 4; ++i)
    {
      auto& _M = M[2].emplace_back(pts.extent(0), 1, pts.extent(0), 1);
      for (std::size_t k = 0; k < _M.extent(0); ++k)
        _M(k, 0, k, 0) = 1.0;
    }
  }
  else
  {
    for (int i = 0; i < 4; ++i)
    {
      x[2].emplace_back(0, 3);
      M[2].emplace_back(0, 1, 0, 1);
    }
  }

  // Points on volume
  if (degree >= 4)
  {
    auto& _x = x[3].emplace_back(vtk_tetrahedron_points<T>(degree - 4));
    auto& _M = M[3].emplace_back(_x.extent(0), 1, _x.extent(0), 1);
    for (std::size_t k = 0; k < _M.extent(0); ++k)
      _M(k, 0, k, 0) = 1.0;
  }
  else
  {
    x[3].emplace_back(0, 3);
    M[3].emplace_back(0, 1, 0, 1);
  }

  return {std::move(x), std::move(M)};
}
//----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::array<std::vector<impl::mdarray_t<T, 2>>, 4>,
          std::array<std::vector<impl::mdarray_t<T, 4>>, 4>>
vtk_data_hexahedron(std::size_t degree)
{
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(cell::type::hexahedron);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;

  // Points at vertices
  x[0].emplace_back(std::vector<T>{0., 0., 0.}, 1, 3);
  x[0].emplace_back(std::vector<T>{1., 0., 0.}, 1, 3);
  x[0].emplace_back(std::vector<T>{1., 1., 0.}, 1, 3);
  x[0].emplace_back(std::vector<T>{0., 1., 0.}, 1, 3);
  x[0].emplace_back(std::vector<T>{0., 0., 1.}, 1, 3);
  x[0].emplace_back(std::vector<T>{1., 0., 1.}, 1, 3);
  x[0].emplace_back(std::vector<T>{1., 1., 1.}, 1, 3);
  x[0].emplace_back(std::vector<T>{0., 1., 1.}, 1, 3);
  for (int i = 0; i < 8; ++i)
    M[0].emplace_back(std::vector<T>{1.0}, 1, 1, 1, 1);

  // Points on edges
  {
    std::array<impl::mdspan_t<T, 2>, 12> xview;
    for (int i = 0; i < 12; ++i)
    {
      auto& _x = x[1].emplace_back(degree - 1, 3);
      xview[i] = impl::mdspan_t<T, 2>(_x.data(), _x.extents());
    }

    for (std::size_t i = 1; i < degree; ++i)
    {
      xview[0](i - 1, 0) = i / static_cast<T>(degree);
      xview[0](i - 1, 1) = 0;
      xview[0](i - 1, 2) = 0;

      xview[1](i - 1, 0) = 1;
      xview[1](i - 1, 1) = i / static_cast<T>(degree);
      xview[1](i - 1, 2) = 0;

      xview[2](i - 1, 0) = i / static_cast<T>(degree);
      xview[2](i - 1, 1) = 1;
      xview[2](i - 1, 2) = 0;

      xview[3](i - 1, 0) = 0;
      xview[3](i - 1, 1) = i / static_cast<T>(degree);
      xview[3](i - 1, 2) = 0;

      xview[4](i - 1, 0) = i / static_cast<T>(degree);
      xview[4](i - 1, 1) = 0;
      xview[4](i - 1, 2) = 1;

      xview[5](i - 1, 0) = 1;
      xview[5](i - 1, 1) = i / static_cast<T>(degree);
      xview[5](i - 1, 2) = 1;

      xview[6](i - 1, 0) = i / static_cast<T>(degree);
      xview[6](i - 1, 1) = 1;
      xview[6](i - 1, 2) = 1;

      xview[7](i - 1, 0) = 0;
      xview[7](i - 1, 1) = i / static_cast<T>(degree);
      xview[7](i - 1, 2) = 1;

      xview[8](i - 1, 0) = 0;
      xview[8](i - 1, 1) = 0;
      xview[8](i - 1, 2) = i / static_cast<T>(degree);

      xview[9](i - 1, 0) = 1;
      xview[9](i - 1, 1) = 0;
      xview[9](i - 1, 2) = i / static_cast<T>(degree);

      xview[10](i - 1, 0) = 1;
      xview[10](i - 1, 1) = 1;
      xview[10](i - 1, 2) = i / static_cast<T>(degree);

      xview[11](i - 1, 0) = 0;
      xview[11](i - 1, 1) = 1;
      xview[11](i - 1, 2) = i / static_cast<T>(degree);
    }

    for (int i = 0; i < 12; ++i)
    {
      auto& _M = M[1].emplace_back(degree - 1, 1, degree - 1, 1);
      for (std::size_t k = 0; k < degree - 1; ++k)
        _M(k, 0, k, 0) = 1.0;
    }
  }

  // Points on faces
  {
    std::array<impl::mdspan_t<T, 2>, 6> xview;
    for (int i = 0; i < 6; ++i)
    {
      auto& _x = x[2].emplace_back((degree - 1) * (degree - 1), 3);
      xview[i] = impl::mdspan_t<T, 2>(_x.data(), _x.extents());
    }

    int n = 0;
    for (std::size_t j = 1; j < degree; ++j)
    {
      for (std::size_t i = 1; i < degree; ++i)
      {
        xview[0](n, 0) = 0;
        xview[0](n, 1) = i / static_cast<T>(degree);
        xview[0](n, 2) = j / static_cast<T>(degree);

        xview[1](n, 0) = 1;
        xview[1](n, 1) = i / static_cast<T>(degree);
        xview[1](n, 2) = j / static_cast<T>(degree);

        xview[2](n, 0) = i / static_cast<T>(degree);
        xview[2](n, 1) = 0;
        xview[2](n, 2) = j / static_cast<T>(degree);

        xview[3](n, 0) = i / static_cast<T>(degree);
        xview[3](n, 1) = 1;
        xview[3](n, 2) = j / static_cast<T>(degree);

        xview[4](n, 0) = i / static_cast<T>(degree);
        xview[4](n, 1) = j / static_cast<T>(degree);
        xview[4](n, 2) = 0;

        xview[5](n, 0) = i / static_cast<T>(degree);
        xview[5](n, 1) = j / static_cast<T>(degree);
        xview[5](n, 2) = 1;

        ++n;
      }
    }

    for (int i = 0; i < 6; ++i)
    {
      auto& _M = M[2].emplace_back(xview.front().extent(0), 1,
                                   xview.front().extent(0), 1);
      for (std::size_t k = 0; k < _M.extent(0); ++k)
        _M(k, 0, k, 0) = 1.0;
    }
  }

  // Interior points
  {
    auto& _x = x[3].emplace_back((degree - 1) * (degree - 1) * (degree - 1), 3);
    int n = 0;
    for (std::size_t k = 1; k < degree; ++k)
    {
      for (std::size_t j = 1; j < degree; ++j)
      {
        for (std::size_t i = 1; i < degree; ++i)
        {
          _x(n, 0) = i / static_cast<T>(degree);
          _x(n, 1) = j / static_cast<T>(degree);
          _x(n, 2) = k / static_cast<T>(degree);
          ++n;
        }
      }
    }

    auto& _M = M[3].emplace_back(_x.extent(0), 1, _x.extent(0), 1);
    for (std::size_t k = 0; k < _x.extent(0); ++k)
      _M(k, 0, k, 0) = 1.0;
  }

  return {std::move(x), std::move(M)};
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
    {
      return {lattice::type::chebyshev, lattice::simplex_method::none, false};
    }
    else
    {
      // TODO: is this the best thing to do for simplices?
      return {lattice::type::chebyshev_plus_endpoints,
              lattice::simplex_method::warp, false};
    }
  }
  case element::lagrange_variant::chebyshev_isaac:
  {
    if (celltype == cell::type::interval
        or celltype == cell::type::quadrilateral
        or celltype == cell::type::hexahedron)
    {
      return {lattice::type::chebyshev, lattice::simplex_method::none, false};
    }
    else
    {
      // TODO: is this the best thing to do for simplices?
      return {lattice::type::chebyshev_plus_endpoints,
              lattice::simplex_method::isaac, false};
    }
  }
  case element::lagrange_variant::chebyshev_centroid:
    return {lattice::type::chebyshev, lattice::simplex_method::centroid, false};
  case element::lagrange_variant::gl_warped:
  {
    if (celltype == cell::type::interval
        or celltype == cell::type::quadrilateral
        or celltype == cell::type::hexahedron)
    {
      return {lattice::type::gl, lattice::simplex_method::none, false};
    }
    else
    {
      // TODO: is this the best thing to do for simplices?
      return {lattice::type::gl_plus_endpoints, lattice::simplex_method::warp,
              false};
    }
  }
  case element::lagrange_variant::gl_isaac:
  {
    if (celltype == cell::type::interval
        or celltype == cell::type::quadrilateral
        or celltype == cell::type::hexahedron)
    {
      return {lattice::type::gl, lattice::simplex_method::none, false};
    }
    else
    {
      // TODO: is this the best thing to do for simplices?
      return {lattice::type::gl_plus_endpoints, lattice::simplex_method::isaac,
              false};
    }
  }
  case element::lagrange_variant::gl_centroid:
    return {lattice::type::gl, lattice::simplex_method::centroid, false};
  default:
    throw std::runtime_error("Unsupported variant");
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
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

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;
  for (std::size_t i = 0; i < tdim; ++i)
  {
    std::size_t num_ent = cell::num_sub_entities(celltype, i);
    x[i] = std::vector<impl::mdarray_t<T, 2>>(num_ent,
                                              impl::mdarray_t<T, 2>(0, tdim));
    M[i] = std::vector<impl::mdarray_t<T, 4>>(
        num_ent, impl::mdarray_t<T, 4>(0, 1, 0, 1));
  }

  const int lattice_degree
      = celltype == cell::type::triangle
            ? degree + 3
            : (celltype == cell::type::tetrahedron ? degree + 4 : degree + 2);

  // Create points in interior
  const auto [pt, shape] = lattice::create<T>(
      celltype, lattice_degree, lattice_type, false, simplex_method);
  x[tdim].emplace_back(pt, shape);

  const std::size_t num_dofs = shape[0];
  auto& _M = M[tdim].emplace_back(num_dofs, 1, num_dofs, 1);
  for (std::size_t i = 0; i < _M.extent(0); ++i)
    _M(i, 0, i, 0) = 1.0;

  return FiniteElement(
      element::family::P, celltype, degree, {},
      impl::mdspan_t<const T, 2>(math::eye<T>(ndofs).data(), ndofs, ndofs),
      impl::to_mdspan(x), impl::to_mdspan(M), 0, maps::type::identity,
      sobolev::space::L2, true, degree, degree, variant,
      element::dpc_variant::unset);
}
//----------------------------------------------------------------------------
std::vector<std::tuple<std::vector<FiniteElement>, std::vector<int>>>
create_tensor_product_factors(cell::type celltype, int degree,
                              element::lagrange_variant variant)
{
  switch (celltype)
  {
  case cell::type::quadrilateral:
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
    return {{{sub_element, sub_element}, std::move(perm)}};
  }
  case cell::type::hexahedron:
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
            perm[p++] = 8 + 12 * n + 6 * n * n + i + n * j + n * n * k;
        }
      }
    }
    return {{{sub_element, sub_element, sub_element}, std::move(perm)}};
  }
  default:
    return {};
  }
}
//----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement create_vtk_element(cell::type celltype, std::size_t degree,
                                 bool discontinuous)
{
  if (celltype == cell::type::point)
    throw std::runtime_error("Invalid celltype");

  if (degree == 0)
    throw std::runtime_error("Cannot create a degree 0 VTK element.");

  // DOF transformation don't yet work on this element, so throw runtime
  // error if trying to make continuous version
  if (!discontinuous)
    throw std::runtime_error("Continuous VTK element not yet supported.");

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;
  switch (celltype)
  {
  case cell::type::interval:
    std::tie(x, M) = vtk_data_interval<T>(degree);
    break;
  case cell::type::triangle:
    std::tie(x, M) = vtk_data_triangle<T>(degree);
    break;
  case cell::type::quadrilateral:
    std::tie(x, M) = vtk_data_quadrilateral<T>(degree);
    break;
  case cell::type::tetrahedron:
    std::tie(x, M) = vtk_data_tetrahedron<T>(degree);
    break;
  case cell::type::hexahedron:
    std::tie(x, M) = vtk_data_hexahedron<T>(degree);
    break;
  default:
    throw std::runtime_error("Unsupported cell type.");
  }

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t ndofs = polyset::dim(celltype, degree);
  if (discontinuous)
  {
    auto [_x, _xshape, _M, _Mshape] = element::make_discontinuous(
        impl::to_mdspan(x), impl::to_mdspan(M), tdim, 1);
    return FiniteElement(
        element::family::P, celltype, degree, {},
        impl::mdspan_t<const T, 2>(math::eye<T>(ndofs).data(), ndofs, ndofs),
        impl::to_mdspan(_x, _xshape), impl::to_mdspan(_M, _Mshape), 0,
        maps::type::identity, sobolev::space::L2, discontinuous, degree, degree,
        element::lagrange_variant::vtk, element::dpc_variant::unset);
  }
  else
  {
    return FiniteElement(
        element::family::P, celltype, degree, {},
        impl::mdspan_t<const T, 2>(math::eye<T>(ndofs).data(), ndofs, ndofs),
        impl::to_mdspan(x), impl::to_mdspan(M), 0, maps::type::identity,
        sobolev::space::H1, discontinuous, degree, degree,
        element::lagrange_variant::vtk, element::dpc_variant::unset);
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement create_legendre(cell::type celltype, int degree,
                              bool discontinuous)
{
  if (!discontinuous)
    throw std::runtime_error("Legendre variant must be discontinuous");

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t ndofs = polyset::dim(celltype, degree);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;

  // Evaluate moment space at quadrature points
  const auto [_pts, wts] = quadrature::make_quadrature<T>(
      quadrature::type::Default, celltype, degree * 2);
  assert(!wts.empty());
  impl::mdspan_t<const T, 2> pts(_pts.data(), wts.size(),
                               _pts.size() / wts.size());
  const auto [_phi, pshape] = polynomials::tabulate(polynomials::type::legendre,
                                                    celltype, degree, pts);
  impl::mdspan_t<const T, 2> phi(_phi.data(), pshape);
  for (std::size_t d = 0; d < tdim; ++d)
  {
    for (std::size_t e = 0; e < topology[d].size(); ++e)
    {
      x[d].emplace_back(0, tdim);
      M[d].emplace_back(0, 1, 0, 1);
    }
  }

  auto& _x = x[tdim].emplace_back(pts.extents());
  std::copy_n(pts.data_handle(), pts.size(), _x.data());
  auto& _M = M[tdim].emplace_back(ndofs, 1, pts.extent(0), 1);
  for (std::size_t i = 0; i < ndofs; ++i)
    for (std::size_t j = 0; j < pts.extent(0); ++j)
      _M(i, 0, j, 0) = phi(i, j) * wts[j];

  sobolev::space space
      = discontinuous ? sobolev::space::L2 : sobolev::space::H1;
  return FiniteElement(
      element::family::P, celltype, degree, {},
      impl::mdspan_t<T, 2>(math::eye<T>(ndofs).data(), ndofs, ndofs),
      impl::to_mdspan(x), impl::to_mdspan(M), 0, maps::type::identity, space,
      discontinuous, degree, degree, element::lagrange_variant::legendre,
      element::dpc_variant::unset);
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement create_bernstein(cell::type celltype, int degree,
                               bool discontinuous)
{
  assert(degree > 0);
  if (celltype != cell::type::interval and celltype != cell::type::triangle
      and celltype != cell::type::tetrahedron)
  {
    throw std::runtime_error(
        "Bernstein elements are currently only supported on simplices.");
  }

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;

  const std::array<std::size_t, 4> nb
      = {1,
         static_cast<std::size_t>(polynomials::dim(
             polynomials::type::bernstein, cell::type::interval, degree)),
         static_cast<std::size_t>(polynomials::dim(
             polynomials::type::bernstein, cell::type::triangle, degree)),
         static_cast<std::size_t>(polynomials::dim(
             polynomials::type::bernstein, cell::type::tetrahedron, degree))};

  constexpr std::array<cell::type, 4> ct
      = {cell::type::point, cell::type::interval, cell::type::triangle,
         cell::type::tetrahedron};

  const std::array<std::size_t, 4> nb_interior
      = {1, degree < 2 ? 0 : nb[1] - 2, degree < 3 ? 0 : nb[2] + 3 - 3 * nb[1],
         degree < 4 ? 0 : nb[3] + 6 * nb[1] - 4 * nb[2] - 4};

  std::array<std::vector<int>, 4> bernstein_bubbles;
  bernstein_bubbles[0].push_back(0);
  { // scope
    int ib = 0;
    for (int i = 0; i <= degree; ++i)
    {
      if (i > 0 and i < degree)
      {
        bernstein_bubbles[1].push_back(ib);
      }
      ++ib;
    }
  }
  { // scope
    int ib = 0;
    for (int i = 0; i <= degree; ++i)
    {
      for (int j = 0; j <= degree - i; ++j)
      {
        if (i > 0 and j > 0 and i + j < degree)
          bernstein_bubbles[2].push_back(ib);
        ++ib;
      }
    }
  }
  { // scope
    int ib = 0;
    for (int i = 0; i <= degree; ++i)
    {
      for (int j = 0; j <= degree - i; ++j)
      {
        for (int k = 0; k <= degree - i - j; ++k)
        {
          if (i > 0 and j > 0 and k > 0 and i + j + k < degree)
            bernstein_bubbles[3].push_back(ib);
          ++ib;
        }
      }
    }
  }

  for (std::size_t v = 0; v < topology[0].size(); ++v)
  {
    const auto [entity, shape] = cell::sub_entity_geometry<T>(celltype, 0, v);
    x[0].emplace_back(entity, shape[0], shape[1]);
    M[0].emplace_back(std::vector<T>{1.0}, 1, 1, 1, 1);
  }

  for (std::size_t d = 1; d <= tdim; ++d)
  {
    if (nb_interior[d] == 0)
    {
      for (std::size_t e = 0; e < topology[d].size(); ++e)
      {
        x[d].emplace_back(0, tdim);
        M[d].emplace_back(0, 1, 0, 1);
      }
    }
    else
    {
      const auto [_pts, wts] = quadrature::make_quadrature<T>(
          quadrature::type::Default, ct[d], degree * 2);
      assert(!wts.empty());
      impl::mdspan_t<const T, 2> pts(_pts.data(), wts.size(),
                                   _pts.size() / wts.size());

      const auto [_phi, pshape] = polynomials::tabulate(
          polynomials::type::legendre, ct[d], degree, pts);
      impl::mdspan_t<const T, 2> phi(_phi.data(), pshape);
      const auto [_bern, bshape] = polynomials::tabulate(
          polynomials::type::bernstein, ct[d], degree, pts);
      impl::mdspan_t<const T, 2> bern(_bern.data(), bshape);

      assert(phi.extent(0) == nb[d]);
      const std::size_t npts = pts.extent(0);

      impl::mdarray_t<T, 2> mat(nb[d], nb[d]);
      for (std::size_t i = 0; i < nb[d]; ++i)
        for (std::size_t j = 0; j < nb[d]; ++j)
          for (std::size_t k = 0; k < wts.size(); ++k)
            mat(i, j) += wts[k] * bern(j, k) * phi(i, k);

      impl::mdarray_t<T, 2> minv(mat.extents());
      {
        std::vector<T> id = math::eye<T>(nb[d]);
        impl::mdspan_t<T, 2> _id(id.data(), nb[d], nb[d]);
        impl::mdspan_t<T, 2> _mat(mat.data(), mat.extents());
        std::vector<T> minv_data = math::solve<T>(_mat, _id);
        std::copy(minv_data.begin(), minv_data.end(), minv.data());
      }

      M[d] = std::vector<impl::mdarray_t<T, 4>>(
          cell::num_sub_entities(celltype, d),
          impl::mdarray_t<T, 4>(nb_interior[d], 1, npts, 1));
      for (std::size_t e = 0; e < topology[d].size(); ++e)
      {
        auto [_entity_x, shape] = cell::sub_entity_geometry<T>(celltype, d, e);
        impl::mdspan_t<T, 2> entity_x(_entity_x.data(), shape);
        std::span<const T> x0(entity_x.data_handle(), shape[1]);
        {
          auto& _x = x[d].emplace_back(pts.extent(0), shape[1]);
          for (std::size_t i = 0; i < _x.extent(0); ++i)
            for (std::size_t j = 0; j < _x.extent(1); ++j)
              _x(i, j) = x0[j];
        }

        for (std::size_t j = 0; j < pts.extent(0); ++j)
          for (std::size_t k0 = 0; k0 < pts.extent(1); ++k0)
            for (std::size_t k1 = 0; k1 < shape[1]; ++k1)
              x[d][e](j, k1) += (entity_x(k0 + 1, k1) - x0[k1]) * pts(j, k0);
        for (std::size_t i = 0; i < bernstein_bubbles[d].size(); ++i)
        {
          for (std::size_t p = 0; p < npts; ++p)
          {
            T tmp = 0.0;
            for (std::size_t k = 0; k < phi.extent(0); ++k)
              tmp += phi(k, p) * minv(bernstein_bubbles[d][i], k);
            M[d][e](i, 0, p, 0) = wts[p] * tmp;
          }
        }
      }
    }
  }

  sobolev::space space
      = discontinuous ? sobolev::space::L2 : sobolev::space::H1;
  const std::size_t ndofs = polyset::dim(celltype, degree);
  return FiniteElement(
      element::family::P, celltype, degree, {},
      impl::mdspan_t<T, 2>(math::eye<T>(ndofs).data(), ndofs, ndofs),
      impl::to_mdspan(x), impl::to_mdspan(M), 0, maps::type::identity, space,
      discontinuous, degree, degree, element::lagrange_variant::bernstein,
      element::dpc_variant::unset);
}
//-----------------------------------------------------------------------------
} // namespace

//----------------------------------------------------------------------------
FiniteElement basix::element::create_lagrange(cell::type celltype, int degree,
                                              lagrange_variant variant,
                                              bool discontinuous,
                                              std::vector<int> dof_ordering)
{
  using T = double;

  if (celltype == cell::type::point)
  {
    if (degree != 0)
      throw std::runtime_error("Can only create order 0 Lagrange on a point");

    std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
    std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;
    x[0].emplace_back(1, 0);
    M[0].emplace_back(std::vector<T>{1.0}, 1, 1, 1, 1);
    return FiniteElement(
        family::P, cell::type::point, 0, {},
        impl::mdspan_t<T, 2>(math::eye<T>(1).data(), 1, 1), impl::to_mdspan(x),
        impl::to_mdspan(M), 0, maps::type::identity, sobolev::space::H1,
        discontinuous, degree, degree, element::lagrange_variant::unset,
        element::dpc_variant::unset, {}, dof_ordering);
  }

  if (variant == lagrange_variant::vtk)
    return create_vtk_element<T>(celltype, degree, discontinuous);

  if (variant == lagrange_variant::legendre)
    return create_legendre<T>(celltype, degree, discontinuous);

  if (variant == element::lagrange_variant::bernstein)
  {
    if (degree == 0)
      variant = lagrange_variant::unset;
    else
      return create_bernstein<T>(celltype, degree, discontinuous);
  }

  if (variant == lagrange_variant::unset)
  {
    if (degree < 3)
      variant = element::lagrange_variant::gll_warped;
    else
    {
      throw std::runtime_error(
          "Lagrange elements of degree > 2 need to be given a variant.");
    }
  }

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
    return create_d_lagrange<T>(celltype, degree, variant, lattice_type,
                                simplex_method);
  }

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t ndofs = polyset::dim(celltype, degree);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;
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
      x[i] = std::vector(num_entities, impl::mdarray_t<T, 2>(0, tdim));
      M[i] = std::vector(num_entities, impl::mdarray_t<T, 4>(0, 1, 0, 1));
    }

    const auto [pt, shape]
        = lattice::create<T>(celltype, 0, lattice_type, true, simplex_method);
    x[tdim].emplace_back(pt, shape[0], shape[1]);
    auto& _M = M[tdim].emplace_back(shape[0], 1, shape[0], 1);
    std::fill(_M.data(), _M.data() + _M.size(), 0);
    for (std::size_t i = 0; i < shape[0]; ++i)
      _M(i, 0, i, 0) = 1;
  }
  else
  {
    // Create points at nodes, ordered by topology (vertices first)
    for (std::size_t dim = 0; dim <= tdim; ++dim)
    {
      // Loop over entities of dimension 'dim'
      for (std::size_t e = 0; e < topology[dim].size(); ++e)
      {
        const auto [entity_x, entity_x_shape]
            = cell::sub_entity_geometry<T>(celltype, dim, e);
        if (dim == 0)
        {
          x[dim].emplace_back(entity_x, entity_x_shape[0], entity_x_shape[1]);
          auto& _M
              = M[dim].emplace_back(entity_x_shape[0], 1, entity_x_shape[0], 1);
          std::fill(_M.data(), _M.data() + _M.size(), 0);
          for (std::size_t i = 0; i < entity_x_shape[0]; ++i)
            _M(i, 0, i, 0) = 1;
        }
        else if (dim == tdim)
        {
          const auto [pt, shape] = lattice::create<T>(
              celltype, degree, lattice_type, false, simplex_method);
          x[dim].emplace_back(pt, shape[0], shape[1]);
          auto& _M = M[dim].emplace_back(shape[0], 1, shape[0], 1);
          std::fill(_M.data(), _M.data() + _M.size(), 0);
          for (std::size_t i = 0; i < shape[0]; ++i)
            _M(i, 0, i, 0) = 1;
        }
        else
        {
          cell::type ct = cell::sub_entity_type(celltype, dim, e);
          const auto [pt, shape] = lattice::create<T>(ct, degree, lattice_type,
                                                      false, simplex_method);
          impl::mdspan_t<const T, 2> lattice(pt.data(), shape);
          std::span<const T> x0(entity_x.data(), entity_x_shape[1]);
          impl::mdspan_t<const T, 2> entity_x_view(entity_x.data(),
                                                 entity_x_shape);

          auto& _x = x[dim].emplace_back(shape[0], entity_x_shape[1]);
          for (std::size_t i = 0; i < shape[0]; ++i)
            for (std::size_t j = 0; j < entity_x_shape[1]; ++j)
              _x(i, j) = x0[j];

          for (std::size_t j = 0; j < shape[0]; ++j)
            for (std::size_t k = 0; k < shape[1]; ++k)
              for (std::size_t q = 0; q < tdim; ++q)
                _x(j, q) += (entity_x_view(k + 1, q) - x0[q]) * lattice(j, k);

          auto& _M = M[dim].emplace_back(shape[0], 1, shape[0], 1);
          std::fill(_M.data(), _M.data() + _M.size(), 0);
          for (std::size_t i = 0; i < shape[0]; ++i)
            _M(i, 0, i, 0) = 1;
        }
      }
    }
  }

  std::array<std::vector<mdspan_t<const T, 2>>, 4> xview = impl::to_mdspan(x);
  std::array<std::vector<mdspan_t<const T, 4>>, 4> Mview = impl::to_mdspan(M);
  std::array<std::vector<std::vector<T>>, 4> xbuffer;
  std::array<std::vector<std::vector<T>>, 4> Mbuffer;
  if (discontinuous)
  {
    std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
    std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
    std::tie(xbuffer, xshape, Mbuffer, Mshape)
        = make_discontinuous(xview, Mview, tdim, 1);
    xview = impl::to_mdspan(xbuffer, xshape);
    Mview = impl::to_mdspan(Mbuffer, Mshape);
  }

  sobolev::space space
      = discontinuous ? sobolev::space::L2 : sobolev::space::H1;
  auto tensor_factors
      = create_tensor_product_factors(celltype, degree, variant);
  return FiniteElement(
      family::P, celltype, degree, {},
      impl::mdspan_t<T, 2>(math::eye<T>(ndofs).data(), ndofs, ndofs), xview,
      Mview, 0, maps::type::identity, space, discontinuous, degree, degree,
      variant, dpc_variant::unset, tensor_factors, dof_ordering);
}
//-----------------------------------------------------------------------------
