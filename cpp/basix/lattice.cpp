// Copyright (c) 2020 Chris Richardson, Garth N. Wells, and Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "lattice.h"
#include "cell.h"
#include "math.h"
#include "polyset.h"
#include "quadrature.h"
#include <cmath>
#include <math.h>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xpad.hpp>

using namespace basix;
namespace stdex = std::experimental;
// using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;

namespace
{
//-----------------------------------------------------------------------------
std::vector<double> linspace(double x0, double x1, std::size_t n)
{
  if (n == 0)
    return {};
  else if (n == 1)
    return {x0};

  std::vector<double> p(n, x0);
  p.back() = x1;
  const double delta = (x1 - x0) / (n - 1);
  for (std::size_t i = 1; i < p.size() - 1; ++i)
    p[i] += i * delta;
  return p;
}
//-----------------------------------------------------------------------------
std::vector<double> create_interval_equispaced_new(std::size_t n, bool exterior)
{
  const double h = exterior ? 0 : 1.0 / static_cast<double>(n);
  const std::size_t num_pts = exterior ? n + 1 : n - 1;
  return linspace(h, 1.0 - h, num_pts);
}
//-----------------------------------------------------------------------------
std::vector<double> create_interval_gll_new(std::size_t n, bool exterior)
{
  if (n == 0)
    return {0.5};
  else
  {
    const std::vector<double> pts = quadrature::get_gll_points(n + 1);
    const std::size_t b = exterior ? 0 : 1;
    std::vector<double> x(n + 1 - 2 * b);
    if (exterior)
    {
      x[0] = pts[0];
      x[n] = pts[1];
    }

    for (std::size_t j = 2; j < n + 1; ++j)
      x[j - 1 - b] = pts[j];

    return x;
  }
}
//-----------------------------------------------------------------------------
std::vector<double> create_interval_chebyshev_new(std::size_t n, bool exterior)
{
  if (exterior)
  {
    throw std::runtime_error(
        "Chebyshev points including endpoints are not supported.");
  }

  std::vector<double> x(n - 1);
  for (std::size_t i = 1; i < n; ++i)
    x[i - 1] = 0.5 - std::cos((2 * i - 1) * M_PI / (2 * n - 2)) / 2.0;

  return x;
}
//-----------------------------------------------------------------------------
std::vector<double> create_interval_gl_new(std::size_t n, bool exterior)
{
  if (exterior)
  {
    throw std::runtime_error(
        "GL points including endpoints are not supported.");
  }

  if (n == 0)
    return {0.5};
  else
    return quadrature::get_gl_points(n - 1);
}
//-----------------------------------------------------------------------------
std::vector<double> create_interval_gl_plus_endpoints_new(std::size_t n,
                                                          bool exterior)
{
  std::vector<double> x_gl = create_interval_gl_new(n, false);
  if (!exterior)
    return x_gl;
  else
  {
    std::vector<double> x(n + 1);
    x[0] = 0.0;
    x[n] = 1.0;
    for (std::size_t i = 0; i < n - 1; ++i)
      x[i + 1] = x_gl[i];

    return x;
  }
}
//-----------------------------------------------------------------------------
std::vector<double> create_interval_chebyshev_plus_endpoints_new(std::size_t n,
                                                                 bool exterior)
{
  std::vector<double> x_cheb = create_interval_chebyshev_new(n, false);
  if (!exterior)
    return x_cheb;
  else
  {
    std::vector<double> x(n + 1);
    x[0] = 0.0;
    x[n] = 1.0;
    for (std::size_t i = 0; i < n - 1; ++i)
      x[i + 1] = x_cheb[i];
    return x;
  }
}
//-----------------------------------------------------------------------------
std::vector<double>
create_interval_new(std::size_t n, lattice::type lattice_type, bool exterior)
{
  if (n == 0)
    return {0.5};
  else
  {
    switch (lattice_type)
    {
    case lattice::type::equispaced:
      return create_interval_equispaced_new(n, exterior);
    case lattice::type::gll:
      return create_interval_gll_new(n, exterior);
    case lattice::type::chebyshev:
      return create_interval_chebyshev_new(n, exterior);
    case lattice::type::gl:
      return create_interval_gl_new(n, exterior);
    case lattice::type::chebyshev_plus_endpoints:
      return create_interval_chebyshev_plus_endpoints_new(n, exterior);
    case lattice::type::gl_plus_endpoints:
      return create_interval_gl_plus_endpoints_new(n, exterior);
    default:
      throw std::runtime_error("Unrecognised lattice type.");
    }
  }
}
//-----------------------------------------------------------------------------
stdex::mdarray<double, stdex::dextents<std::size_t, 2>>
tabulate_dlagrange(std::size_t n, xtl::span<const double> x)
{
  std::vector<double> equi_pts(n + 1);
  for (std::size_t i = 0; i < equi_pts.size(); ++i)
    equi_pts[i] = static_cast<double>(i) / static_cast<double>(n);
  stdex::mdspan<const double, stdex::dextents<std::size_t, 2>> equi_pts_v(
      equi_pts.data(), n + 1, 1);

  const auto [dual_values_b, dshape]
      = polyset::tabulate(cell::type::interval, n, 0, equi_pts_v);
  stdex::mdspan<const double, stdex::dextents<std::size_t, 3>> dual_values(
      dual_values_b.data(), dshape);

  std::vector<double> dualmat_b(dual_values.extent(1) * dual_values.extent(2));
  stdex::mdspan<double, stdex::dextents<std::size_t, 2>> dualmat(
      dualmat_b.data(), dual_values.extent(1), dual_values.extent(2));
  for (std::size_t i = 0; i < dualmat.extent(0); ++i)
    for (std::size_t j = 0; j < dualmat.extent(1); ++j)
      dualmat(i, j) = dual_values(0, i, j);

  using cmdspan2_t
      = stdex::mdspan<const double,
                      stdex::extents<std::size_t, stdex::dynamic_extent, 1>>;
  const auto [tabulated_values_b, tshape] = polyset::tabulate(
      cell::type::interval, n, 0, cmdspan2_t(x.data(), x.size(), 1));
  stdex::mdspan<const double, stdex::dextents<std::size_t, 3>> tabulated_values(
      tabulated_values_b.data(), tshape);

  std::vector<double> tabulated_b(tabulated_values.extent(1)
                                  * tabulated_values.extent(2));
  stdex::mdspan<double, stdex::dextents<std::size_t, 2>> tabulated(
      tabulated_b.data(), tabulated_values.extent(1),
      tabulated_values.extent(2));

  for (std::size_t i = 0; i < tabulated.extent(0); ++i)
    for (std::size_t j = 0; j < tabulated.extent(1); ++j)
      tabulated(i, j) = tabulated_values(0, i, j);

  std::vector<double> c = math::solve(dualmat, tabulated);
  return stdex::mdarray<double, stdex::dextents<std::size_t, 2>>(
      c, tabulated.extents());
}
//-----------------------------------------------------------------------------
std::vector<double> warp_function(lattice::type lattice_type, int n,
                                  xtl::span<const double> x)
{
  std::vector<double> pts = create_interval_new(n, lattice_type, true);
  for (int i = 0; i < n + 1; ++i)
    pts[i] -= static_cast<double>(i) / static_cast<double>(n);

  const stdex::mdarray<double, stdex::dextents<std::size_t, 2>> v
      = tabulate_dlagrange(n, x);
  std::vector<double> w(v.extent(1), 0);
  for (std::size_t i = 0; i < v.extent(0); ++i)
    for (std::size_t j = 0; j < v.extent(1); ++j)
      w[j] += v(i, j) * pts[i];

  return w;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 2>>
create_quad(std::size_t n, lattice::type lattice_type, bool exterior)
{
  if (n == 0)
    return {{0.5, 0.5}, {1, 2}};
  else
  {
    const std::vector<double> r
        = create_interval_new(n, lattice_type, exterior);
    const std::size_t m = r.size();
    std::array<std::size_t, 2> shape = {m * m, 2};
    std::vector<double> xb(shape[0] * shape[1]);
    stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, 2>>
        x(xb.data(), shape);
    std::size_t c = 0;
    for (std::size_t j = 0; j < m; ++j)
    {
      for (std::size_t i = 0; i < m; ++i)
      {
        x(c, 0) = r[i];
        x(c, 1) = r[j];
        c++;
      }
    }

    return {xb, shape};
  }
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 2>>
create_hex(int n, lattice::type lattice_type, bool exterior)
{
  if (n == 0)
    return {{0.5, 0.5, 0.5}, {1, 3}};
  else
  {
    const std::vector<double> r
        = create_interval_new(n, lattice_type, exterior);
    const std::size_t m = r.size();

    std::array<std::size_t, 2> shape = {m * m * m, 3};

    std::vector<double> xb(shape[0] * shape[1]);
    stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, 3>>
        x(xb.data(), shape);

    int c = 0;
    for (std::size_t k = 0; k < m; ++k)
    {
      for (std::size_t j = 0; j < m; ++j)
      {
        for (std::size_t i = 0; i < m; ++i)
        {
          x(c, 0) = r[i];
          x(c, 1) = r[j];
          x(c, 2) = r[k];
          c++;
        }
      }
    }

    return {xb, shape};
  }
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 2>>
create_tri_equispaced_new(std::size_t n, bool exterior)
{
  const std::size_t b = exterior ? 0 : 1;

  std::array<std::size_t, 2> shape = {(n - 3 * b + 1) * (n - 3 * b + 2) / 2, 2};
  std::vector<double> _p(shape[0] * shape[1]);
  stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, 2>>
      p(_p.data(), shape);

  // Displacement from GLL points in 1D, scaled by 1 /(r * (1 - r))
  std::vector<double> r = linspace(0.0, 1.0, 2 * n + 1);
  int c = 0;
  for (std::size_t j = b; j < (n - b + 1); ++j)
  {
    for (std::size_t i = b; i < (n - b + 1 - j); ++i)
    {
      p(c, 0) = r[2 * i];
      p(c, 1) = r[2 * j];
      ++c;
    }
  }

  return {_p, shape};
}
//-----------------------------------------------------------------------------

/// Warp points: see Hesthaven and Warburton, Nodal Discontinuous
/// Galerkin Methods, pp. 175-180, https://doi.org/10.1007/978-0-387-72067-8_6
std::pair<std::vector<double>, std::array<std::size_t, 2>>
create_tri_warped_new(std::size_t n, lattice::type lattice_type, bool exterior)
{
  const std::size_t b = exterior ? 0 : 1;

  // Points
  std::array<std::size_t, 2> shape = {(n - 3 * b + 1) * (n - 3 * b + 2) / 2, 2};
  std::vector<double> _p(shape[0] * shape[1]);
  stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, 2>>
      p(_p.data(), shape);

  // Displacement from GLL points in 1D, scaled by 1 /(r * (1 - r))
  const std::vector<double> r = linspace(0.0, 1.0, 2 * n + 1);

  std::vector<double> wbar = warp_function(lattice_type, n, r);
  for (std::size_t i = 1; i < 2 * n - 1; ++i)
    wbar[i] /= r[i] * (1.0 - r[i]);

  int c = 0;
  for (std::size_t j = b; j < (n - b + 1); ++j)
  {
    for (std::size_t i = b; i < (n - b + 1 - j); ++i)
    {
      const double x = r[2 * i];
      const double y = r[2 * j];
      p(c, 0) = x;
      p(c, 1) = y;
      const std::size_t l = n - j - i;
      const double a = r[2 * l];
      p(c, 0) += x * (a * wbar[n + i - l] + y * wbar[n + i - j]);
      p(c, 1) += y * (a * wbar[n + j - l] + x * wbar[n + j - i]);
      ++c;
    }
  }

  return {_p, shape};
}
//-----------------------------------------------------------------------------
std::vector<double> isaac_point(lattice::type lattice_type,
                                xtl::span<const std::size_t> a)
{
  if (a.size() == 1)
    return {1};
  else
  {
    std::vector<double> res(a.size(), 0);
    double denominator = 0;
    std::vector<std::size_t> sub_a(std::next(a.begin()), a.end());
    const std::size_t size = std::reduce(a.begin(), a.end());
    std::vector<double> x = create_interval_new(size, lattice_type, true);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
      if (i > 0)
        sub_a[i - 1] = a[i - 1];
      const std::size_t sub_size = size - a[i];
      const std::vector<double> sub_res = isaac_point(lattice_type, sub_a);
      for (std::size_t j = 0; j < sub_res.size(); ++j)
        res[j < i ? j : j + 1] += x[sub_size] * sub_res[j];
      denominator += x[sub_size];
    }

    for (std::size_t i = 0; i < res.size(); ++i)
      res[i] /= denominator;

    return res;
  }
}
//-----------------------------------------------------------------------------

/// Warp points, See: Isaac, Recursive, Parameter-Free, Explicitly
/// Defined Interpolation Nodes for Simplices,
/// http://dx.doi.org/10.1137/20M1321802.
std::pair<std::vector<double>, std::array<std::size_t, 2>>
create_tri_isaac_new(std::size_t n, lattice::type lattice_type, bool exterior)
{
  const std::size_t b = exterior ? 0 : 1;
  std::array<std::size_t, 2> shape = {(n - 3 * b + 1) * (n - 3 * b + 2) / 2, 2};
  std::vector<double> _p(shape[0] * shape[1]);
  stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, 2>>
      p(_p.data(), shape);

  int c = 0;
  for (std::size_t j = b; j < (n - b + 1); ++j)
  {
    for (std::size_t i = b; i < (n - b + 1 - j); ++i)
    {
      const std::vector<double> isaac_p = isaac_point(
          lattice_type, std::vector<std::size_t>{i, j, n - i - j});
      for (std::size_t k = 0; k < 2; ++k)
        p(c, k) = isaac_p[k];
      ++c;
    }
  }

  return {_p, shape};
}
//-----------------------------------------------------------------------------

/// See: Blyth, and Pozrikidis, A Lobatto interpolation grid over the
/// triangle, https://dx.doi.org/10.1093/imamat/hxh077
std::pair<std::vector<double>, std::array<std::size_t, 2>>
create_tri_centroid_new(std::size_t n, lattice::type lattice_type,
                        bool exterior)
{
  if (exterior)
  {
    throw std::runtime_error(
        "Centroid method not implemented to include boundaries");
  }

  std::array<std::size_t, 2> shape = {(n - 2) * (n - 1) / 2, 2};
  std::vector<double> _p(shape[0] * shape[1]);
  stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, 2>>
      p(_p.data(), shape);

  std::vector<double> x = create_interval_new(n, lattice_type, false);
  int c = 0;
  for (std::size_t i = 0; i + 1 < x.size(); ++i)
  {
    const double xi = x[i];
    for (std::size_t j = 0; j + i + 1 < x.size(); ++j)
    {
      const double xj = x[j];
      const double xk = x[i + j + 1];
      p(c, 0) = (2 * xj + xk - xi) / 3;
      p(c, 1) = (2 * xi + xk - xj) / 3;
      ++c;
    }
  }

  return {_p, shape};
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 2>>
create_tri_new(std::size_t n, lattice::type lattice_type, bool exterior,
               lattice::simplex_method simplex_method)
{
  if (n == 0)
    return {{1.0 / 3.0, 1.0 / 3.0}, {1, 2}};
  else if (lattice_type == lattice::type::equispaced)
    return create_tri_equispaced_new(n, exterior);
  else
  {
    switch (simplex_method)
    {
    case lattice::simplex_method::warp:
      return create_tri_warped_new(n, lattice_type, exterior);
    case lattice::simplex_method::isaac:
      return create_tri_isaac_new(n, lattice_type, exterior);
    case lattice::simplex_method::centroid:
      return create_tri_centroid_new(n, lattice_type, exterior);
    case lattice::simplex_method::none:
    {
      // Methods will all agree when n <= 3
      if (n <= 3)
        return create_tri_warped_new(n, lattice_type, exterior);
      else
      {
        throw std::runtime_error(
            "A simplex type must be given to create points on a triangle.");
      }
    }
    default:
      throw std::runtime_error("Unrecognised simplex type.");
    }
  }
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 2>>
create_tet_equispaced(std::size_t n, bool exterior)
{
  const std::size_t b = exterior ? 0 : 1;
  const std::vector<double> r = linspace(0.0, 1.0, 2 * n + 1);

  std::array<std::size_t, 2> shape
      = {(n - 4 * b + 1) * (n - 4 * b + 2) * (n - 4 * b + 3) / 6, 3};
  std::vector<double> xb(shape[0] * shape[1]);
  stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, 3>>
      x(xb.data(), shape);

  std::size_t c = 0;
  for (std::size_t k = b; k < (n - b + 1); ++k)
  {
    for (std::size_t j = b; j < (n - b + 1 - k); ++j)
    {
      for (std::size_t i = b; i < (n - b + 1 - j - k); ++i)
      {
        x(c, 0) = r[2 * i];
        x(c, 1) = r[2 * j];
        x(c, 2) = r[2 * k];
        ++c;
      }
    }
  }

  return {xb, shape};
}
//-----------------------------------------------------------------------------

/// See: Isaac, Recursive, Parameter-Free, Explicitly Defined Interpolation
/// Nodes for Simplices http://dx.doi.org/10.1137/20M1321802
std::pair<std::vector<double>, std::array<std::size_t, 2>>
create_tet_isaac(std::size_t n, lattice::type lattice_type, bool exterior)
{
  const std::size_t b = exterior ? 0 : 1;

  // xt::xtensor<double, 2> x(
  //     {(n - 4 * b + 1) * (n - 4 * b + 2) * (n - 4 * b + 3) / 6, 3});
  std::array<std::size_t, 2> shape
      = {(n - 4 * b + 1) * (n - 4 * b + 2) * (n - 4 * b + 3) / 6, 3};
  std::vector<double> xb(shape[0] * shape[1]);
  stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, 3>>
      x(xb.data(), shape);

  int c = 0;
  for (std::size_t k = b; k < (n - b + 1); ++k)
  {
    for (std::size_t j = b; j < (n - b + 1 - k); ++j)
    {
      for (std::size_t i = b; i < (n - b + 1 - j - k); ++i)
      {
        std::vector<double> ip = isaac_point(
            lattice_type, std::vector<std::size_t>{i, j, k, n - i - j - k});
        for (std::size_t l = 0; l < 3; ++l)
          x(c, l) = ip[l];
        ++c;
      }
    }
  }

  return {xb, shape};
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 2>>
create_tet_warped(std::size_t n, lattice::type lattice_type, bool exterior)
{
  const std::size_t b = exterior ? 0 : 1;
  const std::vector<double> r = linspace(0.0, 1.0, 2 * n + 1);
  std::vector<double> wbar = warp_function(lattice_type, n, r);
  for (std::size_t i = 1; i < 2 * n - 1; ++i)
    wbar[i] /= r[i] * (1.0 - r[i]);

  std::array<std::size_t, 2> shape
      = {(n - 4 * b + 1) * (n - 4 * b + 2) * (n - 4 * b + 3) / 6, 3};
  std::vector<double> xb(shape[0] * shape[1]);
  stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, 3>>
      p(xb.data(), shape);

  std::size_t c = 0;
  for (std::size_t k = b; k < (n - b + 1); ++k)
  {
    for (std::size_t j = b; j < (n - b + 1 - k); ++j)
    {
      for (std::size_t i = b; i < (n - b + 1 - j - k); ++i)
      {
        const std::size_t l = n - k - j - i;
        const double x = r[2 * i];
        const double y = r[2 * j];
        const double z = r[2 * k];
        const double a = r[2 * l];
        p(c, 0) = x;
        p(c, 1) = y;
        p(c, 2) = z;
        const double dx = x
                          * (a * wbar[n + i - l] + y * wbar[n + i - j]
                             + z * wbar[n + i - k]);
        const double dy = y
                          * (a * wbar[n + j - l] + z * wbar[n + j - k]
                             + x * wbar[n + j - i]);
        const double dz = z
                          * (a * wbar[n + k - l] + x * wbar[n + k - i]
                             + y * wbar[n + k - j]);
        p(c, 0) += dx;
        p(c, 1) += dy;
        p(c, 2) += dz;

        ++c;
      }
    }
  }

  return {xb, shape};
}
//-----------------------------------------------------------------------------

/// See: Blyth, and Pozrikidis, A Lobatto interpolation grid over the
/// triangle, https://dx.doi.org/10.1093/imamat/hxh077
std::pair<std::vector<double>, std::array<std::size_t, 2>>
create_tet_centroid(std::size_t n, lattice::type lattice_type, bool exterior)
{
  if (exterior)
  {
    throw std::runtime_error(
        "Centroid method not implemented to include boundaries");
  }

  const std::vector<double> x = create_interval_new(n, lattice_type, false);

  std::array<std::size_t, 2> shape = {(n - 3) * (n - 2) * (n - 1) / 6, 3};
  std::vector<double> xb(shape[0] * shape[1]);
  stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, 3>>
      p(xb.data(), shape);

  int c = 0;
  for (std::size_t i = 0; i + 2 < x.size(); ++i)
  {
    const double xi = x[i];
    for (std::size_t j = 0; j + i + 2 < x.size(); ++j)
    {
      const double xj = x[j];
      for (std::size_t k = 0; k + j + i + 2 < x.size(); ++k)
      {
        const double xk = x[k];
        const double xl = x[i + j + k + 2];
        p(c, 0) = (3 * xk + xl - xi - xj) / 4;
        p(c, 1) = (3 * xj + xl - xi - xk) / 4;
        p(c, 2) = (3 * xi + xl - xj - xk) / 4;
        ++c;
      }
    }
  }

  return {xb, shape};
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 2>>
create_tet(std::size_t n, lattice::type lattice_type, bool exterior,
           lattice::simplex_method simplex_method)
{
  if (n == 0)
    return {{0.25, 0.25, 0.25}, {1, 3}};
  else if (lattice_type == lattice::type::equispaced)
    return create_tet_equispaced(n, exterior);
  else
  {
    switch (simplex_method)
    {
    case lattice::simplex_method::warp:
      return create_tet_warped(n, lattice_type, exterior);
    case lattice::simplex_method::isaac:
      return create_tet_isaac(n, lattice_type, exterior);
    case lattice::simplex_method::centroid:
      return create_tet_centroid(n, lattice_type, exterior);
    case lattice::simplex_method::none:
    {
      // Methods will all agree when n <= 3
      if (n <= 3)
        return create_tet_warped(n, lattice_type, exterior);
      else
      {
        throw std::runtime_error(
            "A simplex type must be given to create points on a triangle.");
      }
    }
    default:
      throw std::runtime_error("Unrecognised simplex type.");
    }
  }
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 2>>
create_prism(std::size_t n, lattice::type lattice_type, bool exterior,
             lattice::simplex_method simplex_method)
{
  if (n == 0)
    return {{1.0 / 3.0, 1.0 / 3.0, 0.5}, {1, 3}};
  else
  {
    const auto [xx, trishape]
        = create_tri_new(n, lattice_type, exterior, simplex_method);
    stdex::mdspan<const double,
                  stdex::extents<std::size_t, stdex::dynamic_extent, 2>>
        tri_pts(xx.data(), trishape);

    const std::vector<double> line_pts
        = create_interval_new(n, lattice_type, exterior);

    std::array<std::size_t, 2> shape = {tri_pts.extent(0) * line_pts.size(), 3};
    std::vector<double> xb(shape[0] * shape[1]);
    stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, 3>>
        x(xb.data(), shape);

    for (std::size_t i = 0; i < line_pts.size(); ++i)
      for (std::size_t j = 0; j < tri_pts.extent(0); ++j)
        for (std::size_t k = 0; k < 2; ++k)
          x(i * tri_pts.extent(0) + j, k) = tri_pts(j, k);

    for (std::size_t i = 0; i < line_pts.size(); ++i)
    {
      for (std::size_t j = i * tri_pts.extent(0);
           j < (i + 1) * tri_pts.extent(0); ++j)
      {
        x(j, 2) = line_pts[i];
      }
    }

    return {xb, shape};
  }
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 2>>
create_pyramid_equispaced(int n, bool exterior)
{
  const double h = 1.0 / static_cast<double>(n);
  const std::size_t b = (exterior == false) ? 1 : 0;
  n -= b * 3;
  std::size_t m = (n + 1) * (n + 2) * (2 * n + 3) / 6;

  std::array<std::size_t, 2> shape = {m, 3};
  std::vector<double> xb(shape[0] * shape[1]);
  stdex::mdspan<double, stdex::extents<std::size_t, stdex::dynamic_extent, 3>>
      x(xb.data(), shape);
  int c = 0;
  for (int k = 0; k < n + 1; ++k)
  {
    for (int j = 0; j < n + 1 - k; ++j)
    {
      for (int i = 0; i < n + 1 - k; ++i)
      {
        x(c, 0) = h * (i + b);
        x(c, 1) = h * (j + b);
        x(c, 2) = h * (k + b);
        c++;
      }
    }
  }

  return {xb, shape};
}
//-----------------------------------------------------------------------------
/*
xt::xtensor<double, 2> create_pyramid_gll_warped(int n, bool exterior)
{
  // FIXME
  throw std::runtime_error("GLL on Pyramid is not currently working.");

  const double h = 1.0 / static_cast<double>(n);

  // Interpolate warp factor along interval
  std::vector<double> pts = quadrature::get_gll_points(n + 1);
  std::transform(pts.begin(), pts.end(), pts.begin(),
                 [](auto x) { return 0.5 * x; });
  for (int i = 0; i < n + 1; ++i)
    pts[i] += (0.5 - static_cast<double>(i) / static_cast<double>(n));

  // Get interpolated value at r in range [-1, 1]
  auto w = [&](double r) -> double
  {
    xt::xtensor<double, 2> rr = {{0.5 * (r + 1.0)}};
    xt::xtensor<double, 1> v
        = xt::view(tabulate_dlagrange(n, rr), xt::all(), 0);
    double d = 0.0;
    for (std::size_t i = 0; i < pts.size(); ++i)
      d += v[i] * pts[i];
    return d;
  };

  const std::size_t b = (exterior == false) ? 1 : 0;
  n -= b * 3;
  std::size_t m = (n + 1) * (n + 2) * (2 * n + 3) / 6;
  xt::xtensor<double, 2> points({m, 3});
  int c = 0;
  for (int k = 0; k < n + 1; ++k)
  {
    for (int j = 0; j < n + 1 - k; ++j)
    {
      for (int i = 0; i < n + 1 - k; ++i)
      {
        double x = h * (i + b);
        double y = h * (j + b);
        double z = h * (k + b);

        // Barycentric coordinates of triangle in x-z plane
        const double l1 = x;
        const double l2 = z;
        const double l3 = 1 - x - z;

        // Barycentric coordinates of triangle in y-z plane
        const double l4 = y;
        const double l5 = z;
        const double l6 = 1 - y - z;

        // b1-b6 are the blending factors for each edge
        double b1, f1, f2;
        if (std::fabs(l1) < 1e-12)
        {
          b1 = 1.0;
          f1 = 0.0;
          f2 = 0.0;
        }
        else
        {
          b1 = 2.0 * l3 / (2.0 * l3 + l1) * 2.0 * l2 / (2.0 * l2 + l1);
          f1 = l1 / (l1 + l4);
          f2 = l1 / (l1 + l6);
        }

        // r1-r4 are the edge positions for each of the z>0 edges
        // calculated so that they use the barycentric coordinates of
        // the triangle, if the point lies on a triangular face. f1-f4
        // are face selecting functions, which blend between adjacent
        // triangular faces
        const double r1 = (l2 - l3) * f1 + (l5 - l6) * (1 - f1);
        const double r2 = (l2 - l3) * f2 + (l5 - l4) * (1 - f2);

        double b2;
        if (std::fabs(l2) < 1e-12)
          b2 = 1.0;
        else
          b2 = 2.0 * l3 / (2.0 * l3 + l2) * 2.0 * l1 / (2.0 * l1 + l2);

        double b3, f3, f4;
        if (std::fabs(l3) < 1e-12)
        {
          b3 = 1.0;
          f3 = 0.0;
          f4 = 0.0;
        }
        else
        {
          b3 = 2.0 * l2 / (2.0 * l2 + l3) * 2.0 * l1 / (2.0 * l1 + l3);
          f3 = l3 / (l3 + l4);
          f4 = l3 / (l3 + l6);
        }

        const double r3 = (l2 - l1) * f3 + (l5 - l6) * (1.0 - f3);
        const double r4 = (l2 - l1) * f4 + (l5 - l4) * (1.0 - f4);

        double b4;
        if (std::fabs(l4) < 1e-12)
          b4 = 1.0;
        else
          b4 = 2 * l6 / (2.0 * l6 + l4) * 2.0 * l5 / (2.0 * l5 + l4);

        double b5;
        if (std::fabs(l5) < 1e-12)
          b5 = 1.0;
        else
          b5 = 2.0 * l6 / (2.0 * l6 + l5) * 2.0 * l4 / (2.0 * l4 + l5);

        double b6;
        if (std::fabs(l6) < 1e-12)
          b6 = 1.0;
        else
          b6 = 2.0 * l4 / (2.0 * l4 + l6) * 2.0 * l5 / (2.0 * l5 + l6);

        double dx = -b3 * b4 * w(r3) - b3 * b6 * w(r4) + b2 * w(l1 - l3);
        double dy = -b1 * b6 * w(r2) - b3 * b6 * w(r4) + b5 * w(l4 - l6);
        double dz = b1 * b4 * w(r1) + b1 * b6 * w(r2) + b3 * b4 * w(r3)
                    + b3 * b6 * w(r4);
        x += dx;
        y += dy;
        z += dz;

        points(c, 0) = x;
        points(c, 1) = y;
        points(c, 2) = z;
        c++;
      }
    }
  }

  return points;
}
*/
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 2>>
create_pyramid(int n, lattice::type lattice_type, bool exterior,
               lattice::simplex_method /*simplex_method*/)
{
  if (n == 0)
    return {{0.4, 0.4, 0.2}, {1, 3}};
  else if (lattice_type == lattice::type::equispaced)
    return create_pyramid_equispaced(n, exterior);
  else
  {
    throw std::runtime_error(
        "Non-equispaced points on pyramids not supported yet.");
  }

  // if (lattice_type == lattice::type::gll
  //     and simplex_method == lattice::simplex_method::warp)
  //   return create_pyramid_gll_warped(n, exterior);
}
} // namespace
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> lattice::create(cell::type celltype, int n,
                                       lattice::type type, bool exterior,
                                       lattice::simplex_method simplex_method)
{
  // auto [x, shape] = create_new(celltype, n, type, exterior, simplex_method);
  // return xt::adapt(x, std::vector<std::size_t>{x.size(), 1});

  switch (celltype)
  {
  case cell::type::point:
    return {{0.0}};
  case cell::type::interval:
  {
    auto x = create_interval_new(n, type, exterior);
    return xt::adapt(x, std::vector<std::size_t>{x.size(), 1});
  }
  case cell::type::triangle:
  {
    auto [x, shape] = create_tri_new(n, type, exterior, simplex_method);
    return xt::adapt(x, std::vector<std::size_t>{shape[0], shape[1]});
  }
  case cell::type::tetrahedron:
  {
    auto [x, shape] = create_tet(n, type, exterior, simplex_method);
    return xt::adapt(x, std::vector<std::size_t>{shape[0], shape[1]});
  }
  case cell::type::quadrilateral:
  {
    auto [x, shape] = create_quad(n, type, exterior);
    return xt::adapt(x, std::vector<std::size_t>{shape[0], shape[1]});
  }
  case cell::type::hexahedron:
  {
    auto [x, shape] = create_hex(n, type, exterior);
    return xt::adapt(x, std::vector<std::size_t>{shape[0], shape[1]});
  }
  case cell::type::prism:
  {
    auto [x, shape] = create_prism(n, type, exterior, simplex_method);
    return xt::adapt(x, std::vector<std::size_t>{shape[0], shape[1]});
  }
  case cell::type::pyramid:
  {
    auto [x, shape] = create_pyramid(n, type, exterior, simplex_method);
    return xt::adapt(x, std::vector<std::size_t>{shape[0], shape[1]});
  }
  default:
    throw std::runtime_error("Unsupported cell for lattice");
  }
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 2>>
lattice::create_new(cell::type celltype, int n, lattice::type type,
                    bool exterior, lattice::simplex_method simplex_method)
{
  switch (celltype)
  {
  case cell::type::point:
    return {{0.0}, {1, 1}};
  case cell::type::interval:
  {
    auto x = create_interval_new(n, type, exterior);
    return {x, {x.size(), 1}};
  }
  case cell::type::triangle:
    return create_tri_new(n, type, exterior, simplex_method);
  case cell::type::tetrahedron:
    return create_tet(n, type, exterior, simplex_method);
  case cell::type::quadrilateral:
    return create_quad(n, type, exterior);
  case cell::type::hexahedron:
    return create_hex(n, type, exterior);
  case cell::type::prism:
    return create_prism(n, type, exterior, simplex_method);
  case cell::type::pyramid:
    return create_pyramid(n, type, exterior, simplex_method);
  default:
    throw std::runtime_error("Unsupported cell for lattice");

    // xt::xtensor<double, 2> x
    //     = create(celltype, n, type, exterior, simplex_method);
    // return {std::vector<double>(x.data(), x.data() + x.size()),
    //         {
    //             x.shape(0),
    //             x.shape(1),
    //         }};
  }
}
//-----------------------------------------------------------------------------
