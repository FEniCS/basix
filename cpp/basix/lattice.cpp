// Copyright (c) 2020 Chris Richardson, Garth N. Wells, and Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "lattice.h"
#include "cell.h"
#include "math.h"
#include "polyset.h"
#include "quadrature.h"
#include <math.h>
#include <vector>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

namespace
{
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_interval_equispaced(int n, bool exterior)
{
  const double h = exterior ? 0 : 1.0 / static_cast<double>(n);
  const std::size_t num_pts = exterior ? n + 1 : n - 1;
  xt::xtensor<double, 1> pts = xt::linspace<double>(h, 1.0 - h, num_pts);
  return xt::reshape_view(pts, {num_pts, (std::size_t)1});
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_interval_gll(int n, bool exterior)
{
  if (n == 0)
    return {{0.5}};

  const xt::xtensor<double, 2> _pts = quadrature::get_gll_points(n + 1);

  const std::size_t b = exterior ? 0 : 1;
  std::array<std::size_t, 2> s = {static_cast<std::size_t>(n + 1 - 2 * b), 1};
  xt::xtensor<double, 2> x(s);
  if (exterior)
  {
    x(0, 0) = _pts(0, 0);
    x(n, 0) = _pts(1, 0);
  }
  for (std::size_t j = 2; j < static_cast<std::size_t>(n + 1); ++j)
    x(j - 1 - b, 0) = _pts(j, 0);
  return x;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_interval_chebyshev(int n, bool exterior)
{
  if (exterior)
  {
    throw std::runtime_error(
        "Chebyshev points including endpoints are not supported.");
  }

  std::array<std::size_t, 2> s = {static_cast<std::size_t>(n - 1), 1};
  xt::xtensor<double, 2> x(s);

  for (int i = 1; i < n; ++i)
    x(i - 1, 0) = 0.5 - cos((2 * i - 1) * M_PI / (2 * n - 2)) / 2.0;

  return x;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_interval_gl(int n, bool exterior)
{
  if (exterior)
  {
    throw std::runtime_error(
        "GL points including endpoints are not supported.");
  }

  if (n == 0)
    return {{0.5}};
  const xt::xtensor<double, 2> pts = quadrature::get_gl_points(n - 1);
  std::array<std::size_t, 2> s = {static_cast<std::size_t>(n - 1), 1};
  xt::xtensor<double, 2> x(s);

  for (int i = 0; i < n - 1; ++i)
    x(i, 0) = pts(i, 0);

  return x;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_interval_gl_plus_endpoints(int n, bool exterior)
{
  xt::xtensor<double, 2> x_gl = create_interval_gl(n, false);

  if (!exterior)
    return x_gl;

  std::array<std::size_t, 2> s = {static_cast<std::size_t>(n + 1), 1};
  xt::xtensor<double, 2> x(s);

  x(0, 0) = 0.;
  x(n, 0) = 1.;
  for (int i = 0; i < n - 1; ++i)
    x(i + 1, 0) = x_gl(i, 0);

  return x;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_interval_chebyshev_plus_endpoints(int n,
                                                                bool exterior)
{
  xt::xtensor<double, 2> x_cheb = create_interval_chebyshev(n, false);

  if (!exterior)
    return x_cheb;

  std::array<std::size_t, 2> s = {static_cast<std::size_t>(n + 1), 1};
  xt::xtensor<double, 2> x(s);

  x(0, 0) = 0.;
  x(n, 0) = 1.;
  for (int i = 0; i < n - 1; ++i)
    x(i + 1, 0) = x_cheb(i, 0);

  return x;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_interval(int n, lattice::type lattice_type,
                                       bool exterior)
{
  if (n == 0)
    return {{0.5}};

  switch (lattice_type)
  {
  case lattice::type::equispaced:
    return create_interval_equispaced(n, exterior);
  case lattice::type::gll:
    return create_interval_gll(n, exterior);
  case lattice::type::chebyshev:
    return create_interval_chebyshev(n, exterior);
  case lattice::type::gl:
    return create_interval_gl(n, exterior);

  case lattice::type::chebyshev_plus_endpoints:
    return create_interval_chebyshev_plus_endpoints(n, exterior);
  case lattice::type::gl_plus_endpoints:
    return create_interval_gl_plus_endpoints(n, exterior);
  default:
    throw std::runtime_error("Unrecognised lattice type.");
  }
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> tabulate_dlagrange(int n,
                                          const xt::xtensor<double, 2>& x)
{
  std::array<std::size_t, 2> s = {static_cast<std::size_t>(n + 1), 1};
  xt::xtensor<double, 2> equi_pts(s);
  for (int i = 0; i <= n; ++i)
    equi_pts(i, 0) = static_cast<double>(i) / static_cast<double>(n);
  xt::xtensor<double, 3> dual_values
      = polyset::tabulate(cell::type::interval, n, 0, equi_pts);
  xt::xtensor<double, 2> dualmat
      = xt::view(dual_values, 0, xt::all(), xt::all());

  xt::xtensor<double, 3> tabulated_values
      = polyset::tabulate(cell::type::interval, n, 0, x);
  xt::xtensor<double, 2> tabulated
      = xt::view(tabulated_values, 0, xt::all(), xt::all());

  return math::solve(dualmat, tabulated);
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 1> warp_function(lattice::type lattice_type, int n,
                                     const xt::xtensor<double, 2>& x)
{
  xt::xtensor<double, 2> pts = create_interval(n, lattice_type, true);
  for (int i = 0; i < n + 1; ++i)
    pts(i, 0) -= static_cast<double>(i) / static_cast<double>(n);
  const xt::xtensor<double, 2> v = tabulate_dlagrange(n, x);

  xt::xtensor<double, 1> w = xt::zeros<double>({v.shape(1)});
  for (std::size_t i = 0; i < v.shape(0); ++i)
    for (std::size_t j = 0; j < v.shape(1); ++j)
      w[j] += v(i, j) * pts(i, 0);

  return w;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_quad(int n, lattice::type lattice_type,
                                   bool exterior)
{
  if (n == 0)
    return {{0.5, 0.5}};

  xt::xtensor<double, 2> r = create_interval(n, lattice_type, exterior);

  const std::size_t m = r.shape(0);
  xt::xtensor<double, 2> x({m * m, 2});
  std::size_t c = 0;
  for (std::size_t j = 0; j < m; ++j)
  {
    for (std::size_t i = 0; i < m; ++i)
    {
      x(c, 0) = r(i, 0);
      x(c, 1) = r(j, 0);
      c++;
    }
  }

  return x;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_hex(int n, lattice::type lattice_type,
                                  bool exterior)
{
  if (n == 0)
    return {{0.5, 0.5, 0.5}};

  xt::xtensor<double, 2> r = create_interval(n, lattice_type, exterior);

  const std::size_t m = r.size();
  xt::xtensor<double, 2> x({m * m * m, 3});
  int c = 0;
  for (std::size_t k = 0; k < m; ++k)
  {
    for (std::size_t j = 0; j < m; ++j)
    {
      for (std::size_t i = 0; i < m; ++i)
      {
        x(c, 0) = r(i, 0);
        x(c, 1) = r(j, 0);
        x(c, 2) = r(k, 0);
        c++;
      }
    }
  }

  return x;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_tri_equispaced(int n, bool exterior)
{
  const std::size_t b = exterior ? 0 : 1;

  // Points
  xt::xtensor<double, 2> p({(n - 3 * b + 1) * (n - 3 * b + 2) / 2, 2});

  // Displacement from GLL points in 1D, scaled by 1 /(r * (1 - r))
  xt::xtensor<double, 1> r = xt::linspace<double>(0.0, 1.0, 2 * n + 1);

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
  return p;
}
//-----------------------------------------------------------------------------
// Warp points: see Hesthaven and Warburton, Nodal Discontinuous
// Galerkin Methods, pp. 175-180, https://doi.org/10.1007/978-0-387-72067-8_6
xt::xtensor<double, 2> create_tri_warped(int n, lattice::type lattice_type,
                                         bool exterior)
{
  const std::size_t b = exterior ? 0 : 1;

  // Points
  xt::xtensor<double, 2> p({(n - 3 * b + 1) * (n - 3 * b + 2) / 2, 2});

  // Displacement from GLL points in 1D, scaled by 1 /(r * (1 - r))
  xt::xtensor<double, 1> _r = xt::linspace<double>(0.0, 1.0, 2 * n + 1);
  xt::xtensor<double, 2> r = xt::reshape_view(
      _r, {static_cast<std::size_t>(2 * n + 1), (std::size_t)1});

  xt::xtensor<double, 1> wbar = warp_function(lattice_type, n, r);
  auto s = xt::view(_r, xt::range(1, 2 * n - 1));
  xt::view(wbar, xt::range(1, 2 * n - 1)) /= (s * (1 - s));
  int c = 0;
  for (std::size_t j = b; j < (n - b + 1); ++j)
  {
    for (std::size_t i = b; i < (n - b + 1 - j); ++i)
    {
      const double x = _r[2 * i];
      const double y = _r[2 * j];
      p(c, 0) = x;
      p(c, 1) = y;
      const std::size_t l = n - j - i;
      const double a = _r[2 * l];
      p(c, 0) += x * (a * wbar(n + i - l) + y * wbar(n + i - j));
      p(c, 1) += y * (a * wbar(n + j - l) + x * wbar(n + j - i));
      ++c;
    }
  }
  return p;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 1> isaac_point(lattice::type lattice_type,
                                   xt::xtensor<std::size_t, 1> a)
{
  if (a.shape(0) == 1)
    return {1};
  xt::xtensor<double, 1> res = xt::zeros<double>(a.shape());
  double denominator = 0;
  xt::xtensor<std::size_t, 1> sub_a
      = xt::view(a, xt::range(1, xt::placeholders::_));
  const std::size_t size = xt::sum(a)();
  xt::xtensor<double, 2> x = create_interval(size, lattice_type, true);
  for (std::size_t i = 0; i < a.shape(0); ++i)
  {
    if (i > 0)
      sub_a(i - 1) = a(i - 1);
    const std::size_t sub_size = size - a(i);
    const xt::xtensor<double, 1> sub_res = isaac_point(lattice_type, sub_a);
    for (std::size_t j = 0; j < sub_res.shape(0); ++j)
      res[j < i ? j : j + 1] += x(sub_size, 0) * sub_res[j];
    denominator += x(sub_size, 0);
  }
  for (std::size_t i = 0; i < res.shape(0); ++i)
    res[i] /= denominator;
  return res;
}
//-----------------------------------------------------------------------------
// Warp points, See: Isaac, Recursive, Parameter-Free, Explicitly Defined
// Interpolation Nodes for Simplices, http://dx.doi.org/10.1137/20M1321802
xt::xtensor<double, 2> create_tri_isaac(int n, lattice::type lattice_type,
                                        bool exterior)
{
  const std::size_t b = exterior ? 0 : 1;

  // Points
  xt::xtensor<double, 2> p({(n - 3 * b + 1) * (n - 3 * b + 2) / 2, 2});

  int c = 0;
  for (std::size_t j = b; j < (n - b + 1); ++j)
  {
    for (std::size_t i = b; i < (n - b + 1 - j); ++i)
    {
      xt::view(p, c, xt::all()) = xt::view(
          isaac_point(lattice_type, {i, j, n - i - j}), xt::range(0, 2));
      ++c;
    }
  }
  return p;
}
//-----------------------------------------------------------------------------
// See: Blyth, and Pozrikidis, A Lobatto interpolation grid over the triangle,
// https://dx.doi.org/10.1093/imamat/hxh077
xt::xtensor<double, 2> create_tri_centroid(int n, lattice::type lattice_type,
                                           bool exterior)
{
  if (exterior)
    throw std::runtime_error(
        "Centroid method not implemented to include boundaries");

  // Points
  xt::xtensor<double, 2> p(
      {static_cast<std::size_t>((n - 2) * (n - 1) / 2), 2});
  xt::xtensor<double, 2> x = create_interval(n, lattice_type, false);

  int c = 0;

  for (std::size_t i = 0; i + 1 < x.shape(0); ++i)
  {
    const double xi = x(i, 0);
    for (std::size_t j = 0; j + i + 1 < x.shape(0); ++j)
    {
      const double xj = x(j, 0);
      const double xk = x(i + j + 1, 0);
      p(c, 0) = (2 * xj + xk - xi) / 3;
      p(c, 1) = (2 * xi + xk - xj) / 3;
      ++c;
    }
  }
  return p;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_tri(int n, lattice::type lattice_type,
                                  bool exterior,
                                  lattice::simplex_method simplex_method)
{
  if (n == 0)
    return {{1.0 / 3.0, 1.0 / 3.0}};

  if (lattice_type == lattice::type::equispaced)
    return create_tri_equispaced(n, exterior);

  switch (simplex_method)
  {
  case lattice::simplex_method::warp:
    return create_tri_warped(n, lattice_type, exterior);
  case lattice::simplex_method::isaac:
    return create_tri_isaac(n, lattice_type, exterior);
  case lattice::simplex_method::centroid:
    return create_tri_centroid(n, lattice_type, exterior);
  case lattice::simplex_method::none:
  {
    // Methods will all agree when n <= 3
    if (n <= 3)
      return create_tri_warped(n, lattice_type, exterior);
    throw std::runtime_error(
        "A simplex type must be given to create points on a triangle.");
  }
  default:
    throw std::runtime_error("Unrecognised simplex type.");
  }
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_tet_equispaced(int n, bool exterior)
{
  const std::size_t b = exterior ? 0 : 1;
  xt::xtensor<double, 2> p(
      {(n - 4 * b + 1) * (n - 4 * b + 2) * (n - 4 * b + 3) / 6, 3});
  auto r = xt::linspace<double>(0.0, 1.0, 2 * n + 1);

  std::size_t c = 0;
  for (std::size_t k = b; k < (n - b + 1); ++k)
  {
    for (std::size_t j = b; j < (n - b + 1 - k); ++j)
    {
      for (std::size_t i = b; i < (n - b + 1 - j - k); ++i)
      {
        p(c, 0) = r[2 * i];
        p(c, 1) = r[2 * j];
        p(c, 2) = r[2 * k];
        ++c;
      }
    }
  }

  return p;
}
//-----------------------------------------------------------------------------
// See: Isaac, Recursive, Parameter-Free, Explicitly Defined Interpolation Nodes
// for Simplices http://dx.doi.org/10.1137/20M1321802
xt::xtensor<double, 2> create_tet_isaac(int n, lattice::type lattice_type,
                                        bool exterior)
{
  const std::size_t b = exterior ? 0 : 1;

  // Points
  xt::xtensor<double, 2> p(
      {(n - 4 * b + 1) * (n - 4 * b + 2) * (n - 4 * b + 3) / 6, 3});

  int c = 0;
  for (std::size_t k = b; k < (n - b + 1); ++k)
  {
    for (std::size_t j = b; j < (n - b + 1 - k); ++j)
    {
      for (std::size_t i = b; i < (n - b + 1 - j - k); ++i)
      {
        xt::view(p, c, xt::all())
            = xt::view(isaac_point(lattice_type, {i, j, k, n - i - j - k}),
                       xt::range(0, 3));
        ++c;
      }
    }
  }
  return p;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_tet_warped(int n, lattice::type lattice_type,
                                         bool exterior)
{
  const std::size_t b = exterior ? 0 : 1;
  xt::xtensor<double, 2> p(
      {(n - 4 * b + 1) * (n - 4 * b + 2) * (n - 4 * b + 3) / 6, 3});
  xt::xtensor<double, 1> _r = xt::linspace<double>(0.0, 1.0, 2 * n + 1);
  xt::xtensor<double, 2> r = xt::reshape_view(
      _r, {static_cast<std::size_t>(2 * n + 1), (std::size_t)1});
  auto wbar = warp_function(lattice_type, n, r);
  auto s = xt::view(r, xt::range(1, 2 * n - 1), 0);
  xt::view(wbar, xt::range(1, 2 * n - 1)) /= s * (1 - s);

  std::size_t c = 0;
  for (std::size_t k = b; k < (n - b + 1); ++k)
  {
    for (std::size_t j = b; j < (n - b + 1 - k); ++j)
    {
      for (std::size_t i = b; i < (n - b + 1 - j - k); ++i)
      {
        const std::size_t l = n - k - j - i;
        const double x = r(2 * i, 0);
        const double y = r(2 * j, 0);
        const double z = r(2 * k, 0);
        const double a = r(2 * l, 0);
        p(c, 0) = x;
        p(c, 1) = y;
        p(c, 2) = z;
        const double dx = x
                          * (a * wbar(n + i - l) + y * wbar(n + i - j)
                             + z * wbar(n + i - k));
        const double dy = y
                          * (a * wbar(n + j - l) + z * wbar(n + j - k)
                             + x * wbar(n + j - i));
        const double dz = z
                          * (a * wbar(n + k - l) + x * wbar(n + k - i)
                             + y * wbar(n + k - j));
        p(c, 0) += dx;
        p(c, 1) += dy;
        p(c, 2) += dz;

        ++c;
      }
    }
  }

  return p;
}
//-----------------------------------------------------------------------------
// See: Blyth, and Pozrikidis, A Lobatto interpolation grid over the triangle,
// https://dx.doi.org/10.1093/imamat/hxh077
xt::xtensor<double, 2> create_tet_centroid(int n, lattice::type lattice_type,
                                           bool exterior)
{
  if (exterior)
    throw std::runtime_error(
        "Centroid method not implemented to include boundaries");

  // Points
  xt::xtensor<double, 2> p(
      {static_cast<std::size_t>((n - 3) * (n - 2) * (n - 1) / 6), 3});
  xt::xtensor<double, 2> x = create_interval(n, lattice_type, false);

  int c = 0;

  for (std::size_t i = 0; i + 2 < x.shape(0); ++i)
  {
    const double xi = x(i, 0);
    for (std::size_t j = 0; j + i + 2 < x.shape(0); ++j)
    {
      const double xj = x(j, 0);
      for (std::size_t k = 0; k + j + i + 2 < x.shape(0); ++k)
      {
        const double xk = x(k, 0);
        const double xl = x(i + j + k + 2, 0);
        p(c, 0) = (3 * xk + xl - xi - xj) / 4;
        p(c, 1) = (3 * xj + xl - xi - xk) / 4;
        p(c, 2) = (3 * xi + xl - xj - xk) / 4;
        ++c;
      }
    }
  }
  return p;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_tet(int n, lattice::type lattice_type,
                                  bool exterior,
                                  lattice::simplex_method simplex_method)
{
  if (n == 0)
    return {{0.25, 0.25, 0.25}};

  if (lattice_type == lattice::type::equispaced)
    return create_tet_equispaced(n, exterior);

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
    throw std::runtime_error(
        "A simplex type must be given to create points on a triangle.");
  }
  default:
    throw std::runtime_error("Unrecognised simplex type.");
  }
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_prism(int n, lattice::type lattice_type,
                                    bool exterior,
                                    lattice::simplex_method simplex_method)
{
  if (n == 0)
    return {{1.0 / 3.0, 1.0 / 3.0, 0.5}};

  const xt::xtensor<double, 2> tri_pts
      = create_tri(n, lattice_type, exterior, simplex_method);
  const xt::xtensor<double, 2> line_pts
      = create_interval(n, lattice_type, exterior);

  xt::xtensor<double, 2> x({tri_pts.shape(0) * line_pts.shape(0), 3});
  std::array<std::size_t, 2> reps = line_pts.shape();
  xt::view(x, xt::all(), xt::range(0, 2)).assign(xt::tile(tri_pts, reps));
  for (std::size_t i = 0; i < line_pts.shape(0); ++i)
  {
    auto rows = xt::range(i * tri_pts.shape(0), (i + 1) * tri_pts.shape(0));
    xt::view(x, rows, 2) = line_pts(i, 0);
  }
  return x;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_pyramid_equispaced(int n, bool exterior)
{
  const double h = 1.0 / static_cast<double>(n);

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
        points(c, 0) = h * (i + b);
        points(c, 1) = h * (j + b);
        points(c, 2) = h * (k + b);
        c++;
      }
    }
  }

  return points;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_pyramid_gll_warped(int n, bool exterior)
{
  // FIXME
  throw std::runtime_error("GLL on Pyramid is not currently working.");

  const double h = 1.0 / static_cast<double>(n);

  // Interpolate warp factor along interval
  xt::xtensor<double, 2> pts = quadrature::get_gll_points(n + 1);
  pts *= 0.5;
  for (int i = 0; i < n + 1; ++i)
    pts(i, 0) += (0.5 - static_cast<double>(i) / static_cast<double>(n));

  // Get interpolated value at r in range [-1, 1]
  auto w = [&](double r) -> double
  {
    xt::xtensor<double, 2> rr = {{0.5 * (r + 1.0)}};
    xt::xtensor<double, 1> v
        = xt::view(tabulate_dlagrange(n, rr), xt::all(), 0);
    double d = 0.0;
    for (std::size_t i = 0; i < pts.shape(0); ++i)
      d += v[i] * pts(i, 0);
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
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_pyramid(int n, lattice::type lattice_type,
                                      bool exterior,
                                      lattice::simplex_method simplex_method)
{
  if (n == 0)
    return {{0.4, 0.4, 0.2}};

  if (lattice_type == lattice::type::equispaced)
    return create_pyramid_equispaced(n, exterior);

  throw std::runtime_error(
      "Non-equispaced points on pyramids not supported yet.");

  if (lattice_type == lattice::type::gll
      and simplex_method == lattice::simplex_method::warp)
    return create_pyramid_gll_warped(n, exterior);
}
} // namespace
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> lattice::create(cell::type celltype, int n,
                                       lattice::type type, bool exterior,
                                       lattice::simplex_method simplex_method)
{
  switch (celltype)
  {
  case cell::type::point:
    return {{0.0}};
  case cell::type::interval:
    return create_interval(n, type, exterior);
  case cell::type::triangle:
    return create_tri(n, type, exterior, simplex_method);
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
  }
}
//-----------------------------------------------------------------------------
std::vector<double> lattice::create_new(cell::type celltype, int n,
                                        lattice::type type, bool exterior,
                                        lattice::simplex_method simplex_method)
{
  xt::xtensor<double, 2> x
      = create(celltype, n, type, exterior, simplex_method);
  std::vector<double> data(x.data(), x.data() + x.size());
}
//-----------------------------------------------------------------------------
