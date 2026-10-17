// Copyright (c) 2020 Chris Richardson, Garth N. Wells, and Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "lattice.h"
#include "cell.h"
#include "math.h"
#include "polyset.h"
#include "quadrature.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <math.h>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace basix;

namespace
{
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::vector<T> linspace(T x0, T x1, std::size_t n)
{
  if (n == 0)
    return {};
  else if (n == 1)
    return {x0};
  else
  {

    std::vector<T> p(n, x0);
    p.back() = x1;
    const T delta = (x1 - x0) / (n - 1);
    for (std::size_t i = 1; i < p.size() - 1; ++i)
      p[i] += i * delta;
    return p;
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::vector<T> create_interval_equispaced(std::size_t n, bool exterior)
{
  const T h = exterior ? 0 : 1.0 / static_cast<T>(n);
  const std::size_t num_pts = exterior ? n + 1 : n - 1;
  return linspace<T>(h, 1.0 - h, num_pts);
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::vector<T> create_interval_gll(std::size_t n, bool exterior)
{
  if (n == 0)
    return {0.5};
  else
  {
    const std::vector<T> pts = quadrature::get_gll_points<T>(n + 1);
    const std::size_t b = exterior ? 0 : 1;
    std::vector<T> x(n + 1 - 2 * b);
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
template <std::floating_point T>
std::vector<T> create_interval_chebyshev(std::size_t n, bool exterior)
{
  if (exterior)
  {
    throw std::runtime_error(
        "Chebyshev points including endpoints are not supported.");
  }

  std::vector<T> x(n - 1);
  for (std::size_t i = 1; i < n; ++i)
    x[i - 1] = 0.5 - std::cos((2 * i - 1) * M_PI / (2 * n - 2)) / 2.0;

  return x;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::vector<T> create_interval_gl(std::size_t n, bool exterior)
{
  if (exterior)
  {
    throw std::runtime_error(
        "GL points including endpoints are not supported.");
  }

  if (n == 0)
    return {0.5};
  else
    return quadrature::get_gl_points<T>(n - 1);
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::vector<T> create_interval_gl_plus_endpoints(std::size_t n, bool exterior)
{
  std::vector<T> x_gl = create_interval_gl<T>(n, false);
  if (!exterior)
    return x_gl;
  else
  {
    std::vector<T> x(n + 1);
    x[0] = 0.0;
    x[n] = 1.0;
    for (std::size_t i = 0; i < n - 1; ++i)
      x[i + 1] = x_gl[i];

    return x;
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::vector<T> create_interval_chebyshev_plus_endpoints(std::size_t n,
                                                        bool exterior)
{
  std::vector<T> x_cheb = create_interval_chebyshev<T>(n, false);
  if (!exterior)
    return x_cheb;
  else
  {
    std::vector<T> x(n + 1);
    x[0] = 0.0;
    x[n] = 1.0;
    for (std::size_t i = 0; i < n - 1; ++i)
      x[i + 1] = x_cheb[i];
    return x;
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::vector<T> create_interval(std::size_t n, lattice::type lattice_type,
                               bool exterior)
{
  if (n == 0)
    return {0.5};
  else
  {
    switch (lattice_type)
    {
    case lattice::type::equispaced:
      return create_interval_equispaced<T>(n, exterior);
    case lattice::type::gll:
      return create_interval_gll<T>(n, exterior);
    case lattice::type::chebyshev:
      return create_interval_chebyshev<T>(n, exterior);
    case lattice::type::gl:
      return create_interval_gl<T>(n, exterior);
    case lattice::type::chebyshev_plus_endpoints:
      return create_interval_chebyshev_plus_endpoints<T>(n, exterior);
    case lattice::type::gl_plus_endpoints:
      return create_interval_gl_plus_endpoints<T>(n, exterior);
    default:
      throw std::runtime_error("Unrecognised lattice type.");
    }
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
mdex::mdarray<T, md::dextents<std::size_t, 2>>
tabulate_dlagrange(std::size_t n, std::span<const T> x)
{
  std::vector<T> equi_pts(n + 1);
  for (std::size_t i = 0; i < equi_pts.size(); ++i)
    equi_pts[i] = static_cast<T>(i) / static_cast<T>(n);
  md::mdspan<const T, md::dextents<std::size_t, 2>> equi_pts_v(equi_pts.data(),
                                                               n + 1, 1);

  // polyset::tabulate with nderivs = 0 always returns a leading extent of
  // 1, so the buffer is already exactly the (ndofs, npts) matrix -- no
  // need to copy it into a separate buffer just to drop that extent.
  const auto [dual_values_b, dshape] = polyset::tabulate<T>(
      cell::type::interval, polyset::type::standard, n, 0, equi_pts_v);
  assert(dshape[0] == 1);
  md::mdspan<const T, md::dextents<std::size_t, 2>> dualmat(
      dual_values_b.data(), dshape[1], dshape[2]);

  using cmdspan2_t
      = md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 1>>;
  const auto [tabulated_values_b, tshape]
      = polyset::tabulate<T>(cell::type::interval, polyset::type::standard, n,
                             0, cmdspan2_t(x.data(), x.size(), 1));
  assert(tshape[0] == 1);
  md::mdspan<const T, md::dextents<std::size_t, 2>> tabulated(
      tabulated_values_b.data(), tshape[1], tshape[2]);

  std::vector<T> c = math::solve<T>(dualmat, tabulated);
  return mdex::mdarray<T, md::dextents<std::size_t, 2>>(tabulated.extents(),
                                                        std::move(c));
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::vector<T> warp_function(lattice::type lattice_type, int n,
                             std::span<const T> x)
{
  std::vector<T> pts = create_interval<T>(n, lattice_type, true);
  for (int i = 0; i < n + 1; ++i)
    pts[i] -= static_cast<T>(i) / static_cast<T>(n);

  mdex::mdarray<T, md::dextents<std::size_t, 2>> v = tabulate_dlagrange(n, x);
  std::vector<T> w(v.extent(1), 0);
  for (std::size_t i = 0; i < v.extent(0); ++i)
    for (std::size_t j = 0; j < v.extent(1); ++j)
      w[j] += v(i, j) * pts[i];

  return w;
}
//-----------------------------------------------------------------------------

/// Evaluate the warp function at points `x` in [0, 1], shifted so that
/// the displacement is exactly zero at the two endpoints.
///
/// The warp is interpolated through a Lagrange basis, so it vanishes at
/// the endpoints only up to rounding. Where a blending factor damps the
/// displacement at a vertex, as in create_tri_warped, that error is
/// harmless, but the pyramid blending factors below all degenerate to
/// one at the apex, so the residuals accumulate there and would move
/// the apex off the cell by a few epsilon. The polyset on a pyramid is
/// expressed in the collapsed coordinates `x / (1 - z)` and `y / (1 -
/// z)`, so such a point tabulates to nonsense.
template <std::floating_point T>
std::vector<T> warp_function_zero_ends(lattice::type lattice_type, int n,
                                       std::span<const T> x)
{
  // Evaluate at the endpoints alongside x, then subtract the linear
  // interpolant of the two endpoint residuals
  std::vector<T> xe(x.begin(), x.end());
  xe.push_back(0.0);
  xe.push_back(1.0);
  std::vector<T> w = warp_function<T>(lattice_type, n, xe);
  const T w0 = w[x.size()];
  const T w1 = w[x.size() + 1];
  w.resize(x.size());
  for (std::size_t i = 0; i < w.size(); ++i)
    w[i] -= (1.0 - x[i]) * w0 + x[i] * w1;

  return w;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
create_quad(std::size_t n, lattice::type lattice_type, bool exterior)
{
  if (n == 0)
    return {{0.5, 0.5}, {1, 2}};
  else
  {
    const std::vector<T> r = create_interval<T>(n, lattice_type, exterior);
    const std::size_t m = r.size();
    std::array<std::size_t, 2> shape = {m * m, 2};
    std::vector<T> xb(shape[0] * shape[1]);
    md::mdspan<T, md::extents<std::size_t, md::dynamic_extent, 2>> x(xb.data(),
                                                                     shape);
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

    return {std::move(xb), shape};
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
create_hex(int n, lattice::type lattice_type, bool exterior)
{
  if (n == 0)
    return {{0.5, 0.5, 0.5}, {1, 3}};
  else
  {
    const std::vector<T> r = create_interval<T>(n, lattice_type, exterior);
    const std::size_t m = r.size();

    std::array<std::size_t, 2> shape = {m * m * m, 3};

    std::vector<T> xb(shape[0] * shape[1]);
    md::mdspan<T, md::extents<std::size_t, md::dynamic_extent,
                              md::dynamic_extent, md::dynamic_extent, 3>>
        x(xb.data(), m, m, m, 3);

    for (std::size_t k = 0; k < m; ++k)
    {
      for (std::size_t j = 0; j < m; ++j)
      {
        for (std::size_t i = 0; i < m; ++i)
        {
          x(k, j, i, 0) = r[i];
          x(k, j, i, 1) = r[j];
          x(k, j, i, 2) = r[k];
        }
      }
    }

    return {std::move(xb), shape};
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
create_tri_equispaced(std::size_t n, bool exterior)
{
  const std::size_t b = exterior ? 0 : 1;

  std::array<std::size_t, 2> shape = {(n - 3 * b + 1) * (n - 3 * b + 2) / 2, 2};
  std::vector<T> _p(shape[0] * shape[1]);
  md::mdspan<T, md::extents<std::size_t, md::dynamic_extent, 2>> p(_p.data(),
                                                                   shape);

  // Displacement from GLL points in 1D, scaled by 1 /(r * (1 - r))
  std::vector<T> r = linspace<T>(0.0, 1.0, 2 * n + 1);
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

  return {std::move(_p), shape};
}
//-----------------------------------------------------------------------------

/// Warp points: see Hesthaven and Warburton, Nodal Discontinuous
/// Galerkin Methods, pp. 175-180,
/// https://doi.org/10.1007/978-0-387-72067-8_6
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
create_tri_warped(std::size_t n, lattice::type lattice_type, bool exterior)
{
  const std::size_t b = exterior ? 0 : 1;

  // Points
  std::array<std::size_t, 2> shape = {(n - 3 * b + 1) * (n - 3 * b + 2) / 2, 2};
  std::vector<T> _p(shape[0] * shape[1]);
  md::mdspan<T, md::extents<std::size_t, md::dynamic_extent, 2>> p(_p.data(),
                                                                   shape);

  // Displacement from GLL points in 1D, scaled by 1 /(r * (1 - r))
  const std::vector<T> r = linspace<T>(0.0, 1.0, 2 * n + 1);

  std::vector<T> wbar = warp_function<T>(lattice_type, n, r);
  for (std::size_t i = 1; i < 2 * n - 1; ++i)
    wbar[i] /= r[i] * (1.0 - r[i]);

  int c = 0;
  for (std::size_t j = b; j < (n - b + 1); ++j)
  {
    for (std::size_t i = b; i < (n - b + 1 - j); ++i)
    {
      const T x = r[2 * i];
      const T y = r[2 * j];
      p(c, 0) = x;
      p(c, 1) = y;
      const std::size_t l = n - j - i;
      const T a = r[2 * l];
      p(c, 0) += x * (a * wbar[n + i - l] + y * wbar[n + i - j]);
      p(c, 1) += y * (a * wbar[n + j - l] + x * wbar[n + j - i]);
      ++c;
    }
  }

  return {std::move(_p), shape};
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::vector<T> isaac_point(lattice::type lattice_type,
                           std::span<const std::size_t> a,
                           const std::vector<std::vector<T>>& interval_pts)
{
  if (a.size() == 1)
    return {1};
  else
  {
    std::vector<T> res(a.size(), 0);
    T denominator = 0;
    std::vector<std::size_t> sub_a(std::next(a.begin()), a.end());
    const std::size_t size = std::reduce(a.begin(), a.end());
    // interval_pts[size] is create_interval<T>(size, lattice_type, true),
    // precomputed once by the caller (for sizes 0..n) rather than
    // recomputed on every call to this recursion, which runs once per
    // lattice point.
    const std::vector<T>& x = interval_pts[size];
    for (std::size_t i = 0; i < a.size(); ++i)
    {
      if (i > 0)
        sub_a[i - 1] = a[i - 1];
      const std::size_t sub_size = size - a[i];
      const std::vector sub_res
          = isaac_point<T>(lattice_type, sub_a, interval_pts);
      for (std::size_t j = 0; j < sub_res.size(); ++j)
        res[j < i ? j : j + 1] += x[sub_size] * sub_res[j];
      denominator += x[sub_size];
    }

    std::ranges::for_each(res, [denominator](auto& x) { x /= denominator; });

    return res;
  }
}
//-----------------------------------------------------------------------------

/// Warp points, See: Isaac, Recursive, Parameter-Free, Explicitly
/// Defined Interpolation Nodes for Simplices,
/// http://dx.doi.org/10.1137/20M1321802.
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
create_tri_isaac(std::size_t n, lattice::type lattice_type, bool exterior)
{
  const std::size_t b = exterior ? 0 : 1;
  std::array<std::size_t, 2> shape = {(n - 3 * b + 1) * (n - 3 * b + 2) / 2, 2};
  std::vector<T> _p(shape[0] * shape[1]);
  md::mdspan<T, md::extents<std::size_t, md::dynamic_extent, 2>> p(_p.data(),
                                                                   shape);

  // Return before paying for the interval_pts table below (each entry is
  // a GLL/Gauss eigenvalue solve) when there are no points to generate,
  // e.g. a low-degree Lagrange element's interior lattice.
  if (shape[0] == 0)
    return {std::move(_p), shape};

  // create_interval<T>(s, lattice_type, true) only depends on s, so build
  // the table once for s in [0, n] instead of recomputing it for every
  // one of the O(n^2) lattice points below.
  std::vector<std::vector<T>> interval_pts(n + 1);
  for (std::size_t s = 0; s <= n; ++s)
    interval_pts[s] = create_interval<T>(s, lattice_type, true);

  int c = 0;
  for (std::size_t j = b; j < (n - b + 1); ++j)
  {
    for (std::size_t i = b; i < (n - b + 1 - j); ++i)
    {
      const std::vector isaac_p = isaac_point<T>(
          lattice_type, std::array{i, j, n - i - j}, interval_pts);
      for (std::size_t k = 0; k < 2; ++k)
        p(c, k) = isaac_p[k];
      ++c;
    }
  }

  return {std::move(_p), shape};
}
//-----------------------------------------------------------------------------

/// See: Blyth, and Pozrikidis, A Lobatto interpolation grid over the
/// triangle, https://dx.doi.org/10.1093/imamat/hxh077
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
create_tri_centroid(std::size_t n, lattice::type lattice_type, bool exterior)
{
  if (exterior)
  {
    throw std::runtime_error(
        "Centroid method not implemented to include boundaries");
  }

  std::array<std::size_t, 2> shape = {(n - 2) * (n - 1) / 2, 2};
  std::vector<T> _p(shape[0] * shape[1]);
  md::mdspan<T, md::extents<std::size_t, md::dynamic_extent, 2>> p(_p.data(),
                                                                   shape);

  const std::vector<T> x = create_interval<T>(n, lattice_type, false);
  int c = 0;
  for (std::size_t i = 0; i + 1 < x.size(); ++i)
  {
    const T xi = x[i];
    for (std::size_t j = 0; j + i + 1 < x.size(); ++j)
    {
      const T xj = x[j];
      const T xk = x[i + j + 1];
      p(c, 0) = (2 * xj + xk - xi) / 3;
      p(c, 1) = (2 * xi + xk - xj) / 3;
      ++c;
    }
  }

  return {std::move(_p), shape};
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
create_tri(std::size_t n, lattice::type lattice_type, bool exterior,
           lattice::simplex_method simplex_method)
{
  if (n == 0)
    return {{1.0 / 3.0, 1.0 / 3.0}, {1, 2}};
  else if (lattice_type == lattice::type::equispaced)
    return create_tri_equispaced<T>(n, exterior);
  else
  {
    switch (simplex_method)
    {
    case lattice::simplex_method::warp:
      return create_tri_warped<T>(n, lattice_type, exterior);
    case lattice::simplex_method::isaac:
      return create_tri_isaac<T>(n, lattice_type, exterior);
    case lattice::simplex_method::centroid:
      return create_tri_centroid<T>(n, lattice_type, exterior);
    case lattice::simplex_method::none:
    {
      // Methods will all agree when n <= 3
      if (n <= 3)
        return create_tri_warped<T>(n, lattice_type, exterior);
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
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
create_tet_equispaced(std::size_t n, bool exterior)
{
  const std::size_t b = exterior ? 0 : 1;
  const std::vector<T> r = linspace<T>(0.0, 1.0, 2 * n + 1);

  std::array<std::size_t, 2> shape
      = {(n - 4 * b + 1) * (n - 4 * b + 2) * (n - 4 * b + 3) / 6, 3};
  std::vector<T> xb(shape[0] * shape[1]);
  md::mdspan<T, md::extents<std::size_t, md::dynamic_extent, 3>> x(xb.data(),
                                                                   shape);

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

  return {std::move(xb), shape};
}
//-----------------------------------------------------------------------------

/// See: Isaac, Recursive, Parameter-Free, Explicitly Defined Interpolation
/// Nodes for Simplices http://dx.doi.org/10.1137/20M1321802
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
create_tet_isaac(std::size_t n, lattice::type lattice_type, bool exterior)
{
  const std::size_t b = exterior ? 0 : 1;

  std::array<std::size_t, 2> shape
      = {(n - 4 * b + 1) * (n - 4 * b + 2) * (n - 4 * b + 3) / 6, 3};
  std::vector<T> xb(shape[0] * shape[1]);
  md::mdspan<T, md::extents<std::size_t, md::dynamic_extent, 3>> x(xb.data(),
                                                                   shape);

  // See the comment in create_tri_isaac.
  if (shape[0] == 0)
    return {std::move(xb), shape};

  // See the comment in create_tri_isaac.
  std::vector<std::vector<T>> interval_pts(n + 1);
  for (std::size_t s = 0; s <= n; ++s)
    interval_pts[s] = create_interval<T>(s, lattice_type, true);

  int c = 0;
  for (std::size_t k = b; k < (n - b + 1); ++k)
  {
    for (std::size_t j = b; j < (n - b + 1 - k); ++j)
    {
      for (std::size_t i = b; i < (n - b + 1 - j - k); ++i)
      {
        const std::vector ip = isaac_point<T>(
            lattice_type, std::array{i, j, k, n - i - j - k}, interval_pts);
        for (std::size_t l = 0; l < 3; ++l)
          x(c, l) = ip[l];
        ++c;
      }
    }
  }

  return {std::move(xb), shape};
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
create_tet_warped(std::size_t n, lattice::type lattice_type, bool exterior)
{
  const std::size_t b = exterior ? 0 : 1;
  const std::vector<T> r = linspace<T>(0.0, 1.0, 2 * n + 1);
  std::vector<T> wbar = warp_function<T>(lattice_type, n, r);

  std::transform(std::next(r.begin()), std::prev(r.end()),
                 std::next(wbar.begin()), std::next(wbar.begin()),
                 [](auto r, auto w)
                 { return static_cast<T>(w / (r * (1.0 - r))); });

  std::array<std::size_t, 2> shape
      = {(n - 4 * b + 1) * (n - 4 * b + 2) * (n - 4 * b + 3) / 6, 3};
  std::vector<T> xb(shape[0] * shape[1]);
  md::mdspan<T, md::extents<std::size_t, md::dynamic_extent, 3>> p(xb.data(),
                                                                   shape);

  std::size_t c = 0;
  for (std::size_t k = b; k < (n - b + 1); ++k)
  {
    for (std::size_t j = b; j < (n - b + 1 - k); ++j)
    {
      for (std::size_t i = b; i < (n - b + 1 - j - k); ++i)
      {
        const std::size_t l = n - k - j - i;
        const T x = r[2 * i];
        const T y = r[2 * j];
        const T z = r[2 * k];
        const T a = r[2 * l];
        p(c, 0) = x;
        p(c, 1) = y;
        p(c, 2) = z;
        const T dx = x
                     * (a * wbar[n + i - l] + y * wbar[n + i - j]
                        + z * wbar[n + i - k]);
        const T dy = y
                     * (a * wbar[n + j - l] + z * wbar[n + j - k]
                        + x * wbar[n + j - i]);
        const T dz = z
                     * (a * wbar[n + k - l] + x * wbar[n + k - i]
                        + y * wbar[n + k - j]);
        p(c, 0) += dx;
        p(c, 1) += dy;
        p(c, 2) += dz;

        ++c;
      }
    }
  }

  return {std::move(xb), shape};
}
//-----------------------------------------------------------------------------

/// See: Blyth, and Pozrikidis, A Lobatto interpolation grid over the
/// triangle, https://dx.doi.org/10.1093/imamat/hxh077
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
create_tet_centroid(std::size_t n, lattice::type lattice_type, bool exterior)
{
  if (exterior)
  {
    throw std::runtime_error(
        "Centroid method not implemented to include boundaries");
  }

  const std::vector<T> x = create_interval<T>(n, lattice_type, false);

  std::array<std::size_t, 2> shape = {(n - 3) * (n - 2) * (n - 1) / 6, 3};
  std::vector<T> xb(shape[0] * shape[1]);
  md::mdspan<T, md::extents<std::size_t, md::dynamic_extent, 3>> p(xb.data(),
                                                                   shape);

  int c = 0;
  for (std::size_t i = 0; i + 2 < x.size(); ++i)
  {
    const T xi = x[i];
    for (std::size_t j = 0; j + i + 2 < x.size(); ++j)
    {
      const T xj = x[j];
      for (std::size_t k = 0; k + j + i + 2 < x.size(); ++k)
      {
        const T xk = x[k];
        const T xl = x[i + j + k + 2];
        p(c, 0) = (3 * xk + xl - xi - xj) / 4;
        p(c, 1) = (3 * xj + xl - xi - xk) / 4;
        p(c, 2) = (3 * xi + xl - xj - xk) / 4;
        ++c;
      }
    }
  }

  return {std::move(xb), shape};
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
create_tet(std::size_t n, lattice::type lattice_type, bool exterior,
           lattice::simplex_method simplex_method)
{
  if (n == 0)
    return {{0.25, 0.25, 0.25}, {1, 3}};
  else if (lattice_type == lattice::type::equispaced)
    return create_tet_equispaced<T>(n, exterior);
  else
  {
    switch (simplex_method)
    {
    case lattice::simplex_method::warp:
      return create_tet_warped<T>(n, lattice_type, exterior);
    case lattice::simplex_method::isaac:
      return create_tet_isaac<T>(n, lattice_type, exterior);
    case lattice::simplex_method::centroid:
      return create_tet_centroid<T>(n, lattice_type, exterior);
    case lattice::simplex_method::none:
    {
      // Methods will all agree when n <= 3
      if (n <= 3)
        return create_tet_warped<T>(n, lattice_type, exterior);
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
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
create_prism(std::size_t n, lattice::type lattice_type, bool exterior,
             lattice::simplex_method simplex_method)
{
  using cmdspan22_t
      = md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 2>>;
  using mdspan23_t
      = md::mdspan<T, md::extents<std::size_t, md::dynamic_extent, 3>>;

  if (n == 0)
    return {{1.0 / 3.0, 1.0 / 3.0, 0.5}, {1, 3}};
  else
  {
    const auto [tri_pts_b, trishape]
        = create_tri<T>(n, lattice_type, exterior, simplex_method);
    cmdspan22_t tri_pts(tri_pts_b.data(), trishape);

    const std::vector line_pts = create_interval<T>(n, lattice_type, exterior);
    std::array<std::size_t, 2> shape = {tri_pts.extent(0) * line_pts.size(), 3};
    std::vector<T> xb(shape[0] * shape[1]);
    mdspan23_t x(xb.data(), shape);

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

    return {std::move(xb), shape};
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
create_pyramid_equispaced(int n, bool exterior)
{
  const T h = 1.0 / static_cast<T>(n);
  const std::size_t b = (exterior == false) ? 1 : 0;
  n -= b * 3;
  const std::size_t m = (n + 1) * (n + 2) * (2 * n + 3) / 6;

  std::array<std::size_t, 2> shape = {m, 3};
  std::vector<T> xb(shape[0] * shape[1]);
  md::mdspan<T, md::extents<std::size_t, md::dynamic_extent, 3>> x(xb.data(),
                                                                   shape);
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

  return {std::move(xb), shape};
}
//-----------------------------------------------------------------------------

/// Weight applied to the warp along the edge of a triangle on which the
/// barycentric coordinate `l2` vanishes, in terms of the other two
/// barycentric coordinates. It is one on that edge and decays away from
/// it. The coordinates are formed from lattice indices by the caller,
/// so comparing `l2` against zero is exact, and the denominators are
/// bounded below by `l2` elsewhere.
template <std::floating_point T>
constexpr T blending_factor(T l0, T l1, T l2)
{
  if (l2 == 0.0)
    return 1.0;
  else
    return 4.0 * l0 * l1 / ((2.0 * l0 + l2) * (2.0 * l1 + l2));
}
//-----------------------------------------------------------------------------

/// Warp points: see Hesthaven and Warburton, Nodal Discontinuous
/// Galerkin Methods, pp. 175-180,
/// https://doi.org/10.1007/978-0-387-72067-8_6
///
/// Each point is displaced by a blend of the one-dimensional warps
/// along the eight edges of the pyramid. The blending factors are
/// chosen so that the points on the quadrilateral base are those of
/// create_quad and the points on each triangular face are those of
/// create_tri_warped, which is what allows the lattice to be used on a
/// mesh containing pyramids alongside hexahedra and tetrahedra.
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
create_pyramid_warped(int n, lattice::type lattice_type, bool exterior)
{
  const T h = 1.0 / static_cast<T>(n);
  const int b = exterior ? 0 : 1;
  const int m = n - 3 * b;

  std::array<std::size_t, 2> shape
      = {static_cast<std::size_t>((m + 1) * (m + 2) * (2 * m + 3) / 6), 3};
  std::vector<T> xb(shape[0] * shape[1]);
  md::mdspan<T, md::extents<std::size_t, md::dynamic_extent, 3>> x(xb.data(),
                                                                   shape);

  // The displacement of each point is a weighted sum of six warp
  // values: four for the edges that meet at the apex, and two for the
  // directions of the base. The arguments of the warp function and the
  // weights are computed for all points first, so that the warp
  // function (which solves a dense system) is evaluated once.
  std::vector<T> rb(shape[0] * 6), wtb(shape[0] * 6);
  md::mdspan<T, md::extents<std::size_t, md::dynamic_extent, 6>> r(rb.data(),
                                                                   shape[0], 6);
  md::mdspan<T, md::extents<std::size_t, md::dynamic_extent, 6>> wt(
      wtb.data(), shape[0], 6);

  std::size_t c = 0;
  for (int k = 0; k < m + 1; ++k)
  {
    for (int j = 0; j < m + 1 - k; ++j)
    {
      for (int i = 0; i < m + 1 - k; ++i, ++c)
      {
        // Indices of the point in the degree n lattice
        const int p = i + b, q = j + b, s = k + b;
        x(c, 0) = h * p;
        x(c, 1) = h * q;
        x(c, 2) = h * s;

        // Barycentric coordinates of the triangle in the x-z plane
        // (l0, l1, l2) and of the triangle in the y-z plane (l3, l4,
        // l5), so l1 and l4 are both z. They are formed from the
        // lattice indices so that a coordinate that should vanish is
        // exactly zero.
        const T l0 = h * p, l1 = h * s, l2 = h * (n - p - s);
        const T l3 = h * q, l4 = h * s, l5 = h * (n - q - s);

        // Blending factor for each of the six edges of the two
        // triangles
        const T b0 = blending_factor(l2, l1, l0);
        const T b1 = blending_factor(l2, l0, l1);
        const T b2 = blending_factor(l1, l0, l2);
        const T b3 = blending_factor(l5, l4, l3);
        const T b4 = blending_factor(l5, l3, l4);
        const T b5 = blending_factor(l3, l4, l5);

        // Face selecting functions, which blend between adjacent
        // triangular faces. They are not needed where the edge that
        // they belong to degenerates, which is also where they are
        // undefined.
        T f0 = 0.0, f1 = 0.0, f2 = 0.0, f3 = 0.0;
        if (l0 != 0.0)
        {
          f0 = l0 / (l0 + l3);
          f1 = l0 / (l0 + l5);
        }

        if (l2 != 0.0)
        {
          f2 = l2 / (l2 + l3);
          f3 = l2 / (l2 + l5);
        }

        // Positions along each of the four edges that meet at the
        // apex, computed so that they use the barycentric coordinates
        // of the triangle if the point lies on a triangular face, and
        // along the two directions of the base. The warp function takes
        // its argument in [0, 1], so the positions are mapped from
        // [-1, 1].
        r(c, 0) = 0.5 * (1.0 + (l1 - l2) * f0 + (l4 - l5) * (1.0 - f0));
        r(c, 1) = 0.5 * (1.0 + (l1 - l2) * f1 + (l4 - l3) * (1.0 - f1));
        r(c, 2) = 0.5 * (1.0 + (l1 - l0) * f2 + (l4 - l5) * (1.0 - f2));
        r(c, 3) = 0.5 * (1.0 + (l1 - l0) * f3 + (l4 - l3) * (1.0 - f3));
        r(c, 4) = 0.5 * (1.0 + l0 - l2);
        r(c, 5) = 0.5 * (1.0 + l3 - l5);

        wt(c, 0) = b0 * b3;
        wt(c, 1) = b0 * b5;
        wt(c, 2) = b2 * b3;
        wt(c, 3) = b2 * b5;
        wt(c, 4) = b1;
        wt(c, 5) = b4;
      }
    }
  }

  const std::vector<T> wb = warp_function_zero_ends<T>(lattice_type, n, rb);
  md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 6>> w(
      wb.data(), shape[0], 6);
  for (std::size_t i = 0; i < shape[0]; ++i)
  {
    const T w0 = wt(i, 0) * w(i, 0), w1 = wt(i, 1) * w(i, 1);
    const T w2 = wt(i, 2) * w(i, 2), w3 = wt(i, 3) * w(i, 3);
    x(i, 0) += -w2 - w3 + wt(i, 4) * w(i, 4);
    x(i, 1) += -w1 - w3 + wt(i, 5) * w(i, 5);
    x(i, 2) += w0 + w1 + w2 + w3;
  }

  return {std::move(xb), shape};
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
create_pyramid(int n, lattice::type lattice_type, bool exterior,
               lattice::simplex_method simplex_method)
{
  if (n == 0)
    return {{0.4, 0.4, 0.2}, {1, 3}};
  else if (n <= 2 || lattice_type == lattice::type::equispaced)
    return create_pyramid_equispaced<T>(n, exterior);
  else
  {
    switch (simplex_method)
    {
    case lattice::simplex_method::warp:
      return create_pyramid_warped<T>(n, lattice_type, exterior);
    case lattice::simplex_method::isaac:
    {
      throw std::runtime_error(
          "Points matching Isaac's points on a triangle are not implemented "
          "on a pyramid yet.");
    }
    case lattice::simplex_method::centroid:
    {
      throw std::runtime_error(
          "The centroid method is not implemented on a pyramid.");
    }
    case lattice::simplex_method::none:
    {
      // Methods will all agree when n <= 3
      if (n <= 3)
        return create_pyramid_warped<T>(n, lattice_type, exterior);
      else
      {
        throw std::runtime_error(
            "A simplex type must be given to create points on a pyramid.");
      }
    }
    default:
      throw std::runtime_error("Unrecognised simplex type.");
    }
  }
}
} // namespace
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
lattice::create(cell::type celltype, int n, lattice::type type, bool exterior,
                lattice::simplex_method simplex_method)
{
  switch (celltype)
  {
  case cell::type::point:
    return {{0.0}, {1, 1}};
  case cell::type::interval:
  {
    auto x = create_interval<T>(n, type, exterior);
    return {x, {x.size(), 1}};
  }
  case cell::type::triangle:
    return create_tri<T>(n, type, exterior, simplex_method);
  case cell::type::tetrahedron:
    return create_tet<T>(n, type, exterior, simplex_method);
  case cell::type::quadrilateral:
    return create_quad<T>(n, type, exterior);
  case cell::type::hexahedron:
    return create_hex<T>(n, type, exterior);
  case cell::type::prism:
    return create_prism<T>(n, type, exterior, simplex_method);
  case cell::type::pyramid:
    return create_pyramid<T>(n, type, exterior, simplex_method);
  default:
    throw std::runtime_error("Unsupported cell for lattice");
  }
}
//-----------------------------------------------------------------------------
/// @cond DOXYGEN_SHOULD_SKIP_THIS
// Explicit instantiation for double and float
template std::pair<std::vector<float>, std::array<std::size_t, 2>>
lattice::create(cell::type, int, lattice::type, bool, lattice::simplex_method);
template std::pair<std::vector<double>, std::array<std::size_t, 2>>
lattice::create(cell::type, int, lattice::type, bool, lattice::simplex_method);
/// @endcond
//-----------------------------------------------------------------------------
