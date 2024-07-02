// Copyright (c) 2020 Chris Richardson, Garth N. Wells, and Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "lattice.h"
#include "cell.h"
#include "math.h"
#include "polyset.h"
#include "quadrature.h"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <math.h>
#include <numeric>
#include <vector>

using namespace basix;
namespace stdex
    = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;

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
stdex::mdarray<T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
tabulate_dlagrange(std::size_t n, std::span<const T> x)
{
  std::vector<T> equi_pts(n + 1);
  for (std::size_t i = 0; i < equi_pts.size(); ++i)
    equi_pts[i] = static_cast<T>(i) / static_cast<T>(n);
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      equi_pts_v(equi_pts.data(), n + 1, 1);

  const auto [dual_values_b, dshape] = polyset::tabulate<T>(
      cell::type::interval, polyset::type::standard, n, 0, equi_pts_v);
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
      dual_values(dual_values_b.data(), dshape);

  std::vector<T> dualmat_b(dual_values.extent(1) * dual_values.extent(2));
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      dualmat(dualmat_b.data(), dual_values.extent(1), dual_values.extent(2));
  for (std::size_t i = 0; i < dualmat.extent(0); ++i)
    for (std::size_t j = 0; j < dualmat.extent(1); ++j)
      dualmat[i, j] = dual_values[0, i, j];

  using cmdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const T,
      MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
          std::size_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent, 1>>;
  const auto [tabulated_values_b, tshape]
      = polyset::tabulate<T>(cell::type::interval, polyset::type::standard, n,
                             0, cmdspan2_t(x.data(), x.size(), 1));
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
      tabulated_values(tabulated_values_b.data(), tshape);

  std::vector<T> tabulated_b(tabulated_values.extent(1)
                             * tabulated_values.extent(2));
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      tabulated(tabulated_b.data(), tabulated_values.extent(1),
                tabulated_values.extent(2));

  for (std::size_t i = 0; i < tabulated.extent(0); ++i)
    for (std::size_t j = 0; j < tabulated.extent(1); ++j)
      tabulated[i, j] = tabulated_values[0, i, j];

  std::vector<T> c = math::solve<T>(dualmat, tabulated);
  return stdex::mdarray<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>(
      tabulated.extents(), std::move(c));
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::vector<T> warp_function(lattice::type lattice_type, int n,
                             std::span<const T> x)
{
  std::vector<T> pts = create_interval<T>(n, lattice_type, true);
  for (int i = 0; i < n + 1; ++i)
    pts[i] -= static_cast<T>(i) / static_cast<T>(n);

  stdex::mdarray<T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>> v
      = tabulate_dlagrange(n, x);
  std::vector<T> w(v.extent(1), 0);
  for (std::size_t i = 0; i < v.extent(0); ++i)
    for (std::size_t j = 0; j < v.extent(1); ++j)
      w[j] += v[i, j] * pts[i];

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
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
               std::size_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent, 2>>
        x(xb.data(), shape);
    std::size_t c = 0;
    for (std::size_t j = 0; j < m; ++j)
    {
      for (std::size_t i = 0; i < m; ++i)
      {
        x[c, 0] = r[i];
        x[c, 1] = r[j];
        c++;
      }
    }

    return {std::move(xb), std::move(shape)};
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
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
               std::size_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent,
               MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent,
               MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent, 3>>
        x(xb.data(), m, m, m, 3);

    for (std::size_t k = 0; k < m; ++k)
    {
      for (std::size_t j = 0; j < m; ++j)
      {
        for (std::size_t i = 0; i < m; ++i)
        {
          x[k, j, i, 0] = r[i];
          x[k, j, i, 1] = r[j];
          x[k, j, i, 2] = r[k];
        }
      }
    }

    return {std::move(xb), std::move(shape)};
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
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
             std::size_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent, 2>>
      p(_p.data(), shape);

  // Displacement from GLL points in 1D, scaled by 1 /(r * (1 - r))
  std::vector<T> r = linspace<T>(0.0, 1.0, 2 * n + 1);
  int c = 0;
  for (std::size_t j = b; j < (n - b + 1); ++j)
  {
    for (std::size_t i = b; i < (n - b + 1 - j); ++i)
    {
      p[c, 0] = r[2 * i];
      p[c, 1] = r[2 * j];
      ++c;
    }
  }

  return {std::move(_p), std::move(shape)};
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
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
             std::size_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent, 2>>
      p(_p.data(), shape);

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
      p[c, 0] = x;
      p[c, 1] = y;
      const std::size_t l = n - j - i;
      const T a = r[2 * l];
      p[c, 0] += x * (a * wbar[n + i - l] + y * wbar[n + i - j]);
      p[c, 1] += y * (a * wbar[n + j - l] + x * wbar[n + j - i]);
      ++c;
    }
  }

  return {std::move(_p), std::move(shape)};
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::vector<T> isaac_point(lattice::type lattice_type,
                           std::span<const std::size_t> a)
{
  if (a.size() == 1)
    return {1};
  else
  {
    std::vector<T> res(a.size(), 0);
    T denominator = 0;
    std::vector<std::size_t> sub_a(std::next(a.begin()), a.end());
    const std::size_t size = std::reduce(a.begin(), a.end());
    std::vector<T> x = create_interval<T>(size, lattice_type, true);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
      if (i > 0)
        sub_a[i - 1] = a[i - 1];
      const std::size_t sub_size = size - a[i];
      const std::vector sub_res = isaac_point<T>(lattice_type, sub_a);
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
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
             std::size_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent, 2>>
      p(_p.data(), shape);

  int c = 0;
  for (std::size_t j = b; j < (n - b + 1); ++j)
  {
    for (std::size_t i = b; i < (n - b + 1 - j); ++i)
    {
      const std::vector isaac_p
          = isaac_point<T>(lattice_type, std::array{i, j, n - i - j});
      for (std::size_t k = 0; k < 2; ++k)
        p[c, k] = isaac_p[k];
      ++c;
    }
  }

  return {std::move(_p), std::move(shape)};
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
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
             std::size_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent, 2>>
      p(_p.data(), shape);

  const std::vector<T> x = create_interval<T>(n, lattice_type, false);
  int c = 0;
  for (std::size_t i = 0; i + 1 < x.size(); ++i)
  {
    const T xi = x[i];
    for (std::size_t j = 0; j + i + 1 < x.size(); ++j)
    {
      const T xj = x[j];
      const T xk = x[i + j + 1];
      p[c, 0] = (2 * xj + xk - xi) / 3;
      p[c, 1] = (2 * xi + xk - xj) / 3;
      ++c;
    }
  }

  return {std::move(_p), std::move(shape)};
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
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
             std::size_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent, 3>>
      x(xb.data(), shape);

  std::size_t c = 0;
  for (std::size_t k = b; k < (n - b + 1); ++k)
  {
    for (std::size_t j = b; j < (n - b + 1 - k); ++j)
    {
      for (std::size_t i = b; i < (n - b + 1 - j - k); ++i)
      {
        x[c, 0] = r[2 * i];
        x[c, 1] = r[2 * j];
        x[c, 2] = r[2 * k];
        ++c;
      }
    }
  }

  return {std::move(xb), std::move(shape)};
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
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
             std::size_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent, 3>>
      x(xb.data(), shape);

  int c = 0;
  for (std::size_t k = b; k < (n - b + 1); ++k)
  {
    for (std::size_t j = b; j < (n - b + 1 - k); ++j)
    {
      for (std::size_t i = b; i < (n - b + 1 - j - k); ++i)
      {
        const std::vector ip
            = isaac_point<T>(lattice_type, std::array{i, j, k, n - i - j - k});
        for (std::size_t l = 0; l < 3; ++l)
          x[c, l] = ip[l];
        ++c;
      }
    }
  }

  return {std::move(xb), std::move(shape)};
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
                 [](auto r, auto w) { return w / (r * (1.0 - r)); });

  std::array<std::size_t, 2> shape
      = {(n - 4 * b + 1) * (n - 4 * b + 2) * (n - 4 * b + 3) / 6, 3};
  std::vector<T> xb(shape[0] * shape[1]);
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
             std::size_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent, 3>>
      p(xb.data(), shape);

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
        p[c, 0] = x;
        p[c, 1] = y;
        p[c, 2] = z;
        const T dx = x
                     * (a * wbar[n + i - l] + y * wbar[n + i - j]
                        + z * wbar[n + i - k]);
        const T dy = y
                     * (a * wbar[n + j - l] + z * wbar[n + j - k]
                        + x * wbar[n + j - i]);
        const T dz = z
                     * (a * wbar[n + k - l] + x * wbar[n + k - i]
                        + y * wbar[n + k - j]);
        p[c, 0] += dx;
        p[c, 1] += dy;
        p[c, 2] += dz;

        ++c;
      }
    }
  }

  return {std::move(xb), std::move(shape)};
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
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
             std::size_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent, 3>>
      p(xb.data(), shape);

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
        p[c, 0] = (3 * xk + xl - xi - xj) / 4;
        p[c, 1] = (3 * xj + xl - xi - xk) / 4;
        p[c, 2] = (3 * xi + xl - xj - xk) / 4;
        ++c;
      }
    }
  }

  return {std::move(xb), std::move(shape)};
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
  using cmdspan22_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const T,
      MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
          std::size_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent, 2>>;
  using mdspan23_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
             std::size_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent, 3>>;

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
          x[i * tri_pts.extent(0) + j, k] = tri_pts[j, k];

    for (std::size_t i = 0; i < line_pts.size(); ++i)
    {
      for (std::size_t j = i * tri_pts.extent(0);
           j < (i + 1) * tri_pts.extent(0); ++j)
      {
        x[j, 2] = line_pts[i];
      }
    }

    return {std::move(xb), std::move(shape)};
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
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
             std::size_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent, 3>>
      x(xb.data(), shape);
  int c = 0;
  for (int k = 0; k < n + 1; ++k)
  {
    for (int j = 0; j < n + 1 - k; ++j)
    {
      for (int i = 0; i < n + 1 - k; ++i)
      {
        x[c, 0] = h * (i + b);
        x[c, 1] = h * (j + b);
        x[c, 2] = h * (k + b);
        c++;
      }
    }
  }

  return {std::move(xb), std::move(shape)};
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
create_pyramid(int n, lattice::type lattice_type, bool exterior,
               lattice::simplex_method /*simplex_method*/)
{
  if (n == 0)
    return {{0.4, 0.4, 0.2}, {1, 3}};
  else if (n <= 2 || lattice_type == lattice::type::equispaced)
    return create_pyramid_equispaced<T>(n, exterior);
  else
  {
    throw std::runtime_error(
        "Non-equispaced points on pyramids not supported yet.");
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
/// @cond
// Explicit instantiation for double and float
template std::pair<std::vector<float>, std::array<std::size_t, 2>>
lattice::create(cell::type, int, lattice::type, bool, lattice::simplex_method);
template std::pair<std::vector<double>, std::array<std::size_t, 2>>
lattice::create(cell::type, int, lattice::type, bool, lattice::simplex_method);
/// @endcond
//-----------------------------------------------------------------------------
