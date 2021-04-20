// Copyright (c) 2020 Chris Richardson and Garth N. Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "lattice.h"
#include "cell.h"
#include "e-lagrange.h"
#include "quadrature.h"
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

namespace
{
//-----------------------------------------------------------------------------
xt::xtensor<double, 1> warp_function(int n, const xt::xtensor<double, 1>& x)
{
  [[maybe_unused]] auto [_pts, wts] = quadrature::compute_gll_rule(n + 1);
  _pts *= 0.5;
  for (int i = 0; i < n + 1; ++i)
    _pts[i] += (0.5 - static_cast<double>(i) / static_cast<double>(n));
  std::array<std::size_t, 1> shape0 = {(std::size_t)_pts.size()};
  xt::xtensor<double, 1> pts
      = xt::adapt(_pts.data(), _pts.size(), xt::no_ownership(), shape0);

  FiniteElement L
      = create_dlagrange(cell::type::interval, n, element::variant::EQ);
  xt::xtensor<double, 2> v
      = xt::view(L.tabulate(0, x), 0, xt::all(), xt::all(), 0);

  return xt::linalg::dot(v, pts);
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 1> create_interval(int n, lattice::type lattice_type,
                                       bool exterior)
{
  if (n == 0)
    return {0.5};

  xt::xtensor<double, 1> x;
  if (exterior)
    x = xt::linspace<double>(0.0, 1.0, n + 1);
  else
  {
    const double h = 1.0 / static_cast<double>(n);
    x = xt::linspace<double>(h, 1.0 - h, n - 1);
  }

  if (x.shape(0) > 0 and lattice_type == lattice::type::gll_warped)
    x += warp_function(n, x);

  return x;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_quad(int n, lattice::type lattice_type,
                                   bool exterior)
{
  if (n == 0)
    return {{0.5, 0.5}};

  xt::xtensor<double, 1> r;
  if (exterior)
    r = xt::linspace<double>(0.0, 1.0, n + 1);
  else
  {
    const double h = 1.0 / static_cast<double>(n);
    r = xt::linspace<double>(h, 1.0 - h, n - 1);
  }

  if (r.shape(0) > 0 and lattice_type == lattice::type::gll_warped)
    r += warp_function(n, r);

  const std::size_t m = r.shape(0);
  xt::xtensor<double, 2> x({m * m, 2});
  std::size_t c = 0;
  for (std::size_t j = 0; j < m; ++j)
  {
    for (std::size_t i = 0; i < m; ++i)
    {
      x(c, 0) = r(i);
      x(c, 1) = r(j);
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

  xt::xtensor<double, 1> r;
  if (exterior)
    r = xt::linspace<double>(0.0, 1.0, n + 1);
  else
  {
    const double h = 1.0 / static_cast<double>(n);
    r = xt::linspace<double>(h, 1.0 - h, n - 1);
  }
  if (r.shape(0) > 0 and lattice_type == lattice::type::gll_warped)
    r += warp_function(n, r);

  const std::size_t m = r.size();
  xt::xtensor<double, 2> x({m * m * m, 3});
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

  return x;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_tri(int n, lattice::type lattice_type,
                                  bool exterior)
{
  if (n == 0)
    return {{1.0 / 3.0, 1.0 / 3.0}};

  // Warp points: see Hesthaven and Warburton, Nodal Discontinuous
  // Galerkin Methods, pp. 175-180

  const std::size_t b = exterior ? 0 : 1;

  // Displacement from GLL points in 1D, scaled by 1 /(r * (1 - r))
  xt::xtensor<double, 1> r = xt::linspace<double>(0.0, 1.0, 2 * n + 1);
  xt::xtensor<double, 1> wbar = warp_function(n, r);
  auto s = xt::view(r, xt::range(1, 2 * n - 1));
  xt::view(wbar, xt::range(1, 2 * n - 1)) /= (s * (1 - s));

  // Points
  xt::xtensor<double, 2> p({(n - 3 * b + 1) * (n - 3 * b + 2) / 2, 2});
  int c = 0;
  for (std::size_t j = b; j < (n - b + 1); ++j)
  {
    for (std::size_t i = b; i < (n - b + 1 - j); ++i)
    {
      const double x = r[2 * i];
      const double y = r[2 * j];
      p(c, 0) = x;
      p(c, 1) = y;
      if (lattice_type == lattice::type::gll_warped)
      {
        const std::size_t l = n - j - i;
        const double a = r[2 * l];
        p(c, 0) += x * (a * wbar(n + i - l) + y * wbar(n + i - j));
        p(c, 1) += y * (a * wbar(n + j - l) + x * wbar(n + j - i));
      }
      ++c;
    }
  }

  return p;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_tet(int n, lattice::type lattice_type,
                                  bool exterior)
{
  if (n == 0)
    return {{0.25, 0.25, 0.25}};

  const std::size_t b = exterior ? 0 : 1;
  xt::xtensor<double, 2> p(
      {(n - 4 * b + 1) * (n - 4 * b + 2) * (n - 4 * b + 3) / 6, 3});
  auto r = xt::linspace<double>(0.0, 1.0, 2 * n + 1);
  auto wbar = warp_function(n, r);
  auto s = xt::view(r, xt::range(1, 2 * n - 1));
  xt::view(wbar, xt::range(1, 2 * n - 1)) /= s * (1 - s);

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
        if (lattice_type == lattice::type::gll_warped)
        {
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
        }

        ++c;
      }
    }
  }

  return p;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_prism(int n, lattice::type lattice_type,
                                    bool exterior)
{
  if (n == 0)
    return {{1.0 / 3.0, 1.0 / 3.0, 0.5}};

  const xt::xtensor<double, 2> tri_pts = create_tri(n, lattice_type, exterior);
  const xt::xtensor<double, 1> line_pts
      = create_interval(n, lattice_type, exterior);

  xt::xtensor<double, 2> x({tri_pts.shape(0) * line_pts.shape(0), 3});
  std::array<std::size_t, 2> reps = {line_pts.shape(0), 1};
  xt::view(x, xt::all(), xt::range(0, 2)) = xt::tile(tri_pts, reps);
  for (std::size_t i = 0; i < line_pts.shape(0); ++i)
  {
    auto rows = xt::range(i * tri_pts.shape(0), (i + 1) * tri_pts.shape(0));
    xt::view(x, rows, xt::range(2, 3)) = line_pts(i);
  }

  return x;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_pyramid(int n, lattice::type lattice_type,
                                      bool exterior)
{
  if (n == 0)
    return {{0.4, 0.4, 0.2}};

  const double h = 1.0 / static_cast<double>(n);

  // Interpolate warp factor along interval
  std::pair<xt::xarray<double>, std::vector<double>> pw
      = quadrature::compute_gll_rule(n + 1);
  xt::xtensor<double, 1> pts = std::get<0>(pw);
  pts *= 0.5;
  for (int i = 0; i < n + 1; ++i)
    pts[i] += (0.5 - static_cast<double>(i) / static_cast<double>(n));

  // Get interpolated value at r in range [-1, 1]
  FiniteElement L
      = create_dlagrange(cell::type::interval, n, element::variant::EQ);
  auto w = [&](double r) -> double {
    xt::xtensor<double, 1> rr = {0.5 * (r + 1.0)};
    xt::xtensor<double, 1> v = xt::view(L.tabulate(0, rr), 0, 0, xt::all(), 0);
    double d = 0.0;
    for (std::size_t i = 0; i < pts.shape(0); ++i)
      d += v[i] * pts[i];
    return d;
    // return v.dot(pts);
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

        if (lattice_type == lattice::type::gll_warped)
        {
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
        }

        points(c, 0) = x;
        points(c, 1) = y;
        points(c, 2) = z;
        c++;
      }
    }
  }

  return points;
}
} // namespace
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> lattice::create(cell::type celltype, int n,
                                       lattice::type type, bool exterior)
{
  switch (celltype)
  {
  case cell::type::point:
    return {{0.0}};
  case cell::type::interval:
  {
    xt::xtensor<double, 1> x = create_interval(n, type, exterior);
    std::array<std::size_t, 2> s = {x.shape(0), 1};
    return xt::reshape_view(x, s);
  }
  case cell::type::triangle:
    return create_tri(n, type, exterior);
  case cell::type::tetrahedron:
    return create_tet(n, type, exterior);
  case cell::type::quadrilateral:
    return create_quad(n, type, exterior);
  case cell::type::hexahedron:
    return create_hex(n, type, exterior);
  case cell::type::prism:
    return create_prism(n, type, exterior);
  case cell::type::pyramid:
    return create_pyramid(n, type, exterior);
  default:
    throw std::runtime_error("Unsupported cell for lattice");
  }
}
//-----------------------------------------------------------------------------
