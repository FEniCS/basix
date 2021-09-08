// Copyright (c) 2020 Chris Richardson and Garth N. Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "lattice.h"
#include "cell.h"
#include "polyset.h"
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
xt::xtensor<double, 2> tabulate_dlagrange(int n,
                                          const xt::xtensor<double, 1>& x)
{
  std::array<std::size_t, 1> s = {static_cast<std::size_t>(n + 1)};
  xt::xtensor<double, 1> equi_pts(s);
  for (int i = 0; i <= n; ++i)
    equi_pts(i) = static_cast<double>(i) / static_cast<double>(n);

  xt::xtensor<double, 3> dual_values
      = polyset::tabulate(cell::type::interval, n, 0, equi_pts);
  xt::xtensor<double, 2, xt::layout_type::column_major> dualmat(
      {dual_values.shape(2), dual_values.shape(1)});
  dualmat.assign(xt::transpose(xt::view(dual_values, 0, xt::all(), xt::all())));

  xt::xtensor<double, 3> tabulated_values
      = polyset::tabulate(cell::type::interval, n, 0, x);
  xt::xtensor<double, 2, xt::layout_type::column_major> tabulated(
      {tabulated_values.shape(2), tabulated_values.shape(1)});
  tabulated.assign(
      xt::transpose(xt::view(tabulated_values, 0, xt::all(), xt::all())));

  return xt::transpose(xt::linalg::solve(dualmat, tabulated));
}
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

  xt::xtensor<double, 2> v = tabulate_dlagrange(n, x);

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

  if (x.shape(0) > 0
      and (lattice_type == lattice::type::gll
           or lattice_type == lattice::type::gll_isaac))
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

  if (r.shape(0) > 0 and lattice_type == lattice::type::gll)
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
  if (r.shape(0) > 0 and lattice_type == lattice::type::gll)
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
xt::xtensor<double, 2> create_tri_gll_warped(int n, bool exterior)
{
  // Warp points: see Hesthaven and Warburton, Nodal Discontinuous
  // Galerkin Methods, pp. 175-180
  const std::size_t b = exterior ? 0 : 1;

  // Points
  xt::xtensor<double, 2> p({(n - 3 * b + 1) * (n - 3 * b + 2) / 2, 2});

  // Displacement from GLL points in 1D, scaled by 1 /(r * (1 - r))
  xt::xtensor<double, 1> r = xt::linspace<double>(0.0, 1.0, 2 * n + 1);
  xt::xtensor<double, 1> wbar = warp_function(n, r);
  auto s = xt::view(r, xt::range(1, 2 * n - 1));
  xt::view(wbar, xt::range(1, 2 * n - 1)) /= (s * (1 - s));

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
      p(c, 0) += x * (a * wbar(n + i - l) + y * wbar(n + i - j));
      p(c, 1) += y * (a * wbar(n + j - l) + x * wbar(n + j - i));
      ++c;
    }
    }
    return p;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 1> isaac_point(xt::xtensor<std::size_t, 1> a)
{
  if (a.shape(0) == 1)
    return {1};
  xt::xtensor<double, 1> res = xt::zeros<double>(a.shape());
  double denominator = 0;
  xt::xtensor<std::size_t, 1> sub_a
      = xt::view(a, xt::range(1, xt::placeholders::_));
  const std::size_t size = xt::sum(a)();
  xt::xtensor<double, 1> x = create_interval(size, lattice::type::gll, true);
  for (std::size_t i = 0; i < a.shape(0); ++i)
  {
    if (i > 0)
      sub_a(i - 1) = a(i - 1);
    const std::size_t sub_size = size - a(i);
    const xt::xtensor<double, 1> sub_res = isaac_point(sub_a);
    for (std::size_t j = 0; j < sub_res.shape(0); ++j)
      res[j < i ? j : j + 1] += x[sub_size] * sub_res[j];
    denominator += x[sub_size];
  }
  for (std::size_t i = 0; i < res.shape(0); ++i)
    res[i] /= denominator;
  return res;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_tri_isaac(int n, bool exterior)
{
  // See http://dx.doi.org/10.1137/20M1321802
  const std::size_t b = exterior ? 0 : 1;

  // Points
  xt::xtensor<double, 2> p({(n - 3 * b + 1) * (n - 3 * b + 2) / 2, 2});

  int c = 0;
  for (std::size_t j = b; j < (n - b + 1); ++j)
  {
    for (std::size_t i = b; i < (n - b + 1 - j); ++i)
    {
      xt::view(p, c, xt::all())
          = xt::view(isaac_point({i, j, n - i - j}), xt::range(0, 2));
      ++c;
    }
  }
  return p;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_tri(int n, lattice::type lattice_type,
                                  bool exterior)
{
  if (n == 0)
    return {{1.0 / 3.0, 1.0 / 3.0}};

  switch (lattice_type)
  {
  case lattice::type::equispaced:
    return create_tri_equispaced(n, exterior);
  case lattice::type::gll:
    return create_tri_gll_warped(n, exterior);
  case lattice::type::gll_isaac:
    return create_tri_isaac(n, exterior);
  default:
    throw std::runtime_error("Unrecognised lattice type.");
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
xt::xtensor<double, 2> create_tet_isaac(int n, bool exterior)
{
  // See http://dx.doi.org/10.1137/20M1321802
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
            = xt::view(isaac_point({i, j, k, n - i - j - k}), xt::range(0, 3));
        ++c;
      }
    }
  }
  return p;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_tet_gll_warped(int n, bool exterior)
{
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
xt::xtensor<double, 2> create_tet(int n, lattice::type lattice_type,
                                  bool exterior)
{
  if (n == 0)
    return {{0.25, 0.25, 0.25}};

  switch (lattice_type)
  {
  case lattice::type::equispaced:
    return create_tet_equispaced(n, exterior);
  case lattice::type::gll:
    return create_tet_gll_warped(n, exterior);
  case lattice::type::gll_isaac:
    return create_tet_isaac(n, exterior);
  default:
    throw std::runtime_error("Unrecognised lattice type.");
  }
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
  xt::view(x, xt::all(), xt::range(0, 2)).assign(xt::tile(tri_pts, reps));
  for (std::size_t i = 0; i < line_pts.shape(0); ++i)
  {
    auto rows = xt::range(i * tri_pts.shape(0), (i + 1) * tri_pts.shape(0));
    xt::view(x, rows, 2) = line_pts(i);
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
  auto w = [&](double r) -> double
  {
    xt::xtensor<double, 1> rr = {0.5 * (r + 1.0)};
    xt::xtensor<double, 1> v
        = xt::view(tabulate_dlagrange(n, rr), 0, xt::all());
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

        if (lattice_type == lattice::type::gll)
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
lattice::type lattice::str_to_type(std::string name)
{
  static const std::map<std::string, lattice::type> name_to_type
      = {{"equispaced", lattice::type::equispaced},
         {"gll", lattice::type::gll},
         {"gll_isaac", lattice::type::gll_isaac}};

  auto it = name_to_type.find(name);
  if (it == name_to_type.end())
    throw std::runtime_error("Can't find name " + name);

  return it->second;
}
//-----------------------------------------------------------------------------
std::string lattice::type_to_str(lattice::type type)
{
  static const std::map<lattice::type, std::string> name_to_type
      = {{lattice::type::equispaced, "equispaced"},
         {lattice::type::gll, "gll"},
         {lattice::type::gll_isaac, "gll_isaac"}};

  auto it = name_to_type.find(type);
  if (it == name_to_type.end())
    throw std::runtime_error("Can't find type");

  return it->second;
}
//-----------------------------------------------------------------------------
