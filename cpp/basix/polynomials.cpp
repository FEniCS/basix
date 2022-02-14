// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "polynomials.h"
#include "polyset.h"
#include <xtensor/xview.hpp>

using namespace basix;

namespace
{
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> chebyshev_interval(int d, const xt::xarray<double>& x)
{
  xt::xtensor<double, 2> out({x.shape(0), static_cast<std::size_t>(polyset::dim(
                                              cell::type::interval, d))});
  for (int n = 0; n <= d; ++n)
    for (std::size_t p = 0; p < x.shape(0); ++p)
      out(p, n) = std::cos(n * std::acos(2 * x(p) - 1));

  return out;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> chebyshev_quad(int d, const xt::xarray<double>& x)
{
  xt::xtensor<double, 2> r0 = chebyshev_interval(d, xt::view(x, xt::all(), 0));
  xt::xtensor<double, 2> r1 = chebyshev_interval(d, xt::view(x, xt::all(), 1));
  xt::xtensor<double, 2> out({x.shape(0), static_cast<std::size_t>(polyset::dim(
                                              cell::type::quadrilateral, d))});
  std::size_t n = 0;
  for (std::size_t i = 0; i < r0.shape(1); ++i)
  {
    for (std::size_t j = 0; j < r1.shape(1); ++j)
    {
      for (std::size_t p = 0; p < x.shape(0); ++p)
        out(p, n) = r0(p, i) * r1(p, j);
      ++n;
    }
  }
  assert(out.shape(1) == n);

  return out;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> chebyshev_hex(int d, const xt::xarray<double>& x)
{
  xt::xtensor<double, 2> r0 = chebyshev_interval(d, xt::view(x, xt::all(), 0));
  xt::xtensor<double, 2> r1 = chebyshev_interval(d, xt::view(x, xt::all(), 1));
  xt::xtensor<double, 2> r2 = chebyshev_interval(d, xt::view(x, xt::all(), 2));
  xt::xtensor<double, 2> out({x.shape(0), static_cast<std::size_t>(polyset::dim(
                                              cell::type::hexahedron, d))});
  std::size_t n = 0;
  for (std::size_t i = 0; i < r0.shape(1); ++i)
  {
    for (std::size_t j = 0; j < r1.shape(1); ++j)
    {
      for (std::size_t k = 0; k < r2.shape(1); ++k)
      {
        for (std::size_t p = 0; p < x.shape(0); ++p)
          out(p, n) = r0(p, i) * r1(p, j) * r2(p, k);
        ++n;
      }
    }
  }
  assert(out.shape(1) == n);

  return out;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> tabulate_chebyshev(cell::type celltype, int d,
                                          const xt::xarray<double>& x)
{
  switch (celltype)
  {
  case cell::type::interval:
    return chebyshev_interval(d, x);
  case cell::type::quadrilateral:
    return chebyshev_quad(d, x);
  case cell::type::hexahedron:
    return chebyshev_hex(d, x);
  default:
    throw std::runtime_error("not implemented yet");
  }
}
//-----------------------------------------------------------------------------
} // namespace
//-----------------------------------------------------------------------------
xt::xtensor<double, 1>
polynomials::tabulate_bubble(cell::type celltype, const xt::xarray<double>& pts)
{
  switch (celltype)
  {
  case cell::type::interval:
  {
    auto x = pts;
    return x * (1 - x);
  }
  case cell::type::triangle:
  {
    auto x = xt::col(pts, 0);
    auto y = xt::col(pts, 1);
    return x * y * (1 - x - y);
  }
  case cell::type::quadrilateral:
  {
    auto x = xt::col(pts, 0);
    auto y = xt::col(pts, 1);
    return x * (1 - x) * y * (1 - y);
  }
  case cell::type::tetrahedron:
  {
    auto x = xt::col(pts, 0);
    auto y = xt::col(pts, 1);
    auto z = xt::col(pts, 2);
    return x * y * z * (1 - x - y - z);
  }
  case cell::type::hexahedron:
  {
    auto x = xt::col(pts, 0);
    auto y = xt::col(pts, 1);
    auto z = xt::col(pts, 2);
    return x * (1 - x) * y * (1 - y) * z * (1 - z);
  }
  case cell::type::prism:
  {
    auto x = xt::col(pts, 0);
    auto y = xt::col(pts, 1);
    auto z = xt::col(pts, 2);
    return x * y * z * (1 - z) * (1 - x - y);
  }
  case cell::type::pyramid:
  {
    auto x = xt::col(pts, 0);
    auto y = xt::col(pts, 1);
    auto z = xt::col(pts, 2);
    return x * y * z * (1 - y - z) * (1 - x - z);
  }
  default:
    throw std::runtime_error("Unsupported cell type");
  }
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> polynomials::tabulate(polynomials::type polytype,
                                             cell::type celltype, int d,
                                             const xt::xarray<double>& x)
{
  switch (polytype)
  {
  case polynomials::type::legendre:
    return xt::view(polyset::tabulate(celltype, d, 0, x), 0, xt::all(),
                    xt::all());
  case polynomials::type::chebyshev:
    return tabulate_chebyshev(celltype, d, x);
  default:
    throw std::runtime_error("not implemented yet");
  }
}
//-----------------------------------------------------------------------------
int polynomials::dim(polynomials::type, cell::type cell, int d)
{
  return polyset::dim(cell, d);
}
