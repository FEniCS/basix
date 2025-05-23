// Copyright (c) 2020 Chris Richardson and Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "quadrature.h"
#include "math.h"
#include "mdspan.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <numeric>
#include <quadraturerules.h>
#include <span>
#include <vector>

using namespace basix;

template <typename T, std::size_t d>
using mdspan_t = md::mdspan<T, md::dextents<std::size_t, d>>;
template <typename T, std::size_t d>
using mdarray_t = mdex::mdarray<T, md::dextents<std::size_t, d>>;

namespace
{
//----------------------------------------------------------------------------

// Load a rule from the quadraturerules library
template <std::floating_point T>
std::array<std::vector<T>, 2>
get_from_quadraturerules(quadraturerules::QuadratureRule rule,
                         cell::type celltype, int m)
{
  switch (celltype)
  {
  case cell::type::interval:
  {
    auto [_x, _w] = quadraturerules::single_integral_quadrature(
        rule, quadraturerules::Domain::Interval, m);
    std::vector<T> x(_w.size());
    std::vector<T> w(_w.size());
    for (std::size_t i = 0; i < _w.size(); ++i)
    {
      // Convert from barycentric coordinates
      x[i] = _x[2 * i + 1];
      w[i] = _w[i];
    }
    return {std::move(x), std::move(w)};
  }
  case cell::type::quadrilateral:
  {
    auto [_x, _w] = quadraturerules::single_integral_quadrature(
        rule, quadraturerules::Domain::Quadrilateral, m);
    std::vector<T> x(_w.size() * 2);
    std::vector<T> w(_w.size());
    for (std::size_t i = 0; i < _w.size(); ++i)
    {
      // Convert from barycentric coordinates
      x[2 * i] = _x[4 * i + 1];
      x[2 * i + 1] = _x[4 * i + 2];
      w[i] = _w[i];
    }
    return {std::move(x), std::move(w)};
  }
  case cell::type::hexahedron:
  {
    auto [_x, _w] = quadraturerules::single_integral_quadrature(
        rule, quadraturerules::Domain::Quadrilateral, m);
    std::vector<T> x(_w.size() * 3);
    std::vector<T> w(_w.size());
    for (std::size_t i = 0; i < _w.size(); ++i)
    {
      // Convert from barycentric coordinates
      x[3 * i] = _x[8 * i + 1];
      x[3 * i + 1] = _x[8 * i + 2];
      x[3 * i + 2] = _x[8 * i + 4];
      w[i] = _w[i];
    }
    return {std::move(x), std::move(w)};
  }
  case cell::type::triangle:
  {
    auto [_x, _w] = quadraturerules::single_integral_quadrature(
        rule, quadraturerules::Domain::Triangle, m);
    std::vector<T> x(_w.size() * 2);
    std::vector<T> w(_w.size());
    for (std::size_t i = 0; i < _w.size(); ++i)
    {
      // Convert from barycentric coordinates
      x[2 * i] = _x[3 * i + 1];
      x[2 * i + 1] = _x[3 * i + 2];
      // Scale to cell with volume 1/2
      w[i] = _w[i] * 0.5;
    }
    return {std::move(x), std::move(w)};
  }
  case cell::type::tetrahedron:
  {
    auto [_x, _w] = quadraturerules::single_integral_quadrature(
        rule, quadraturerules::Domain::Tetrahedron, m);
    std::vector<T> x(_w.size() * 3);
    std::vector<T> w(_w.size());
    for (std::size_t i = 0; i < _w.size(); ++i)
    {
      // Convert from barycentric coordinates
      x[3 * i] = _x[4 * i + 1];
      x[3 * i + 1] = _x[4 * i + 2];
      x[3 * i + 2] = _x[4 * i + 3];
      // Scale to cell with volume 1/6
      w[i] = _w[i] / 6.0;
    }
    return {std::move(x), std::move(w)};
  }
  case cell::type::prism:
  {
    auto [_x, _w] = quadraturerules::single_integral_quadrature(
        rule, quadraturerules::Domain::Tetrahedron, m);
    std::vector<T> x(_w.size() * 3);
    std::vector<T> w(_w.size());
    for (std::size_t i = 0; i < _w.size(); ++i)
    {
      // Convert from barycentric coordinates
      x[3 * i] = _x[6 * i + 1];
      x[3 * i + 1] = _x[6 * i + 2];
      x[3 * i + 2] = _x[6 * i + 3];
      // Scale to cell with volume 1/2
      w[i] = _w[i] * 0.5;
    }
    return {std::move(x), std::move(w)};
  }
  case cell::type::pyramid:
  {
    auto [_x, _w] = quadraturerules::single_integral_quadrature(
        rule, quadraturerules::Domain::Tetrahedron, m);
    std::vector<T> x(_w.size() * 3);
    std::vector<T> w(_w.size());
    for (std::size_t i = 0; i < _w.size(); ++i)
    {
      // Convert from barycentric coordinates
      x[3 * i] = _x[5 * i + 1];
      x[3 * i + 1] = _x[5 * i + 2];
      x[3 * i + 2] = _x[5 * i + 4];
      // Scale to cell with volume 1/3
      w[i] = _w[i] / 3.0;
    }
    return {std::move(x), std::move(w)};
  }
  default:
  {
    throw std::runtime_error("Invalid cell type.");
  }
  }
}
//-----------------------------------------------------------------------------

/// Generate the recursion coefficients alpha_k, beta_k
///
/// P_{k+1}(x) = (x-alpha_k)*P_{k}(x) - beta_k P_{k-1}(x)
///
/// for the Jacobi polynomials which are orthogonal on [-1,1]
/// with respect to the weight w(x)=[(1-x)^a]*[(1+x)^b]
///
/// Adapted from the MATLAB code by Dirk Laurie and Walter Gautschi
/// http://www.cs.purdue.edu/archives/2002/wxg/codes/r_jacobi.m
///
/// @param[in] N polynomial order
/// @param[in] a weight parameter
/// @param[in] b weight parameter
/// @returns (0) alpha and (1) beta recursion coefficients
template <std::floating_point T>
std::array<std::vector<T>, 2> rec_jacobi(int N, T a, T b)
{
  const T nu = (b - a) / (a + b + 2.0);
  const T mu = std::pow(2.0, (a + b + 1)) * std::tgamma(a + 1.0)
               * std::tgamma(b + 1.0) / std::tgamma(a + b + 2.0);

  std::vector<T> n(N - 1);
  std::iota(n.begin(), n.end(), 1.0);
  std::vector<T> nab(n.size());
  std::ranges::transform(n, nab.begin(),
                         [a, b](auto x) { return 2.0 * x + a + b; });

  std::vector<T> alpha(N);
  alpha.front() = nu;
  std::ranges::transform(nab, std::next(alpha.begin()), [a, b](auto nab)
                         { return (b * b - a * a) / (nab * (nab + 2.0)); });

  std::vector<T> beta(N);
  beta.front() = mu;
  std::ranges::transform(n, nab, std::next(beta.begin()),
                         [a, b](auto n, auto nab)
                         {
                           return 4 * (n + a) * (n + b) * n * (n + a + b)
                                  / (nab * nab * (nab + 1.0) * (nab - 1.0));
                         });

  return {std::move(alpha), std::move(beta)};
}
//----------------------------------------------------------------------------

/// @todo Add detail on alpha and beta
/// @brief Compute Gauss points and weights on the domain [-1, 1] using
/// the Golub-Welsch algorithm
///
/// https://en.wikipedia.org/wiki/Gaussian_quadrature#The_Golub-Welsch_algorithm
/// @param[in] alpha
/// @param[in] beta
/// @return (0) coordinates and (1) weights
template <std::floating_point T>
std::array<std::vector<T>, 2> gauss(std::span<const T> alpha,
                                    std::span<const T> beta)
{
  std::vector<T> Abuffer(alpha.size() * alpha.size(), 0);
  mdspan_t<T, 2> A(Abuffer.data(), alpha.size(), alpha.size());
  for (std::size_t i = 0; i < alpha.size(); ++i)
    A(i, i) = alpha[i];
  for (std::size_t i = 0; i < alpha.size() - 1; ++i)
  {
    A(i + 1, i) = std::sqrt(beta[i + 1]);
    A(i, i + 1) = std::sqrt(beta[i + 1]);
  }

  auto [evals, evecs] = math::eigh<T>(Abuffer, alpha.size());

  // Determine weights from the first component of each eigenvector
  std::vector<T> w(alpha.size());
  for (std::size_t i = 0; i < alpha.size(); ++i)
    w[i] = beta[0] * evecs[i * alpha.size()] * evecs[i * alpha.size()];

  return {std::move(evals), std::move(w)};
}
//----------------------------------------------------------------------------

/// @brief Compute the Lobatto nodes and weights with the preassigned
/// nodes xl1, xl2
///
/// Based on the section 7 of the paper "Some modified matrix eigenvalue
/// problems", https://doi.org/10.1137/1015032.
///
/// @param[in] alpha recursion coefficients
/// @param[in] beta recursion coefficients
/// @param[in] xl1 assigned node location
/// @param[in] xl2 assigned node location
/// @returns (0) quadrature positions and (1) quadrature weights.
template <std::floating_point T>
std::array<std::vector<T>, 2> lobatto(std::span<const T> alpha,
                                      std::span<const T> beta, T xl1, T xl2)
{
  assert(alpha.size() == beta.size());

  // Solve tridiagonal system using Thomas algorithm
  T g1(0.0), g2(0.0);
  const std::size_t n = alpha.size();
  for (std::size_t i = 1; i < n - 1; ++i)
  {
    g1 = std::sqrt(beta[i]) / (alpha[i] - xl1 - std::sqrt(beta[i - 1]) * g1);
    g2 = std::sqrt(beta[i]) / (alpha[i] - xl2 - std::sqrt(beta[i - 1]) * g2);
  }
  g1 = 1.0 / (alpha[n - 1] - xl1 - std::sqrt(beta[n - 2]) * g1);
  g2 = 1.0 / (alpha[n - 1] - xl2 - std::sqrt(beta[n - 2]) * g2);

  std::vector<T> alpha_l(alpha.begin(), alpha.end());
  alpha_l[n - 1] = (g1 * xl2 - g2 * xl1) / (g1 - g2);
  std::vector<T> beta_l(beta.begin(), beta.end());
  beta_l[n - 1] = (xl2 - xl1) / (g1 - g2);

  return gauss(std::span<const T>(alpha_l), std::span<const T>(beta_l));
}
//-----------------------------------------------------------------------------

/// Evaluate the nth Jacobi polynomial and derivatives with weight
/// parameters (a, 0) at points x
/// @param[in] a Jacobi weight a
/// @param[in] n Order of polynomial
/// @param[in] nderiv Number of derivatives (if zero, just compute
/// polynomial itself)
/// @param[in] x Points at which to evaluate
/// @return Array of polynomial derivative values (rows) at points
/// (columns)
template <std::floating_point T>
mdarray_t<T, 2> compute_jacobi_deriv(T a, std::size_t n, std::size_t nderiv,
                                     std::span<const T> x)
{
  std::vector<std::size_t> shape = {x.size()};
  mdarray_t<T, 3> J(nderiv + 1, n + 1, x.size());
  mdarray_t<T, 2> Jd(n + 1, x.size());
  for (std::size_t i = 0; i < nderiv + 1; ++i)
  {
    if (i == 0)
    {
      for (std::size_t j = 0; j < Jd.extent(1); ++j)
        Jd(0, j) = 1.0;
    }
    else
    {
      for (std::size_t j = 0; j < Jd.extent(1); ++j)
        Jd(0, j) = 0.0;
    }

    if (n > 0)
    {
      if (i == 0)
      {
        for (std::size_t j = 0; j < Jd.extent(1); ++j)
          Jd(1, j) = (x[j] * (a + 2.0) + a) * 0.5;
      }
      else if (i == 1)
      {
        for (std::size_t j = 0; j < Jd.extent(1); ++j)
          Jd(1, j) = a * 0.5 + 1;
      }
      else
      {
        for (std::size_t j = 0; j < Jd.extent(1); ++j)
          Jd(1, j) = 0.0;
      }
    }

    for (std::size_t j = 2; j < n + 1; ++j)
    {
      const T a1 = 2 * j * (j + a) * (2 * j + a - 2);
      const T a2 = (2 * j + a - 1) * (a * a) / a1;
      const T a3 = (2 * j + a - 1) * (2 * j + a) / (2 * j * (j + a));
      const T a4 = 2 * (j + a - 1) * (j - 1) * (2 * j + a) / a1;
      for (std::size_t k = 0; k < Jd.extent(1); ++k)
        Jd(j, k) = Jd(j - 1, k) * (x[k] * a3 + a2) - Jd(j - 2, k) * a4;
      if (i > 0)
      {
        for (std::size_t k = 0; k < Jd.extent(1); ++k)
          Jd(j, k) += i * a3 * J(i - 1, j - 1, k);
      }
    }

    for (std::size_t j = 0; j < Jd.extent(0); ++j)
      for (std::size_t k = 0; k < Jd.extent(1); ++k)
        J(i, j, k) = Jd(j, k);
  }

  mdarray_t<T, 2> result(nderiv + 1, x.size());
  for (std::size_t i = 0; i < result.extent(0); ++i)
    for (std::size_t j = 0; j < result.extent(1); ++j)
      result(i, j) = J(i, n, j);

  return result;
}
//----------------------------------------------------------------------------

/// Computes the m roots of \f$P_{m}^{a,0}\f$ on [-1,1] by Newton's
/// method. The initial guesses are the Chebyshev points.  Algorithm
/// implemented from the pseudocode given by Karniadakis and Sherwin.
template <std::floating_point T>
std::vector<T> compute_gauss_jacobi_points(T a, int m)
{
  constexpr T eps = 1.0e-8;
  constexpr int max_iter = 100;
  std::vector<T> x(m);
  for (int k = 0; k < m; ++k)
  {
    // Initial guess
    x[k] = -std::cos((2.0 * k + 1.0) * M_PI / (2.0 * m));
    if (k > 0)
      x[k] = 0.5 * (x[k] + x[k - 1]);

    int j = 0;
    while (j < max_iter)
    {
      T s = 0;
      for (int i = 0; i < k; ++i)
        s += 1.0 / (x[k] - x[i]);
      std::span<const T> _x(&x[k], 1);
      mdarray_t<T, 2> f = compute_jacobi_deriv<T>(a, m, 1, _x);
      T delta = f(0, 0) / (f(1, 0) - f(0, 0) * s);
      x[k] -= delta;
      if (std::abs(delta) < eps)
        break;
      ++j;
    }
  }

  return x;
}
//-----------------------------------------------------------------------------

/// @note Computes on [-1, 1]
template <std::floating_point T>
std::array<std::vector<T>, 2> compute_gauss_jacobi_rule(T a, int m)
{
  std::vector<T> pts = compute_gauss_jacobi_points<T>(a, m);
  mdarray_t<T, 2> Jd = compute_jacobi_deriv<T>(a, m, 1, pts);
  T a1 = std::pow(2.0, a + 1.0);
  std::vector<T> wts(m);
  for (int i = 0; i < m; ++i)
  {
    T x = pts[i];
    T f = Jd(1, i);
    wts[i] = a1 / (1.0 - x * x) / (f * f);
  }

  return {std::move(pts), std::move(wts)};
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::array<std::vector<T>, 2> make_quadrature_line(int m)
{
  auto [ptx, wx] = compute_gauss_jacobi_rule<T>(0.0, m);
  std::ranges::transform(wx, wx.begin(), [](auto w) { return 0.5 * w; });
  std::ranges::transform(ptx, ptx.begin(),
                         [](auto x) { return 0.5 * (x + 1.0); });
  return {std::move(ptx), std::move(wx)};
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::array<std::vector<T>, 2> make_quadrature_triangle_collapsed(std::size_t m)
{
  auto [ptx, wx] = compute_gauss_jacobi_rule<T>(0.0, m);
  auto [pty, wy] = compute_gauss_jacobi_rule<T>(1.0, m);

  std::vector<T> pts(m * m * 2);
  mdspan_t<T, 2> x(pts.data(), m * m, 2);
  std::vector<T> wts(m * m);
  int c = 0;
  for (std::size_t i = 0; i < m; ++i)
  {
    for (std::size_t j = 0; j < m; ++j)
    {
      x(c, 0) = 0.25 * (1.0 + ptx[i]) * (1.0 - pty[j]);
      x(c, 1) = 0.5 * (1.0 + pty[j]);
      wts[c] = wx[i] * wy[j] * 0.125;
      ++c;
    }
  }

  return {std::move(pts), std::move(wts)};
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::array<std::vector<T>, 2>
make_quadrature_tetrahedron_collapsed(std::size_t m)
{
  auto [ptx, wx] = compute_gauss_jacobi_rule<T>(0.0, m);
  auto [pty, wy] = compute_gauss_jacobi_rule<T>(1.0, m);
  auto [ptz, wz] = compute_gauss_jacobi_rule<T>(2.0, m);

  std::vector<T> pts(m * m * m * 3);
  mdspan_t<T, 2> x(pts.data(), m * m * m, 3);
  std::vector<T> wts(m * m * m);
  int c = 0;
  for (std::size_t i = 0; i < m; ++i)
  {
    for (std::size_t j = 0; j < m; ++j)
    {
      for (std::size_t k = 0; k < m; ++k)
      {
        x(c, 0) = 0.125 * (1.0 + ptx[i]) * (1.0 - pty[j]) * (1.0 - ptz[k]);
        x(c, 1) = 0.25 * (1. + pty[j]) * (1. - ptz[k]);
        x(c, 2) = 0.5 * (1.0 + ptz[k]);
        wts[c] = wx[i] * wy[j] * wz[k] * 0.125 * 0.125;
        ++c;
      }
    }
  }

  return {std::move(pts), std::move(wts)};
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::array<std::vector<T>, 2> make_gauss_jacobi_quadrature(cell::type celltype,
                                                           std::size_t m)
{
  const std::size_t np = (m + 2) / 2;
  switch (celltype)
  {
  case cell::type::interval:
    return make_quadrature_line<T>(np);
  case cell::type::quadrilateral:
  {
    auto [QptsL, QwtsL] = make_quadrature_line<T>(np);
    std::vector<T> pts(np * np * 2);
    mdspan_t<T, 2> x(pts.data(), np * np, 2);
    std::vector<T> wts(np * np);
    int c = 0;
    for (std::size_t i = 0; i < np; ++i)
    {
      for (std::size_t j = 0; j < np; ++j)
      {
        x(c, 0) = QptsL[i];
        x(c, 1) = QptsL[j];
        wts[c] = QwtsL[i] * QwtsL[j];
        ++c;
      }
    }
    return {std::move(pts), std::move(wts)};
  }
  case cell::type::hexahedron:
  {
    auto [QptsL, QwtsL] = make_quadrature_line<T>(np);
    std::vector<T> pts(np * np * np * 3);
    mdspan_t<T, 2> x(pts.data(), np * np * np, 3);
    std::vector<T> wts(np * np * np);
    int c = 0;
    for (std::size_t i = 0; i < np; ++i)
    {
      for (std::size_t j = 0; j < np; ++j)
      {
        for (std::size_t k = 0; k < np; ++k)
        {
          x(c, 0) = QptsL[i];
          x(c, 1) = QptsL[j];
          x(c, 2) = QptsL[k];
          wts[c] = QwtsL[i] * QwtsL[j] * QwtsL[k];
          ++c;
        }
      }
    }
    return {std::move(pts), std::move(wts)};
  }
  case cell::type::prism:
  {
    const auto [QptsL, QwtsL] = make_quadrature_line<T>(np);
    const auto [_QptsT, QwtsT] = make_quadrature_triangle_collapsed<T>(np);
    mdspan_t<const T, 2> QptsT(_QptsT.data(), QwtsT.size(),
                               _QptsT.size() / QwtsT.size());
    std::vector<T> pts(np * QptsT.extent(0) * 3);
    mdspan_t<T, 2> x(pts.data(), np * QptsT.extent(0), 3);
    std::vector<T> wts(np * QptsT.extent(0));
    int c = 0;
    for (std::size_t i = 0; i < QptsT.extent(0); ++i)
    {
      for (std::size_t k = 0; k < np; ++k)
      {
        x(c, 0) = QptsT(i, 0);
        x(c, 1) = QptsT(i, 1);
        x(c, 2) = QptsL[k];
        wts[c] = QwtsT[i] * QwtsL[k];
        ++c;
      }
    }
    return {std::move(pts), std::move(wts)};
  }
  case cell::type::pyramid:
  {
    auto [pts, wts]
        = make_gauss_jacobi_quadrature<T>(cell::type::hexahedron, m + 2);
    mdspan_t<T, 2> x(pts.data(), pts.size() / 3, 3);
    for (std::size_t i = 0; i < x.extent(0); ++i)
    {
      const auto z = x(i, 2);
      x(i, 0) *= (1 - z);
      x(i, 1) *= (1 - z);
      wts[i] *= (1 - z) * (1 - z);
    }
    return {std::move(pts), std::move(wts)};
  }
  case cell::type::triangle:
    return make_quadrature_triangle_collapsed<T>(np);
  case cell::type::tetrahedron:
    return make_quadrature_tetrahedron_collapsed<T>(np);
  default:
    throw std::runtime_error("Unsupported celltype for make_quadrature");
  }
}

//-----------------------------------------------------------------------------

/// The Gauss-Lobatto-Legendre quadrature rules on the interval using
/// Greg von Winckel's implementation. This facilitates implementing
/// spectral elements. The quadrature rule uses m points for a degree of
/// precision of 2m-3.
template <std::floating_point T>
std::array<std::vector<T>, 2> compute_gll_rule(int m)
{
  if (m < 2)
  {
    throw std::runtime_error(
        "Gauss-Lobatto-Legendre quadrature invalid for fewer than 2 points");
  }

  // Calculate the recursion coefficients
  // auto [alpha, beta] = rec_jacobi<T>(m, 0.0, 0.0);
  std::array<std::vector<T>, 2> coeffs = rec_jacobi<T>(m, 0.0, 0.0);

  // Compute Lobatto nodes and weights
  auto [xs_ref, ws_ref] = lobatto<T>(std::span<const T>(coeffs[0]),
                                     std::span<const T>(coeffs[1]), -1.0, 1.0);

  // Reorder to match 1d dof ordering
  std::rotate(xs_ref.rbegin(), xs_ref.rbegin() + 1, xs_ref.rend() - 1);
  std::rotate(ws_ref.rbegin(), ws_ref.rbegin() + 1, ws_ref.rend() - 1);
  return {xs_ref, ws_ref};
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::array<std::vector<T>, 2> make_gll_line(int m)
{
  auto [ptx, wx] = compute_gll_rule<T>(m);
  std::ranges::transform(wx, wx.begin(), [](auto w) { return 0.5 * w; });
  std::ranges::transform(ptx, ptx.begin(),
                         [](auto x) { return 0.5 * (x + 1.0); });
  return {ptx, wx};
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::array<std::vector<T>, 2> make_gll_quadrature(cell::type celltype,
                                                  std::size_t m)
{
  const std::size_t np = (m + 4) / 2;
  switch (celltype)
  {
  case cell::type::interval:
    return make_gll_line<T>(np);
  case cell::type::quadrilateral:
  {
    auto [QptsL, QwtsL] = make_gll_line<T>(np);
    std::vector<T> pts(np * np * 2);
    mdspan_t<T, 2> x(pts.data(), np * np, 2);
    std::vector<T> wts(np * np);
    int c = 0;
    for (std::size_t i = 0; i < np; ++i)
    {
      for (std::size_t j = 0; j < np; ++j)
      {
        x(c, 0) = QptsL[i];
        x(c, 1) = QptsL[j];
        wts[c] = QwtsL[i] * QwtsL[j];
        ++c;
      }
    }
    return {std::move(pts), std::move(wts)};
  }
  case cell::type::hexahedron:
  {
    auto [QptsL, QwtsL] = make_gll_line<T>(np);
    std::vector<T> pts(np * np * np * 3);
    mdspan_t<T, 2> x(pts.data(), np * np * np, 3);
    std::vector<T> wts(np * np * np);
    int c = 0;
    for (std::size_t i = 0; i < np; ++i)
    {
      for (std::size_t j = 0; j < np; ++j)
      {
        for (std::size_t k = 0; k < np; ++k)
        {
          x(c, 0) = QptsL[i];
          x(c, 1) = QptsL[j];
          x(c, 2) = QptsL[k];
          wts[c] = QwtsL[i] * QwtsL[j] * QwtsL[k];
          ++c;
        }
      }
    }
    return {std::move(pts), std::move(wts)};
  }
  case cell::type::prism:
    throw std::runtime_error("Prism not yet supported");
  case cell::type::pyramid:
    throw std::runtime_error("Pyramid not yet supported");
  case cell::type::triangle:
    throw std::runtime_error("Triangle not yet supported");
  case cell::type::tetrahedron:
    throw std::runtime_error("Tetrahedron not yet supported");
  default:
    throw std::runtime_error("Unsupported celltype for make_quadrature");
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::array<std::vector<T>, 2> make_strang_fix_quadrature(cell::type celltype,
                                                         std::size_t m)
{
  if (celltype == cell::type::triangle)
  {
    if (m == 2)
    {
      // Scheme from Strang and Fix, 3 points, degree of precision 2
      std::vector<T> x
          = {1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0, 2.0 / 3.0, 1.0 / 6.0};
      std::vector<T> w = {1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0};
      return {std::move(x), std::move(w)};
    }
    else if (m == 3)
    {
      // Scheme from Strang and Fix, 6 points, degree of precision 3
      std::vector<T> x
          = {0.659027622374092, 0.231933368553031, 0.659027622374092,
             0.109039009072877, 0.231933368553031, 0.659027622374092,
             0.231933368553031, 0.109039009072877, 0.109039009072877,
             0.659027622374092, 0.109039009072877, 0.231933368553031};
      std::vector<T> w(6, 1.0 / 12.0);
      return {std::move(x), std::move(w)};
    }
    else if (m == 4)
    {
      // Scheme from Strang and Fix, 6 points, degree of precision 4
      std::vector<T> x
          = {0.816847572980459, 0.091576213509771, 0.091576213509771,
             0.816847572980459, 0.091576213509771, 0.091576213509771,
             0.108103018168070, 0.445948490915965, 0.445948490915965,
             0.108103018168070, 0.445948490915965, 0.445948490915965};
      std::vector<T> w
          = {0.054975871827661,  0.054975871827661,  0.054975871827661,
             0.1116907948390055, 0.1116907948390055, 0.1116907948390055};
      return {std::move(x), std::move(w)};
    }
    else if (m == 5)
    {
      // Scheme from Strang and Fix, 7 points, degree of precision 5
      std::vector<T> x
          = {0.33333333333333333, 0.33333333333333333, 0.79742698535308720,
             0.10128650732345633, 0.10128650732345633, 0.79742698535308720,
             0.10128650732345633, 0.10128650732345633, 0.05971587178976981,
             0.47014206410511505, 0.47014206410511505, 0.05971587178976981,
             0.47014206410511505, 0.47014206410511505};
      std::vector<T> w = {0.1125,
                          0.06296959027241358,
                          0.06296959027241358,
                          0.06296959027241358,
                          0.06619707639425308,
                          0.06619707639425308,
                          0.06619707639425308};
      return {std::move(x), std::move(w)};
    }
    else if (m == 6)
    {
      // Scheme from Strang and Fix, 12 points, degree of precision 6
      std::vector<T> x
          = {0.873821971016996, 0.063089014491502, 0.063089014491502,
             0.873821971016996, 0.063089014491502, 0.063089014491502,
             0.501426509658179, 0.249286745170910, 0.249286745170910,
             0.501426509658179, 0.249286745170910, 0.249286745170910,
             0.636502499121399, 0.310352451033785, 0.636502499121399,
             0.053145049844816, 0.310352451033785, 0.636502499121399,
             0.310352451033785, 0.053145049844816, 0.053145049844816,
             0.636502499121399, 0.053145049844816, 0.310352451033785};
      std::vector<T> w
          = {0.0254224531851035, 0.0254224531851035, 0.0254224531851035,
             0.0583931378631895, 0.0583931378631895, 0.0583931378631895,
             0.041425537809187,  0.041425537809187,  0.041425537809187,
             0.041425537809187,  0.041425537809187,  0.041425537809187};
      return {std::move(x), std::move(w)};
    }
    else
      throw std::runtime_error("Strang-Fix not implemented for this order.");
  }

  throw std::runtime_error("Strang-Fix not implemented for this cell type.");
}
//-------------------------------------------------------------------------------------------------------
template <std::floating_point T>
std::array<std::vector<T>, 2>
make_zienkiewicz_taylor_quadrature(cell::type celltype, std::size_t m)
{
  if (celltype == cell::type::triangle)
  {
    if (m == 0 or m == 1)
    {
      // Scheme from Zienkiewicz and Taylor, 1 point, degree of precision 1
      return {std::vector<T>{1.0 / 3.0, 1.0 / 3.0}, {0.5}};
    }
    else
    {
      throw std::runtime_error(
          "Zienkiewicz-Taylor not implemented for this order.");
    }
  }

  if (celltype == cell::type::tetrahedron)
  {
    if (m == 0 or m == 1)
    {
      // Scheme from Zienkiewicz and Taylor, 1 point, degree of precision 1
      return {std::vector<T>{0.25, 0.25, 0.25}, {1.0 / 6.0}};
    }
    else if (m == 2)
    {
      // Scheme from Zienkiewicz and Taylor, 4 points, degree of
      // precision
      // 2
      constexpr T a = 0.585410196624969, b = 0.138196601125011;
      std::vector<T> x = {a, b, b, b, a, b, b, b, a, b, b, b};
      return {std::move(x), {1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0}};
    }
    else if (m == 3)
    {
      // Scheme from Zienkiewicz and Taylor, 5 points, degree of
      // precision 3 Note : this scheme has a negative weight
      std::vector<T> x{
          0.2500000000000000, 0.2500000000000000, 0.2500000000000000,
          0.5000000000000000, 0.1666666666666666, 0.1666666666666666,
          0.1666666666666666, 0.5000000000000000, 0.1666666666666666,
          0.1666666666666666, 0.1666666666666666, 0.5000000000000000,
          0.1666666666666666, 0.1666666666666666, 0.1666666666666666};
      std::vector<T> w{-0.8 / 6.0, 0.45 / 6.0, 0.45 / 6.0, 0.45 / 6.0,
                       0.45 / 6.0};
      return {std::move(x), std::move(w)};
    }
    else
    {
      throw std::runtime_error(
          "Zienkiewicz-Taylor not implemented for this order.");
    }
  }

  throw std::runtime_error(
      "Zienkiewicz-Taylor not implemented for this cell type.");
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::array<std::vector<T>, 2> make_keast_quadrature(cell::type celltype,
                                                    std::size_t m)
{
  if (celltype == cell::type::tetrahedron)
  {
    if (m == 4)
    {
      // Keast rule, 14 points, degree of precision 4
      // Values taken from
      // http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
      // (KEAST5)
      std::vector<T> x
          = {0.0000000000000000, 0.5000000000000000, 0.5000000000000000,
             0.5000000000000000, 0.0000000000000000, 0.5000000000000000,
             0.5000000000000000, 0.5000000000000000, 0.0000000000000000,
             0.5000000000000000, 0.0000000000000000, 0.0000000000000000,
             0.0000000000000000, 0.5000000000000000, 0.0000000000000000,
             0.0000000000000000, 0.0000000000000000, 0.5000000000000000,
             0.6984197043243866, 0.1005267652252045, 0.1005267652252045,
             0.1005267652252045, 0.1005267652252045, 0.1005267652252045,
             0.1005267652252045, 0.1005267652252045, 0.6984197043243866,
             0.1005267652252045, 0.6984197043243866, 0.1005267652252045,
             0.0568813795204234, 0.3143728734931922, 0.3143728734931922,
             0.3143728734931922, 0.3143728734931922, 0.3143728734931922,
             0.3143728734931922, 0.3143728734931922, 0.0568813795204234,
             0.3143728734931922, 0.0568813795204234, 0.3143728734931922};
      std::vector<T> w
          = {0.003174603174603167, 0.003174603174603167, 0.003174603174603167,
             0.003174603174603167, 0.003174603174603167, 0.003174603174603167,
             0.014764970790496783, 0.014764970790496783, 0.014764970790496783,
             0.014764970790496783, 0.022139791114265117, 0.022139791114265117,
             0.022139791114265117, 0.022139791114265117};
      return {std::move(x), std::move(w)};
    }
    else if (m == 5)
    {
      // Keast rule, 15 points, degree of precision 5
      // Values taken from
      // http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
      // (KEAST6)
      std::vector<T> x
          = {0.2500000000000000, 0.2500000000000000, 0.2500000000000000,
             0.0000000000000000, 0.3333333333333333, 0.3333333333333333,
             0.3333333333333333, 0.3333333333333333, 0.3333333333333333,
             0.3333333333333333, 0.3333333333333333, 0.0000000000000000,
             0.3333333333333333, 0.0000000000000000, 0.3333333333333333,
             0.7272727272727273, 0.0909090909090909, 0.0909090909090909,
             0.0909090909090909, 0.0909090909090909, 0.0909090909090909,
             0.0909090909090909, 0.0909090909090909, 0.7272727272727273,
             0.0909090909090909, 0.7272727272727273, 0.0909090909090909,
             0.4334498464263357, 0.0665501535736643, 0.0665501535736643,
             0.0665501535736643, 0.4334498464263357, 0.0665501535736643,
             0.0665501535736643, 0.0665501535736643, 0.4334498464263357,
             0.0665501535736643, 0.4334498464263357, 0.4334498464263357,
             0.4334498464263357, 0.0665501535736643, 0.4334498464263357,
             0.4334498464263357, 0.4334498464263357, 0.0665501535736643};
      std::vector<T> w
          = {0.030283678097089182, 0.006026785714285717, 0.006026785714285717,
             0.006026785714285717, 0.006026785714285717, 0.011645249086028967,
             0.011645249086028967, 0.011645249086028967, 0.011645249086028967,
             0.010949141561386449, 0.010949141561386449, 0.010949141561386449,
             0.010949141561386449, 0.010949141561386449, 0.010949141561386449};
      return {std::move(x), std::move(w)};
    }
    else if (m == 6)
    {
      // Keast rule, 24 points, degree of precision 6
      // Values taken from
      // http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
      // (KEAST7)
      std::vector<T> x
          = {0.3561913862225449, 0.2146028712591517, 0.2146028712591517,
             0.2146028712591517, 0.2146028712591517, 0.2146028712591517,
             0.2146028712591517, 0.2146028712591517, 0.3561913862225449,
             0.2146028712591517, 0.3561913862225449, 0.2146028712591517,
             0.8779781243961660, 0.0406739585346113, 0.0406739585346113,
             0.0406739585346113, 0.0406739585346113, 0.0406739585346113,
             0.0406739585346113, 0.0406739585346113, 0.8779781243961660,
             0.0406739585346113, 0.8779781243961660, 0.0406739585346113,
             0.0329863295731731, 0.3223378901422757, 0.3223378901422757,
             0.3223378901422757, 0.3223378901422757, 0.3223378901422757,
             0.3223378901422757, 0.3223378901422757, 0.0329863295731731,
             0.3223378901422757, 0.0329863295731731, 0.3223378901422757,
             0.2696723314583159, 0.0636610018750175, 0.0636610018750175,
             0.0636610018750175, 0.2696723314583159, 0.0636610018750175,
             0.0636610018750175, 0.0636610018750175, 0.2696723314583159,
             0.6030056647916491, 0.0636610018750175, 0.0636610018750175,
             0.0636610018750175, 0.6030056647916491, 0.0636610018750175,
             0.0636610018750175, 0.0636610018750175, 0.6030056647916491,
             0.0636610018750175, 0.2696723314583159, 0.6030056647916491,
             0.2696723314583159, 0.6030056647916491, 0.0636610018750175,
             0.6030056647916491, 0.0636610018750175, 0.2696723314583159,
             0.0636610018750175, 0.6030056647916491, 0.2696723314583159,
             0.2696723314583159, 0.0636610018750175, 0.6030056647916491,
             0.6030056647916491, 0.2696723314583159, 0.0636610018750175};
      std::vector<T> w = {
          0.0066537917096946494, 0.0066537917096946494, 0.0066537917096946494,
          0.0066537917096946494, 0.0016795351758867834, 0.0016795351758867834,
          0.0016795351758867834, 0.0016795351758867834, 0.009226196923942399,
          0.009226196923942399,  0.009226196923942399,  0.009226196923942399,
          0.008035714285714283,  0.008035714285714283,  0.008035714285714283,
          0.008035714285714283,  0.008035714285714283,  0.008035714285714283,
          0.008035714285714283,  0.008035714285714283,  0.008035714285714283,
          0.008035714285714283,  0.008035714285714283,  0.008035714285714283};
      return {std::move(x), std::move(w)};
    }
    else if (m == 7)
    {
      // Keast rule, 31 points, degree of precision 7
      // Values taken from
      // http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
      // (KEAST8)
      std::vector<T> x
          = {0.2500000000000000, 0.2500000000000000, 0.2500000000000000,
             0.7653604230090441, 0.0782131923303186, 0.0782131923303186,
             0.0782131923303186, 0.0782131923303186, 0.0782131923303186,
             0.0782131923303186, 0.0782131923303186, 0.7653604230090441,
             0.0782131923303186, 0.7653604230090441, 0.0782131923303186,
             0.6344703500082868, 0.1218432166639044, 0.1218432166639044,
             0.1218432166639044, 0.1218432166639044, 0.1218432166639044,
             0.1218432166639044, 0.1218432166639044, 0.6344703500082868,
             0.1218432166639044, 0.6344703500082868, 0.1218432166639044,
             0.0023825066607383, 0.3325391644464206, 0.3325391644464206,
             0.3325391644464206, 0.3325391644464206, 0.3325391644464206,
             0.3325391644464206, 0.3325391644464206, 0.0023825066607383,
             0.3325391644464206, 0.0023825066607383, 0.3325391644464206,
             0.0000000000000000, 0.5000000000000000, 0.5000000000000000,
             0.5000000000000000, 0.0000000000000000, 0.5000000000000000,
             0.5000000000000000, 0.5000000000000000, 0.0000000000000000,
             0.5000000000000000, 0.0000000000000000, 0.0000000000000000,
             0.0000000000000000, 0.5000000000000000, 0.0000000000000000,
             0.0000000000000000, 0.0000000000000000, 0.5000000000000000,
             0.2000000000000000, 0.1000000000000000, 0.1000000000000000,
             0.1000000000000000, 0.2000000000000000, 0.1000000000000000,
             0.1000000000000000, 0.1000000000000000, 0.2000000000000000,
             0.6000000000000000, 0.1000000000000000, 0.1000000000000000,
             0.1000000000000000, 0.6000000000000000, 0.1000000000000000,
             0.1000000000000000, 0.1000000000000000, 0.6000000000000000,
             0.1000000000000000, 0.2000000000000000, 0.6000000000000000,
             0.2000000000000000, 0.6000000000000000, 0.1000000000000000,
             0.6000000000000000, 0.1000000000000000, 0.2000000000000000,
             0.1000000000000000, 0.6000000000000000, 0.2000000000000000,
             0.2000000000000000, 0.1000000000000000, 0.6000000000000000,
             0.6000000000000000, 0.2000000000000000, 0.1000000000000000};
      std::vector<T> w
          = {0.0182642234661088,   0.010599941524414166, 0.010599941524414166,
             0.010599941524414166, 0.010599941524414166, -0.06251774011432995,
             -0.06251774011432995, -0.06251774011432995, -0.06251774011432995,
             0.004891425263073534, 0.004891425263073534, 0.004891425263073534,
             0.004891425263073534, 0.0009700176366843,   0.0009700176366843,
             0.0009700176366843,   0.0009700176366843,   0.0009700176366843,
             0.0009700176366843,   0.02755731922398508,  0.02755731922398508,
             0.02755731922398508,  0.02755731922398508,  0.02755731922398508,
             0.02755731922398508,  0.02755731922398508,  0.02755731922398508,
             0.02755731922398508,  0.02755731922398508,  0.02755731922398508,
             0.02755731922398508};
      return {std::move(x), std::move(w)};
    }
    else if (m == 8)
    {
      // Keast rule, 45 points, degree of precision 8
      // Values taken from
      // http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
      // (KEAST9)
      std::vector<T> x
          = {0.2500000000000000, 0.2500000000000000, 0.2500000000000000,
             0.6175871903000830, 0.1274709365666390, 0.1274709365666390,
             0.1274709365666390, 0.1274709365666390, 0.1274709365666390,
             0.1274709365666390, 0.1274709365666390, 0.6175871903000830,
             0.1274709365666390, 0.6175871903000830, 0.1274709365666390,
             0.9037635088221031, 0.0320788303926323, 0.0320788303926323,
             0.0320788303926323, 0.0320788303926323, 0.0320788303926323,
             0.0320788303926323, 0.0320788303926323, 0.9037635088221031,
             0.0320788303926323, 0.9037635088221031, 0.0320788303926323,
             0.4502229043567190, 0.0497770956432810, 0.0497770956432810,
             0.0497770956432810, 0.4502229043567190, 0.0497770956432810,
             0.0497770956432810, 0.0497770956432810, 0.4502229043567190,
             0.0497770956432810, 0.4502229043567190, 0.4502229043567190,
             0.4502229043567190, 0.0497770956432810, 0.4502229043567190,
             0.4502229043567190, 0.4502229043567190, 0.0497770956432810,
             0.3162695526014501, 0.1837304473985499, 0.1837304473985499,
             0.1837304473985499, 0.3162695526014501, 0.1837304473985499,
             0.1837304473985499, 0.1837304473985499, 0.3162695526014501,
             0.1837304473985499, 0.3162695526014501, 0.3162695526014501,
             0.3162695526014501, 0.1837304473985499, 0.3162695526014501,
             0.3162695526014501, 0.3162695526014501, 0.1837304473985499,
             0.0229177878448171, 0.2319010893971509, 0.2319010893971509,
             0.2319010893971509, 0.0229177878448171, 0.2319010893971509,
             0.2319010893971509, 0.2319010893971509, 0.0229177878448171,
             0.5132800333608811, 0.2319010893971509, 0.2319010893971509,
             0.2319010893971509, 0.5132800333608811, 0.2319010893971509,
             0.2319010893971509, 0.2319010893971509, 0.5132800333608811,
             0.2319010893971509, 0.0229177878448171, 0.5132800333608811,
             0.0229177878448171, 0.5132800333608811, 0.2319010893971509,
             0.5132800333608811, 0.2319010893971509, 0.0229177878448171,
             0.2319010893971509, 0.5132800333608811, 0.0229177878448171,
             0.0229177878448171, 0.2319010893971509, 0.5132800333608811,
             0.5132800333608811, 0.0229177878448171, 0.2319010893971509,
             0.7303134278075384, 0.0379700484718286, 0.0379700484718286,
             0.0379700484718286, 0.7303134278075384, 0.0379700484718286,
             0.0379700484718286, 0.0379700484718286, 0.7303134278075384,
             0.1937464752488044, 0.0379700484718286, 0.0379700484718286,
             0.0379700484718286, 0.1937464752488044, 0.0379700484718286,
             0.0379700484718286, 0.0379700484718286, 0.1937464752488044,
             0.0379700484718286, 0.7303134278075384, 0.1937464752488044,
             0.7303134278075384, 0.1937464752488044, 0.0379700484718286,
             0.1937464752488044, 0.0379700484718286, 0.7303134278075384,
             0.0379700484718286, 0.1937464752488044, 0.7303134278075384,
             0.7303134278075384, 0.0379700484718286, 0.1937464752488044,
             0.1937464752488044, 0.7303134278075384, 0.0379700484718286};
      std::vector<T> w = {
          -0.03932700664129262,  0.0040813160593427,    0.0040813160593427,
          0.0040813160593427,    0.0040813160593427,    0.0006580867733043499,
          0.0006580867733043499, 0.0006580867733043499, 0.0006580867733043499,
          0.00438425882512285,   0.00438425882512285,   0.00438425882512285,
          0.00438425882512285,   0.00438425882512285,   0.00438425882512285,
          0.013830063842509817,  0.013830063842509817,  0.013830063842509817,
          0.013830063842509817,  0.013830063842509817,  0.013830063842509817,
          0.004240437424683717,  0.004240437424683717,  0.004240437424683717,
          0.004240437424683717,  0.004240437424683717,  0.004240437424683717,
          0.004240437424683717,  0.004240437424683717,  0.004240437424683717,
          0.004240437424683717,  0.004240437424683717,  0.004240437424683717,
          0.0022387397396142,    0.0022387397396142,    0.0022387397396142,
          0.0022387397396142,    0.0022387397396142,    0.0022387397396142,
          0.0022387397396142,    0.0022387397396142,    0.0022387397396142,
          0.0022387397396142,    0.0022387397396142,    0.0022387397396142};
      return {std::move(x), std::move(w)};
    }
    else
      throw std::runtime_error("Keast not implemented for this order.");
  }
  else
    throw std::runtime_error("Keast not implemented for this cell type.");
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::array<std::vector<T>, 2> make_xiao_gimbutas_quadrature(cell::type celltype,
                                                            int m)
{
  return get_from_quadraturerules<T>(
      quadraturerules::QuadratureRule::XiaoGimbutas, celltype, m);
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::array<std::vector<T>, 2>
make_macroedge_quadrature(quadrature::type rule, cell::type celltype, int m)
{
  auto standard_q = quadrature::make_quadrature<T>(rule, celltype,
                                                   polyset::type::standard, m);
  if (m == 0)
  {
    return standard_q;
  }
  switch (celltype)
  {
  case cell::type::interval:
  {
    const std::size_t npts = standard_q[0].size();
    std::vector<T> x(npts * 2);
    std::vector<T> w(npts * 2);
    for (std::size_t i = 0; i < npts; ++i)
    {
      x[i] = 0.5 * standard_q[0][i];
      x[npts + i] = 0.5 + 0.5 * standard_q[0][i];
      w[i] = 0.5 * standard_q[1][i];
      w[npts + i] = 0.5 * standard_q[1][i];
    }
    return {std::move(x), std::move(w)};
  }
  case cell::type::triangle:
  {
    const std::size_t npts = standard_q[0].size() / 2;
    std::vector<T> x(npts * 8);
    std::vector<T> w(npts * 4);
    for (std::size_t i = 0; i < npts; ++i)
    {
      x[2 * i] = 0.5 * standard_q[0][2 * i];
      x[2 * i + 1] = 0.5 * standard_q[0][2 * i + 1];
      x[2 * (npts + i)] = 0.5 + 0.5 * standard_q[0][2 * i];
      x[2 * (npts + i) + 1] = 0.5 * standard_q[0][2 * i + 1];
      x[2 * (2 * npts + i)] = 0.5 * standard_q[0][2 * i];
      x[2 * (2 * npts + i) + 1] = 0.5 + 0.5 * standard_q[0][2 * i + 1];
      x[2 * (3 * npts + i)] = 0.5 - 0.5 * standard_q[0][2 * i];
      x[2 * (3 * npts + i) + 1] = 0.5 - 0.5 * standard_q[0][2 * i + 1];
      w[i] = 0.25 * standard_q[1][i];
      w[npts + i] = 0.25 * standard_q[1][i];
      w[2 * npts + i] = 0.25 * standard_q[1][i];
      w[3 * npts + i] = 0.25 * standard_q[1][i];
    }
    return {std::move(x), std::move(w)};
  }
  case cell::type::tetrahedron:
  {
    const std::size_t npts = standard_q[0].size() / 3;
    std::vector<T> x(npts * 24);
    std::vector<T> w(npts * 8);
    for (std::size_t i = 0; i < npts; ++i)
    {
      x[3 * i] = 0.5 * standard_q[0][3 * i];
      x[3 * i + 1] = 0.5 * standard_q[0][3 * i + 1];
      x[3 * i + 2] = 0.5 * standard_q[0][3 * i + 2];
      x[3 * (i + npts)] = 1.0 - 0.5 * standard_q[0][3 * i]
                          - 0.5 * standard_q[0][3 * i + 1]
                          - 0.5 * standard_q[0][3 * i + 2];
      x[3 * (i + npts) + 1] = 0.5 * standard_q[0][3 * i];
      x[3 * (i + npts) + 2] = 0.5 * standard_q[0][3 * i + 2];
      x[3 * (i + 2 * npts)] = 0.5 * standard_q[0][3 * i];
      x[3 * (i + 2 * npts) + 1] = 1.0 - 0.5 * standard_q[0][3 * i]
                                  - 0.5 * standard_q[0][3 * i + 1]
                                  - 0.5 * standard_q[0][3 * i + 2];
      x[3 * (i + 2 * npts) + 2] = 0.5 * standard_q[0][3 * i + 1];
      x[3 * (i + 3 * npts)] = 0.5 * standard_q[0][3 * i + 1];
      x[3 * (i + 3 * npts) + 1] = 0.5 * standard_q[0][3 * i];
      x[3 * (i + 3 * npts) + 2] = 1.0 - 0.5 * standard_q[0][3 * i]
                                  - 0.5 * standard_q[0][3 * i + 1]
                                  - 0.5 * standard_q[0][3 * i + 2];
      x[3 * (i + 4 * npts)] = 0.5 - 0.5 * standard_q[0][3 * i + 1]
                              - 0.5 * standard_q[0][3 * i + 2];
      x[3 * (i + 4 * npts) + 1] = 0.5 * standard_q[0][3 * i + 2];
      x[3 * (i + 4 * npts) + 2]
          = 0.5 - 0.5 * standard_q[0][3 * i] - 0.5 * standard_q[0][3 * i + 2];
      x[3 * (i + 5 * npts)] = 0.5 * standard_q[0][3 * i]
                              + 0.5 * standard_q[0][3 * i + 1]
                              + 0.5 * standard_q[0][3 * i + 2];
      x[3 * (i + 5 * npts) + 1] = 0.5 - 0.5 * standard_q[0][3 * i + 1]
                                  - 0.5 * standard_q[0][3 * i + 2];
      x[3 * (i + 5 * npts) + 2] = 0.5 * standard_q[0][3 * i + 1];
      x[3 * (i + 6 * npts)] = 0.5 - 0.5 * standard_q[0][3 * i + 1]
                              - 0.5 * standard_q[0][3 * i + 2];
      x[3 * (i + 6 * npts) + 1] = 0.5 * standard_q[0][3 * i]
                                  + 0.5 * standard_q[0][3 * i + 1]
                                  + 0.5 * standard_q[0][3 * i + 2];
      x[3 * (i + 6 * npts) + 2]
          = 0.5 - 0.5 * standard_q[0][3 * i] - 0.5 * standard_q[0][3 * i + 1];
      x[3 * (i + 7 * npts)] = 0.5 * standard_q[0][3 * i + 2];
      x[3 * (i + 7 * npts) + 1] = 0.5 - 0.5 * standard_q[0][3 * i + 1]
                                  - 0.5 * standard_q[0][3 * i + 2];
      x[3 * (i + 7 * npts) + 2] = 0.5 * standard_q[0][3 * i]
                                  + 0.5 * standard_q[0][3 * i + 1]
                                  + 0.5 * standard_q[0][3 * i + 2];
      w[i] = 0.125 * standard_q[1][i];
      w[npts + i] = 0.125 * standard_q[1][i];
      w[2 * npts + i] = 0.125 * standard_q[1][i];
      w[3 * npts + i] = 0.125 * standard_q[1][i];
      w[4 * npts + i] = 0.125 * standard_q[1][i];
      w[5 * npts + i] = 0.125 * standard_q[1][i];
      w[6 * npts + i] = 0.125 * standard_q[1][i];
      w[7 * npts + i] = 0.125 * standard_q[1][i];
    }
    return {std::move(x), std::move(w)};
  }
  case cell::type::quadrilateral:
  {
    const std::size_t npts = standard_q[0].size() / 2;
    std::vector<T> x(npts * 8);
    std::vector<T> w(npts * 4);
    for (std::size_t i = 0; i < npts; ++i)
    {
      x[2 * i] = 0.5 * standard_q[0][2 * i];
      x[2 * i + 1] = 0.5 * standard_q[0][2 * i + 1];
      x[2 * (npts + i)] = 0.5 + 0.5 * standard_q[0][2 * i];
      x[2 * (npts + i) + 1] = 0.5 * standard_q[0][2 * i + 1];
      x[2 * (2 * npts + i)] = 0.5 * standard_q[0][2 * i];
      x[2 * (2 * npts + i) + 1] = 0.5 + 0.5 * standard_q[0][2 * i + 1];
      x[2 * (3 * npts + i)] = 0.5 + 0.5 * standard_q[0][2 * i];
      x[2 * (3 * npts + i) + 1] = 0.5 + 0.5 * standard_q[0][2 * i + 1];
      w[i] = 0.25 * standard_q[1][i];
      w[npts + i] = 0.25 * standard_q[1][i];
      w[2 * npts + i] = 0.25 * standard_q[1][i];
      w[3 * npts + i] = 0.25 * standard_q[1][i];
    }
    return {std::move(x), std::move(w)};
  }
  case cell::type::hexahedron:
  {
    const std::size_t npts = standard_q[0].size() / 3;
    std::vector<T> x(npts * 24);
    std::vector<T> w(npts * 8);
    for (std::size_t i = 0; i < npts; ++i)
    {
      x[3 * i] = 0.5 * standard_q[0][3 * i];
      x[3 * i + 1] = 0.5 * standard_q[0][3 * i + 1];
      x[3 * i + 2] = 0.5 * standard_q[0][3 * i + 2];
      x[3 * (npts + i)] = 0.5 + 0.5 * standard_q[0][3 * i];
      x[3 * (npts + i) + 1] = 0.5 * standard_q[0][3 * i + 1];
      x[3 * (npts + i) + 2] = 0.5 * standard_q[0][3 * i + 2];
      x[3 * (2 * npts + i)] = 0.5 * standard_q[0][3 * i];
      x[3 * (2 * npts + i) + 1] = 0.5 + 0.5 * standard_q[0][3 * i + 1];
      x[3 * (2 * npts + i) + 2] = 0.5 * standard_q[0][3 * i + 2];
      x[3 * (3 * npts + i)] = 0.5 + 0.5 * standard_q[0][3 * i];
      x[3 * (3 * npts + i) + 1] = 0.5 + 0.5 * standard_q[0][3 * i + 1];
      x[3 * (3 * npts + i) + 2] = 0.5 * standard_q[0][3 * i + 2];
      x[3 * (4 * npts + i)] = 0.5 * standard_q[0][3 * i];
      x[3 * (4 * npts + i) + 1] = 0.5 * standard_q[0][3 * i + 1];
      x[3 * (4 * npts + i) + 2] = 0.5 + 0.5 * standard_q[0][3 * i + 2];
      x[3 * (5 * npts + i)] = 0.5 + 0.5 * standard_q[0][3 * i];
      x[3 * (5 * npts + i) + 1] = 0.5 * standard_q[0][3 * i + 1];
      x[3 * (5 * npts + i) + 2] = 0.5 + 0.5 * standard_q[0][3 * i + 2];
      x[3 * (6 * npts + i)] = 0.5 * standard_q[0][3 * i];
      x[3 * (6 * npts + i) + 1] = 0.5 + 0.5 * standard_q[0][3 * i + 1];
      x[3 * (6 * npts + i) + 2] = 0.5 + 0.5 * standard_q[0][3 * i + 2];
      x[3 * (7 * npts + i)] = 0.5 + 0.5 * standard_q[0][3 * i];
      x[3 * (7 * npts + i) + 1] = 0.5 + 0.5 * standard_q[0][3 * i + 1];
      x[3 * (7 * npts + i) + 2] = 0.5 + 0.5 * standard_q[0][3 * i + 2];
      w[i] = 0.125 * standard_q[1][i];
      w[npts + i] = 0.125 * standard_q[1][i];
      w[2 * npts + i] = 0.125 * standard_q[1][i];
      w[3 * npts + i] = 0.125 * standard_q[1][i];
      w[4 * npts + i] = 0.125 * standard_q[1][i];
      w[5 * npts + i] = 0.125 * standard_q[1][i];
      w[6 * npts + i] = 0.125 * standard_q[1][i];
      w[7 * npts + i] = 0.125 * standard_q[1][i];
    }
    return {std::move(x), std::move(w)};
  }
  default:
    throw std::runtime_error("Macro quadrature not supported on this cell.");
  }
}
//-----------------------------------------------------------------------------
} // namespace
//-----------------------------------------------------------------------------
quadrature::type quadrature::get_default_rule(cell::type celltype, int m)
{
  if (celltype == cell::type::triangle)
  {
    if (m <= 1)
      return type::zienkiewicz_taylor;
    else if (m <= 6)
      return type::strang_fix;
    else if (m <= 30)
      return type::xiao_gimbutas;
    else
      return type::gauss_jacobi;
  }
  else if (celltype == cell::type::tetrahedron)
  {
    if (m <= 3)
      return type::zienkiewicz_taylor;
    else if (m <= 8)
      return type::keast;
    else if (m <= 15)
      return type::xiao_gimbutas;
    else
      return type::gauss_jacobi;
  }
  else
    return type::gauss_jacobi;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::array<std::vector<T>, 2> quadrature::gauss_jacobi_rule(T a, int m)
{
  auto [pts, wts] = compute_gauss_jacobi_rule(a, m);
  for (std::size_t i = 0; i < wts.size(); ++i)
  {
    pts[i] += 1.0;
    pts[i] /= 2.0;
    wts[i] /= std::pow(2.0, a + 1);
  }
  return {std::move(pts), std::move(wts)};
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::array<std::vector<T>, 2>
quadrature::make_quadrature(quadrature::type rule, cell::type celltype,
                            polyset::type polytype, int m)
{
  switch (polytype)
  {
  case polyset::type::standard:
    switch (rule)
    {
    case quadrature::type::Default:
      return make_quadrature<T>(get_default_rule(celltype, m), celltype,
                                polytype, m);
    case quadrature::type::gauss_jacobi:
      return make_gauss_jacobi_quadrature<T>(celltype, m);
    case quadrature::type::gll:
      return make_gll_quadrature<T>(celltype, m);
    case quadrature::type::xiao_gimbutas:
      return make_xiao_gimbutas_quadrature<T>(celltype, m);
    case quadrature::type::zienkiewicz_taylor:
      return make_zienkiewicz_taylor_quadrature<T>(celltype, m);
    case quadrature::type::keast:
      return make_keast_quadrature<T>(celltype, m);
    case quadrature::type::strang_fix:
      return make_strang_fix_quadrature<T>(celltype, m);
    default:
      throw std::runtime_error("Unknown quadrature rule");
    }
  case polyset::type::macroedge:
    return make_macroedge_quadrature<T>(rule, celltype, m);
  default:
    throw std::runtime_error("Unsupported polyset type");
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::vector<T> quadrature::get_gl_points(int m)
{
  std::vector<T> pts = compute_gauss_jacobi_points<T>(0, m);
  std::ranges::transform(pts, pts.begin(),
                         [](auto x) { return 0.5 + 0.5 * x; });
  return pts;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::vector<T> quadrature::get_gll_points(int m)
{
  return make_gll_line<T>(m)[0];
}
//-----------------------------------------------------------------------------
/// @cond DOXYGEN_SHOULD_SKIP_THIS
template std::array<std::vector<float>, 2>
quadrature::make_quadrature(quadrature::type, cell::type, polyset::type, int);
template std::array<std::vector<double>, 2>
quadrature::make_quadrature(quadrature::type, cell::type, polyset::type, int);

template std::vector<float> quadrature::get_gl_points(int);
template std::vector<double> quadrature::get_gl_points(int);

template std::vector<float> quadrature::get_gll_points(int);
template std::vector<double> quadrature::get_gll_points(int);

template std::array<std::vector<float>, 2> quadrature::gauss_jacobi_rule(float,
                                                                         int);
template std::array<std::vector<double>, 2>
quadrature::gauss_jacobi_rule(double, int);
/// @endcond
//-----------------------------------------------------------------------------
