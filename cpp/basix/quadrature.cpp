// Copyright (c) 2020 Chris Richardson and Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "quadrature.h"
#include "math.h"
#include "mdspan.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace basix;

template <typename T, std::size_t d>
using mdspan_t = md::mdspan<T, md::dextents<std::size_t, d>>;
template <typename T, std::size_t d>
using mdarray_t = mdex::mdarray<T, md::dextents<std::size_t, d>>;

namespace
{
//----------------------------------------------------------------------------

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
  std::iota(n.begin(), n.end(), T(1));
  std::vector<T> nab(n.size());
  std::ranges::transform(n, nab.begin(), [a, b](auto x)
                         { return static_cast<T>(2.0 * x + a + b); });

  std::vector<T> alpha(N);
  alpha.front() = nu;
  std::ranges::transform(
      nab, std::next(alpha.begin()), [a, b](auto nab)
      { return static_cast<T>((b * b - a * a) / (nab * (nab + 2.0))); });

  std::vector<T> beta(N);
  beta.front() = mu;
  std::ranges::transform(n, nab, std::next(beta.begin()),
                         [a, b](auto n, auto nab)
                         {
                           return static_cast<T>(
                               4 * (n + a) * (n + b) * n * (n + a + b)
                               / (nab * nab * (nab + 1.0) * (nab - 1.0)));
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

/// Evaluate the mth Jacobi polynomial and its first derivative, with
/// weight parameters (a, 0), at a single point x. Same recurrence as
/// compute_jacobi_deriv (nderiv = 1), but tracks only the last two
/// values of each sequence in scalars rather than allocating
/// (2, m+1, x.size()) arrays per Newton iteration.
template <std::floating_point T>
std::array<T, 2> jacobi_value_deriv(T a, std::size_t m, T x)
{
  if (m == 0)
    return {1.0, 0.0};

  T p0 = 1.0, d0 = 0.0;
  T p1 = (x * (a + 2.0) + a) * 0.5, d1 = a * 0.5 + 1;
  if (m == 1)
    return {p1, d1};

  for (std::size_t j = 2; j < m + 1; ++j)
  {
    const T a1 = 2 * j * (j + a) * (2 * j + a - 2);
    const T a2 = (2 * j + a - 1) * (a * a) / a1;
    const T a3 = (2 * j + a - 1) * (2 * j + a) / (2 * j * (j + a));
    const T a4 = 2 * (j + a - 1) * (j - 1) * (2 * j + a) / a1;
    const T p2 = p1 * (x * a3 + a2) - p0 * a4;
    const T d2 = d1 * (x * a3 + a2) - d0 * a4 + a3 * p1;
    p0 = p1;
    p1 = p2;
    d0 = d1;
    d1 = d2;
  }

  return {p1, d1};
}
//-----------------------------------------------------------------------------

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
      auto [f0, f1] = jacobi_value_deriv<T>(a, m, x[k]);
      T delta = f0 / (f1 - f0 * s);
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
  std::ranges::transform(wx, wx.begin(),
                         [](auto w) { return static_cast<T>(0.5 * w); });
  std::ranges::transform(ptx, ptx.begin(), [](auto x)
                         { return static_cast<T>(0.5 * (x + 1.0)); });
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
  std::ranges::transform(wx, wx.begin(),
                         [](auto w) { return static_cast<T>(0.5 * w); });
  std::ranges::transform(ptx, ptx.begin(), [](auto x)
                         { return static_cast<T>(0.5 * (x + 1.0)); });
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
  if (celltype == cell::type::triangle)
  {
    if (m == 1)
    {
      // Xiao Gimbutas, 3 points, degree 1
      std::vector<T> x = {0.3333333333333333, 0.3333333333333333};
      std::vector<T> w = {0.5};
      return {std::move(x), std::move(w)};
    }
    else if (m == 2)
    {
      // Xiao Gimbutas, 3 points, degree 2
      std::vector<T> x
          = {0.16666666666666666, 0.16666666666666666, 0.16666666666666666,
             0.6666666666666667,  0.6666666666666667,  0.16666666666666666};
      std::vector<T> w
          = {0.16666666666666666, 0.16666666666666666, 0.16666666666666666};
      return {std::move(x), std::move(w)};
    }
    else if (m == 3)
    {
      // Xiao Gimbutas, 3 points, degree 3
      std::vector<T> x
          = {0.4459484909159649,  0.4459484909159649, 0.09157621350977085,
             0.09157621350977085, 0.4459484909159649, 0.10810301816807022,
             0.09157621350977085, 0.8168475729804583, 0.10810301816807022,
             0.4459484909159649,  0.8168475729804583, 0.09157621350977085};
      std::vector<T> w
          = {0.11169079483900574, 0.05497587182766094, 0.11169079483900574,
             0.05497587182766094, 0.11169079483900574, 0.05497587182766094};
      return {std::move(x), std::move(w)};
    }
    else if (m == 4)
    {
      // Xiao Gimbutas, 3 points, degree 4
      std::vector<T> x
          = {0.4459484909159649,  0.4459484909159649, 0.09157621350977085,
             0.09157621350977085, 0.4459484909159649, 0.10810301816807022,
             0.09157621350977085, 0.8168475729804583, 0.10810301816807022,
             0.4459484909159649,  0.8168475729804583, 0.09157621350977085};
      std::vector<T> w
          = {0.11169079483900574, 0.05497587182766094, 0.11169079483900574,
             0.05497587182766094, 0.11169079483900574, 0.05497587182766094};
      return {std::move(x), std::move(w)};
    }
    else if (m == 5)
    {
      // Xiao Gimbutas, 3 points, degree 5
      std::vector<T> x
          = {0.3333333333333333,  0.3333333333333333,  0.1012865073234564,
             0.1012865073234564,  0.47014206410511505, 0.47014206410511505,
             0.1012865073234564,  0.7974269853530872,  0.47014206410511505,
             0.05971587178976989, 0.7974269853530872,  0.1012865073234564,
             0.05971587178976989, 0.47014206410511505};
      std::vector<T> w = {0.1125,
                          0.06296959027241357,
                          0.0661970763942531,
                          0.06296959027241357,
                          0.0661970763942531,
                          0.06296959027241357,
                          0.0661970763942531};
      return {std::move(x), std::move(w)};
    }
    else if (m == 6)
    {
      // Xiao Gimbutas, 3 points, degree 6
      std::vector<T> x
          = {0.21942998254978302,  0.21942998254978302,  0.48013796411221504,
             0.48013796411221504,  0.21942998254978302,  0.561140034900434,
             0.48013796411221504,  0.039724071775569914, 0.561140034900434,
             0.21942998254978302,  0.039724071775569914, 0.48013796411221504,
             0.019371724361240805, 0.14161901592396814,  0.8390092597147911,
             0.019371724361240805, 0.14161901592396814,  0.8390092597147911,
             0.14161901592396814,  0.019371724361240805, 0.8390092597147911,
             0.14161901592396814,  0.019371724361240805, 0.8390092597147911};
      std::vector<T> w
          = {0.08566656207649052, 0.04036554479651549, 0.08566656207649052,
             0.04036554479651549, 0.08566656207649052, 0.04036554479651549,
             0.02031727989683033, 0.02031727989683033, 0.02031727989683033,
             0.02031727989683033, 0.02031727989683033, 0.02031727989683033};
      return {std::move(x), std::move(w)};
    }
    else if (m == 7)
    {
      // Xiao Gimbutas, 3 points, degree 7
      std::vector<T> x
          = {0.47319565368925104,  0.47319565368925104,  0.057797640054506494,
             0.057797640054506494, 0.24166360639724743,  0.24166360639724743,
             0.47319565368925104,  0.05360869262149792,  0.057797640054506494,
             0.884404719890987,    0.24166360639724743,  0.5166727872055051,
             0.05360869262149792,  0.47319565368925104,  0.884404719890987,
             0.057797640054506494, 0.5166727872055051,   0.24166360639724743,
             0.046971206130085534, 0.2593390118657857,   0.6936897820041288,
             0.046971206130085534, 0.2593390118657857,   0.6936897820041288,
             0.2593390118657857,   0.046971206130085534, 0.6936897820041288,
             0.2593390118657857,   0.046971206130085534, 0.6936897820041288};
      std::vector<T> w
          = {0.02659041664838023,  0.020459085197028434, 0.06386262428056692,
             0.02659041664838023,  0.020459085197028434, 0.06386262428056692,
             0.02659041664838023,  0.020459085197028434, 0.06386262428056692,
             0.027877270270345547, 0.027877270270345547, 0.027877270270345547,
             0.027877270270345547, 0.027877270270345547, 0.027877270270345547};
      return {std::move(x), std::move(w)};
    }
    else if (m == 8)
    {
      // Xiao Gimbutas, 3 points, degree 8
      std::vector<T> x
          = {0.3333333333333333,   0.3333333333333333,  0.17056930775176027,
             0.17056930775176027,  0.4592925882927231,  0.4592925882927231,
             0.05054722831703107,  0.05054722831703107, 0.17056930775176027,
             0.6588613844964795,   0.4592925882927231,  0.08141482341455375,
             0.05054722831703107,  0.8989055433659379,  0.6588613844964795,
             0.17056930775176027,  0.08141482341455375, 0.4592925882927231,
             0.8989055433659379,   0.05054722831703107, 0.008394777409957675,
             0.26311282963463806,  0.7284923929554044,  0.008394777409957675,
             0.26311282963463806,  0.7284923929554044,  0.26311282963463806,
             0.008394777409957675, 0.7284923929554044,  0.26311282963463806,
             0.008394777409957675, 0.7284923929554044};
      std::vector<T> w
          = {0.0721578038388936,   0.05160868526735912,  0.04754581713364232,
             0.01622924881159904,  0.05160868526735912,  0.04754581713364232,
             0.01622924881159904,  0.05160868526735912,  0.04754581713364232,
             0.01622924881159904,  0.013615157087217498, 0.013615157087217498,
             0.013615157087217498, 0.013615157087217498, 0.013615157087217498,
             0.013615157087217498};
      return {std::move(x), std::move(w)};
    }
    else if (m == 9)
    {
      // Xiao Gimbutas, 3 points, degree 9
      std::vector<T> x
          = {0.3333333333333333,  0.3333333333333333,  0.4896825191987376,
             0.4896825191987376,  0.1882035356190328,  0.1882035356190328,
             0.43708959149293664, 0.43708959149293664, 0.04472951339445275,
             0.04472951339445275, 0.4896825191987376,  0.02063496160252476,
             0.1882035356190328,  0.6235929287619344,  0.43708959149293664,
             0.12582081701412673, 0.04472951339445275, 0.9105409732110945,
             0.02063496160252476, 0.4896825191987376,  0.6235929287619344,
             0.1882035356190328,  0.12582081701412673, 0.43708959149293664,
             0.9105409732110945,  0.04472951339445275, 0.0368384120547363,
             0.2219629891607657,  0.741198598784498,   0.0368384120547363,
             0.2219629891607657,  0.741198598784498,   0.2219629891607657,
             0.0368384120547363,  0.741198598784498,   0.2219629891607657,
             0.0368384120547363,  0.741198598784498};
      std::vector<T> w
          = {0.04856789814139942,  0.015667350113569536, 0.03982386946360513,
             0.03891377050238714,  0.012788837829349017, 0.015667350113569536,
             0.03982386946360513,  0.03891377050238714,  0.012788837829349017,
             0.015667350113569536, 0.03982386946360513,  0.03891377050238714,
             0.012788837829349017, 0.021641769688644688, 0.021641769688644688,
             0.021641769688644688, 0.021641769688644688, 0.021641769688644688,
             0.021641769688644688};
      return {std::move(x), std::move(w)};
    }
    else if (m == 10)
    {
      // Xiao Gimbutas, 3 points, degree 10
      std::vector<T> x
          = {0.3333333333333333,   0.3333333333333333,   0.4951734598011705,
             0.4951734598011705,   0.019139415242841296, 0.019139415242841296,
             0.18448501268524653,  0.18448501268524653,  0.42823482094371884,
             0.42823482094371884,  0.4951734598011705,   0.009653080397658997,
             0.019139415242841296, 0.9617211695143174,   0.18448501268524653,
             0.6310299746295069,   0.42823482094371884,  0.14353035811256232,
             0.009653080397658997, 0.4951734598011705,   0.9617211695143174,
             0.019139415242841296, 0.6310299746295069,   0.18448501268524653,
             0.14353035811256232,  0.42823482094371884,  0.03472362048232748,
             0.13373475510086913,  0.03758272734119169,  0.3266931362813369,
             0.8315416244168035,   0.03472362048232748,  0.6357241363774714,
             0.03758272734119169,  0.13373475510086913,  0.8315416244168035,
             0.3266931362813369,   0.6357241363774714,   0.13373475510086913,
             0.03472362048232748,  0.3266931362813369,   0.03758272734119169,
             0.8315416244168035,   0.13373475510086913,  0.6357241363774714,
             0.3266931362813369,   0.03472362048232748,  0.8315416244168035,
             0.03758272734119169,  0.6357241363774714};
      std::vector<T> w
          = {0.041807437186986963, 0.004896295249209152, 0.003192679615059327,
             0.039316884873188636, 0.03762366398427199,  0.004896295249209152,
             0.003192679615059327, 0.039316884873188636, 0.03762366398427199,
             0.004896295249209152, 0.003192679615059327, 0.039316884873188636,
             0.03762366398427199,  0.014481140731628171, 0.019369524543009452,
             0.014481140731628171, 0.019369524543009452, 0.014481140731628171,
             0.019369524543009452, 0.014481140731628171, 0.019369524543009452,
             0.014481140731628171, 0.019369524543009452, 0.014481140731628171,
             0.019369524543009452};
      return {std::move(x), std::move(w)};
    }
    else if (m == 11)
    {
      // Xiao Gimbutas, 3 points, degree 11
      std::vector<T> x
          = {0.3333333333333333,    0.3333333333333333,   0.030846895635588123,
             0.030846895635588123,  0.49878016517846074,  0.49878016517846074,
             0.11320782728669404,   0.11320782728669404,  0.4366550163931761,
             0.4366550163931761,    0.21448345861926937,  0.21448345861926937,
             0.030846895635588123,  0.9383062087288238,   0.49878016517846074,
             0.0024396696430785125, 0.11320782728669404,  0.7735843454266119,
             0.4366550163931761,    0.12668996721364778,  0.21448345861926937,
             0.5710330827614613,    0.9383062087288238,   0.030846895635588123,
             0.0024396696430785125, 0.49878016517846074,  0.7735843454266119,
             0.11320782728669404,   0.12668996721364778,  0.4366550163931761,
             0.5710330827614613,    0.21448345861926937,  0.014366662569555624,
             0.1593036198376935,    0.04766406697215078,  0.31063121631346313,
             0.8263297175927509,    0.014366662569555624, 0.6417047167143861,
             0.04766406697215078,   0.1593036198376935,   0.8263297175927509,
             0.31063121631346313,   0.6417047167143861,   0.1593036198376935,
             0.014366662569555624,  0.31063121631346313,  0.04766406697215078,
             0.8263297175927509,    0.1593036198376935,   0.6417047167143861,
             0.31063121631346313,   0.014366662569555624, 0.8263297175927509,
             0.04766406697215078,   0.6417047167143861};
      std::vector<T> w = {
          0.040722567354675644,  0.006124648475353982,  0.0062327459369406904,
          0.02006462119065416,   0.031547436079949344,  0.033922553871847574,
          0.006124648475353982,  0.0062327459369406904, 0.02006462119065416,
          0.031547436079949344,  0.033922553871847574,  0.006124648475353982,
          0.0062327459369406904, 0.02006462119065416,   0.031547436079949344,
          0.033922553871847574,  0.007278811668904623,  0.020321424327943236,
          0.007278811668904623,  0.020321424327943236,  0.007278811668904623,
          0.020321424327943236,  0.007278811668904623,  0.020321424327943236,
          0.007278811668904623,  0.020321424327943236,  0.007278811668904623,
          0.020321424327943236};
      return {std::move(x), std::move(w)};
    }
    else if (m == 12)
    {
      // Xiao Gimbutas, 3 points, degree 12
      std::vector<T> x
          = {0.27146250701492614,  0.27146250701492614,  0.10925782765935432,
             0.10925782765935432,  0.4401116486585931,   0.4401116486585931,
             0.4882037509455415,   0.4882037509455415,   0.02464636343633564,
             0.02464636343633564,  0.27146250701492614,  0.45707498597014773,
             0.10925782765935432,  0.7814843446812914,   0.4401116486585931,
             0.11977670268281382,  0.4882037509455415,   0.02359249810891695,
             0.02464636343633564,  0.9507072731273287,   0.45707498597014773,
             0.27146250701492614,  0.7814843446812914,   0.10925782765935432,
             0.11977670268281382,  0.4401116486585931,   0.02359249810891695,
             0.4882037509455415,   0.9507072731273287,   0.02464636343633564,
             0.1162960196779266,   0.25545422863851736,  0.021382490256170623,
             0.12727971723358936,  0.023034156355267166, 0.29165567973834094,
             0.6282497516835561,   0.1162960196779266,   0.85133779251024,
             0.021382490256170623, 0.6853101639063919,   0.023034156355267166,
             0.25545422863851736,  0.6282497516835561,   0.12727971723358936,
             0.85133779251024,     0.29165567973834094,  0.6853101639063919,
             0.25545422863851736,  0.1162960196779266,   0.12727971723358936,
             0.021382490256170623, 0.29165567973834094,  0.023034156355267166,
             0.6282497516835561,   0.25545422863851736,  0.85133779251024,
             0.12727971723358936,  0.6853101639063919,   0.29165567973834094,
             0.1162960196779266,   0.6282497516835561,   0.021382490256170623,
             0.85133779251024,     0.023034156355267166, 0.6853101639063919};
      std::vector<T> w = {
          0.03127060659795138,   0.014243026034438775,  0.024959167464030475,
          0.012133419040726017,  0.0039658212549868194, 0.03127060659795138,
          0.014243026034438775,  0.024959167464030475,  0.012133419040726017,
          0.0039658212549868194, 0.03127060659795138,   0.014243026034438775,
          0.024959167464030475,  0.012133419040726017,  0.0039658212549868194,
          0.021613681829707104,  0.007541838788255721,  0.01089179251930378,
          0.021613681829707104,  0.007541838788255721,  0.01089179251930378,
          0.021613681829707104,  0.007541838788255721,  0.01089179251930378,
          0.021613681829707104,  0.007541838788255721,  0.01089179251930378,
          0.021613681829707104,  0.007541838788255721,  0.01089179251930378,
          0.021613681829707104,  0.007541838788255721,  0.01089179251930378};
      return {std::move(x), std::move(w)};
    }
    else if (m == 13)
    {
      // Xiao Gimbutas, 3 points, degree 13
      std::vector<T> x
          = {0.3333333333333333,   0.3333333333333333,   0.4961358947410461,
             0.4961358947410461,   0.4696086896534919,   0.4696086896534919,
             0.23111028494908226,  0.23111028494908226,  0.4144775702790546,
             0.4144775702790546,   0.11355991257213327,  0.11355991257213327,
             0.024895931491216494, 0.024895931491216494, 0.4961358947410461,
             0.007728210517907841, 0.4696086896534919,   0.06078262069301621,
             0.23111028494908226,  0.5377794301018355,   0.4144775702790546,
             0.17104485944189085,  0.11355991257213327,  0.7728801748557335,
             0.024895931491216494, 0.950208137017567,    0.007728210517907841,
             0.4961358947410461,   0.06078262069301621,  0.4696086896534919,
             0.5377794301018355,   0.23111028494908226,  0.17104485944189085,
             0.4144775702790546,   0.7728801748557335,   0.11355991257213327,
             0.950208137017567,    0.024895931491216494, 0.01898800438375904,
             0.2920786885766364,   0.09773603106601653,  0.26674525331035115,
             0.021966344206529244, 0.1267997757838373,   0.6889333070396046,
             0.01898800438375904,  0.6355187156236324,   0.09773603106601653,
             0.8512338800096335,   0.021966344206529244, 0.2920786885766364,
             0.6889333070396046,   0.26674525331035115,  0.6355187156236324,
             0.1267997757838373,   0.8512338800096335,   0.2920786885766364,
             0.01898800438375904,  0.26674525331035115,  0.09773603106601653,
             0.1267997757838373,   0.021966344206529244, 0.6889333070396046,
             0.2920786885766364,   0.6355187156236324,   0.26674525331035115,
             0.8512338800096335,   0.1267997757838373,   0.01898800438375904,
             0.6889333070396046,   0.09773603106601653,  0.6355187156236324,
             0.021966344206529244, 0.8512338800096335};
      std::vector<T> w
          = {0.02581132333214541,   0.004970738180536294, 0.01639062080186149,
             0.023031204796389124,  0.0234735477710776,   0.015451548987879897,
             0.0040146998976292115, 0.004970738180536294, 0.01639062080186149,
             0.023031204796389124,  0.0234735477710776,   0.015451548987879897,
             0.0040146998976292115, 0.004970738180536294, 0.01639062080186149,
             0.023031204796389124,  0.0234735477710776,   0.015451548987879897,
             0.0040146998976292115, 0.00906274932310044,  0.018605980228630768,
             0.007696536341891089,  0.00906274932310044,  0.018605980228630768,
             0.007696536341891089,  0.00906274932310044,  0.018605980228630768,
             0.007696536341891089,  0.00906274932310044,  0.018605980228630768,
             0.007696536341891089,  0.00906274932310044,  0.018605980228630768,
             0.007696536341891089,  0.00906274932310044,  0.018605980228630768,
             0.007696536341891089};
      return {std::move(x), std::move(w)};
    }
    else if (m == 14)
    {
      // Xiao Gimbutas, 3 points, degree 14
      std::vector<T> x
          = {0.41764471934045394,  0.41764471934045394,  0.0617998830908727,
             0.0617998830908727,   0.2734775283088387,   0.2734775283088387,
             0.1772055324125435,   0.1772055324125435,   0.0193909612487011,
             0.0193909612487011,   0.4889639103621786,   0.4889639103621786,
             0.41764471934045394,  0.16471056131909212,  0.0617998830908727,
             0.8764002338182546,   0.2734775283088387,   0.4530449433823226,
             0.1772055324125435,   0.645588935174913,    0.0193909612487011,
             0.9612180775025978,   0.4889639103621786,   0.022072179275642756,
             0.16471056131909212,  0.41764471934045394,  0.8764002338182546,
             0.0617998830908727,   0.4530449433823226,   0.2734775283088387,
             0.645588935174913,    0.1772055324125435,   0.9612180775025978,
             0.0193909612487011,   0.022072179275642756, 0.4889639103621786,
             0.014646950055654471, 0.29837288213625773,  0.09291624935697185,
             0.336861459796345,    0.05712475740364799,  0.17226668782135557,
             0.001268330932872076, 0.11897449769695682,  0.6869801678080878,
             0.014646950055654471, 0.5702222908466832,   0.09291624935697185,
             0.7706085547749965,   0.05712475740364799,  0.8797571713701712,
             0.001268330932872076, 0.29837288213625773,  0.6869801678080878,
             0.336861459796345,    0.5702222908466832,   0.17226668782135557,
             0.7706085547749965,   0.11897449769695682,  0.8797571713701712,
             0.29837288213625773,  0.014646950055654471, 0.336861459796345,
             0.09291624935697185,  0.17226668782135557,  0.05712475740364799,
             0.11897449769695682,  0.001268330932872076, 0.6869801678080878,
             0.29837288213625773,  0.5702222908466832,   0.336861459796345,
             0.7706085547749965,   0.17226668782135557,  0.8797571713701712,
             0.11897449769695682,  0.014646950055654471, 0.6869801678080878,
             0.09291624935697185,  0.5702222908466832,   0.05712475740364799,
             0.7706085547749965,   0.001268330932872076, 0.8797571713701712};
      std::vector<T> w
          = {0.016394176772062678, 0.007216849834888334, 0.025887052253645793,
             0.02108129436849651,  0.002461701801200041, 0.010941790684714446,
             0.016394176772062678, 0.007216849834888334, 0.025887052253645793,
             0.02108129436849651,  0.002461701801200041, 0.010941790684714446,
             0.016394176772062678, 0.007216849834888334, 0.025887052253645793,
             0.02108129436849651,  0.002461701801200041, 0.010941790684714446,
             0.007218154056766921, 0.019285755393530342, 0.012332876606281839,
             0.002505114419250336, 0.007218154056766921, 0.019285755393530342,
             0.012332876606281839, 0.002505114419250336, 0.007218154056766921,
             0.019285755393530342, 0.012332876606281839, 0.002505114419250336,
             0.007218154056766921, 0.019285755393530342, 0.012332876606281839,
             0.002505114419250336, 0.007218154056766921, 0.019285755393530342,
             0.012332876606281839, 0.002505114419250336, 0.007218154056766921,
             0.019285755393530342, 0.012332876606281839, 0.002505114419250336};
      return {std::move(x), std::move(w)};
    }
    else if (m == 15)
    {
      // Xiao Gimbutas, 3 points, degree 15
      std::vector<T> x = {
          0.3333333333333333,    0.3333333333333333,    0.1299782299330779,
          0.1299782299330779,    0.4600769492970597,    0.4600769492970597,
          0.4916858166302972,    0.4916858166302972,    0.22153234079514206,
          0.22153234079514206,   0.39693373740906057,   0.39693373740906057,
          0.0563419176961002,    0.0563419176961002,    0.1299782299330779,
          0.7400435401338442,    0.4600769492970597,    0.07984610140588055,
          0.4916858166302972,    0.016628366739405598,  0.22153234079514206,
          0.5569353184097159,    0.39693373740906057,   0.20613252518187886,
          0.0563419176961002,    0.8873161646077996,    0.7400435401338442,
          0.1299782299330779,    0.07984610140588055,   0.4600769492970597,
          0.016628366739405598,  0.4916858166302972,    0.5569353184097159,
          0.22153234079514206,   0.20613252518187886,   0.39693373740906057,
          0.8873161646077996,    0.0563419176961002,    0.08459422148219181,
          0.18232178340719132,   0.016027089786345473,  0.15020038406523872,
          0.09765044243024235,   0.32311131516371266,   0.018454251904633165,
          0.3079476814836729,    0.0011135352740137417, 0.03803522930110929,
          0.733083995110617,     0.08459422148219181,   0.8337725261484158,
          0.016027089786345473,  0.5792382424060449,    0.09765044243024235,
          0.673598066611694,     0.018454251904633165,  0.960851235424877,
          0.0011135352740137417, 0.18232178340719132,   0.733083995110617,
          0.15020038406523872,   0.8337725261484158,    0.32311131516371266,
          0.5792382424060449,    0.3079476814836729,    0.673598066611694,
          0.03803522930110929,   0.960851235424877,     0.18232178340719132,
          0.08459422148219181,   0.15020038406523872,   0.016027089786345473,
          0.32311131516371266,   0.09765044243024235,   0.3079476814836729,
          0.018454251904633165,  0.03803522930110929,   0.0011135352740137417,
          0.733083995110617,     0.18232178340719132,   0.8337725261484158,
          0.15020038406523872,   0.5792382424060449,    0.32311131516371266,
          0.673598066611694,     0.3079476814836729,    0.960851235424877,
          0.03803522930110929,   0.08459422148219181,   0.733083995110617,
          0.016027089786345473,  0.8337725261484158,    0.09765044243024235,
          0.5792382424060449,    0.018454251904633165,  0.673598066611694,
          0.0011135352740137417, 0.960851235424877};
      std::vector<T> w = {
          0.01486520987403566,   0.00369875203352305,   0.010797043968219226,
          0.0079161381750109,    0.023143643052599038,  0.023168020695603617,
          0.007542237123798534,  0.00369875203352305,   0.010797043968219226,
          0.0079161381750109,    0.023143643052599038,  0.023168020695603617,
          0.007542237123798534,  0.00369875203352305,   0.010797043968219226,
          0.0079161381750109,    0.023143643052599038,  0.023168020695603617,
          0.007542237123798534,  0.012115004391562803,  0.00561425214943903,
          0.015537610235255475,  0.008218381046413948,  0.0012376330072789582,
          0.012115004391562803,  0.00561425214943903,   0.015537610235255475,
          0.008218381046413948,  0.0012376330072789582, 0.012115004391562803,
          0.00561425214943903,   0.015537610235255475,  0.008218381046413948,
          0.0012376330072789582, 0.012115004391562803,  0.00561425214943903,
          0.015537610235255475,  0.008218381046413948,  0.0012376330072789582,
          0.012115004391562803,  0.00561425214943903,   0.015537610235255475,
          0.008218381046413948,  0.0012376330072789582, 0.012115004391562803,
          0.00561425214943903,   0.015537610235255475,  0.008218381046413948,
          0.0012376330072789582};
      return {std::move(x), std::move(w)};
    }
    else if (m == 16)
    {
      // Xiao Gimbutas, 3 points, degree 16
      std::vector<T> x
          = {0.3333333333333333,    0.3333333333333333,    0.06667447224023837,
             0.06667447224023837,   0.24132168070137838,   0.24132168070137838,
             0.41279809595522365,   0.41279809595522365,   0.15006373658703515,
             0.15006373658703515,   0.46954803099668496,   0.46954803099668496,
             0.017041629405718517,  0.017041629405718517,  0.06667447224023837,
             0.8666510555195233,    0.24132168070137838,   0.5173566385972432,
             0.41279809595522365,   0.1744038080895527,    0.15006373658703515,
             0.6998725268259297,    0.46954803099668496,   0.060903938006630076,
             0.017041629405718517,  0.965916741188563,     0.8666510555195233,
             0.06667447224023837,   0.5173566385972432,    0.24132168070137838,
             0.1744038080895527,    0.41279809595522365,   0.6998725268259297,
             0.15006373658703515,   0.060903938006630076,  0.46954803099668496,
             0.965916741188563,     0.017041629405718517,  0.009664954403660254,
             0.41376948582708517,   0.030305943355186365,  0.30417944822947973,
             0.010812972776103751,  0.08960908902270585,   0.10665316053614844,
             0.29661537240038294,   0.051354315344013114,  0.16976335515028973,
             0.0036969427073556124, 0.21404877992584728,   0.5765655597692546,
             0.009664954403660254,  0.6655146084153339,    0.030305943355186365,
             0.8995779382011905,    0.010812972776103751,  0.5967314670634686,
             0.10665316053614844,   0.7788823295056971,    0.051354315344013114,
             0.7822542773667971,    0.0036969427073556124, 0.41376948582708517,
             0.5765655597692546,    0.30417944822947973,   0.6655146084153339,
             0.08960908902270585,   0.8995779382011905,    0.29661537240038294,
             0.5967314670634686,    0.16976335515028973,   0.7788823295056971,
             0.21404877992584728,   0.7822542773667971,    0.41376948582708517,
             0.009664954403660254,  0.30417944822947973,   0.030305943355186365,
             0.08960908902270585,   0.010812972776103751,  0.29661537240038294,
             0.10665316053614844,   0.16976335515028973,   0.051354315344013114,
             0.21404877992584728,   0.0036969427073556124, 0.5765655597692546,
             0.41376948582708517,   0.6655146084153339,    0.30417944822947973,
             0.8995779382011905,    0.08960908902270585,   0.5967314670634686,
             0.29661537240038294,   0.7788823295056971,    0.16976335515028973,
             0.7822542773667971,    0.21404877992584728,   0.009664954403660254,
             0.5765655597692546,    0.030305943355186365,  0.6655146084153339,
             0.010812972776103751,  0.8995779382011905,    0.10665316053614844,
             0.5967314670634686,    0.051354315344013114,  0.7788823295056971,
             0.0036969427073556124, 0.7822542773667971};
      std::vector<T> w
          = {0.023113955157095672,  0.006212712797780504, 0.020592020534896276,
             0.020492609893407683,  0.014391748351374455, 0.013546834733855226,
             0.001894567619132111,  0.006212712797780504, 0.020592020534896276,
             0.020492609893407683,  0.014391748351374455, 0.013546834733855226,
             0.001894567619132111,  0.006212712797780504, 0.020592020534896276,
             0.020492609893407683,  0.014391748351374455, 0.013546834733855226,
             0.001894567619132111,  0.004091105276611069, 0.006991803562326784,
             0.0028759349852485795, 0.015823030840991622, 0.008826540523551642,
             0.0023073453198645673, 0.004091105276611069, 0.006991803562326784,
             0.0028759349852485795, 0.015823030840991622, 0.008826540523551642,
             0.0023073453198645673, 0.004091105276611069, 0.006991803562326784,
             0.0028759349852485795, 0.015823030840991622, 0.008826540523551642,
             0.0023073453198645673, 0.004091105276611069, 0.006991803562326784,
             0.0028759349852485795, 0.015823030840991622, 0.008826540523551642,
             0.0023073453198645673, 0.004091105276611069, 0.006991803562326784,
             0.0028759349852485795, 0.015823030840991622, 0.008826540523551642,
             0.0023073453198645673, 0.004091105276611069, 0.006991803562326784,
             0.0028759349852485795, 0.015823030840991622, 0.008826540523551642,
             0.0023073453198645673};
      return {std::move(x), std::move(w)};
    }
    else if (m == 17)
    {
      // Xiao Gimbutas, 3 points, degree 17
      std::vector<T> x
          = {0.4171034443615992,   0.4171034443615992,   0.18035811626637066,
             0.18035811626637066,  0.2857065024365867,   0.2857065024365867,
             0.06665406347959701,  0.06665406347959701,  0.014755491660754072,
             0.014755491660754072, 0.46559787161889027,  0.46559787161889027,
             0.4171034443615992,   0.16579311127680163,  0.18035811626637066,
             0.6392837674672587,   0.2857065024365867,   0.42858699512682663,
             0.06665406347959701,  0.866691873040806,    0.014755491660754072,
             0.9704890166784919,   0.46559787161889027,  0.06880425676221946,
             0.16579311127680163,  0.4171034443615992,   0.6392837674672587,
             0.18035811626637066,  0.42858699512682663,  0.2857065024365867,
             0.866691873040806,    0.06665406347959701,  0.9704890166784919,
             0.014755491660754072, 0.06880425676221946,  0.46559787161889027,
             0.011575175903180683, 0.07250547079900238,  0.013229672760086951,
             0.41547545929522905,  0.013135870834002753, 0.27179187005535477,
             0.15750547792686992,  0.29921894247697034,  0.06734937786736123,
             0.3062815917461865,   0.07804234056828245,  0.16872251349525944,
             0.016017642362119337, 0.15919228747279268,  0.9159193532978169,
             0.011575175903180683, 0.5712948679446841,   0.013229672760086951,
             0.7150722591106424,   0.013135870834002753, 0.5432755795961598,
             0.15750547792686992,  0.6263690303864522,   0.06734937786736123,
             0.7532351459364581,   0.07804234056828245,  0.824790070165088,
             0.016017642362119337, 0.07250547079900238,  0.9159193532978169,
             0.41547545929522905,  0.5712948679446841,   0.27179187005535477,
             0.7150722591106424,   0.29921894247697034,  0.5432755795961598,
             0.3062815917461865,   0.6263690303864522,   0.16872251349525944,
             0.7532351459364581,   0.15919228747279268,  0.824790070165088,
             0.07250547079900238,  0.011575175903180683, 0.41547545929522905,
             0.013229672760086951, 0.27179187005535477,  0.013135870834002753,
             0.29921894247697034,  0.15750547792686992,  0.3062815917461865,
             0.06734937786736123,  0.16872251349525944,  0.07804234056828245,
             0.15919228747279268,  0.016017642362119337, 0.9159193532978169,
             0.07250547079900238,  0.5712948679446841,   0.41547545929522905,
             0.7150722591106424,   0.27179187005535477,  0.5432755795961598,
             0.29921894247697034,  0.6263690303864522,   0.3062815917461865,
             0.7532351459364581,   0.16872251349525944,  0.824790070165088,
             0.15919228747279268,  0.011575175903180683, 0.9159193532978169,
             0.013229672760086951, 0.5712948679446841,   0.013135870834002753,
             0.7150722591106424,   0.15750547792686992,  0.5432755795961598,
             0.06734937786736123,  0.6263690303864522,   0.07804234056828245,
             0.7532351459364581,   0.016017642362119337, 0.824790070165088};
      std::vector<T> w
          = {0.013655463264051053, 0.013156315294008993, 0.01885811857639764,
             0.006229500401152722, 0.001386943788818821, 0.01250972547524868,
             0.013655463264051053, 0.013156315294008993, 0.01885811857639764,
             0.006229500401152722, 0.001386943788818821, 0.01250972547524868,
             0.013655463264051053, 0.013156315294008993, 0.01885811857639764,
             0.006229500401152722, 0.001386943788818821, 0.01250972547524868,
             0.002292174200867934, 0.005199219977919768, 0.004346107250500596,
             0.013085812967668494, 0.011243886273345534, 0.01027894916022726,
             0.003989150102964797, 0.002292174200867934, 0.005199219977919768,
             0.004346107250500596, 0.013085812967668494, 0.011243886273345534,
             0.01027894916022726,  0.003989150102964797, 0.002292174200867934,
             0.005199219977919768, 0.004346107250500596, 0.013085812967668494,
             0.011243886273345534, 0.01027894916022726,  0.003989150102964797,
             0.002292174200867934, 0.005199219977919768, 0.004346107250500596,
             0.013085812967668494, 0.011243886273345534, 0.01027894916022726,
             0.003989150102964797, 0.002292174200867934, 0.005199219977919768,
             0.004346107250500596, 0.013085812967668494, 0.011243886273345534,
             0.01027894916022726,  0.003989150102964797, 0.002292174200867934,
             0.005199219977919768, 0.004346107250500596, 0.013085812967668494,
             0.011243886273345534, 0.01027894916022726,  0.003989150102964797};
      return {std::move(x), std::move(w)};
    }
    else if (m == 18)
    {
      // Xiao Gimbutas, 3 points, degree 18
      std::vector<T> x
          = {0.3333333333333333,   0.3333333333333333,    0.4749182113240457,
             0.4749182113240457,   0.15163850697260495,   0.15163850697260495,
             0.4110671018759195,   0.4110671018759195,    0.2656146099053742,
             0.2656146099053742,   0.0037589443410684376, 0.0037589443410684376,
             0.072438705567333,    0.072438705567333,     0.4749182113240457,
             0.05016357735190857,  0.15163850697260495,   0.6967229860547901,
             0.4110671018759195,   0.177865796248161,     0.2656146099053742,
             0.46877078018925156,  0.0037589443410684376, 0.9924821113178631,
             0.072438705567333,    0.855122588865334,     0.05016357735190857,
             0.4749182113240457,   0.6967229860547901,    0.15163850697260495,
             0.177865796248161,    0.4110671018759195,    0.46877078018925156,
             0.2656146099053742,   0.9924821113178631,    0.0037589443410684376,
             0.855122588865334,    0.072438705567333,     0.09042704035434063,
             0.3850440344131637,   0.012498932483495477,  0.04727614183265175,
             0.05401173533902428,  0.30206195771287075,   0.010505018819241962,
             0.2565061597742415,   0.06612245802840343,   0.17847912556588763,
             0.14906691012577386,  0.2685733063960138,    0.011691824674667157,
             0.41106566867461836,  0.014331524778941987,  0.1327788302713893,
             0.5245289252324957,   0.09042704035434063,   0.9402249256838529,
             0.012498932483495477, 0.6439263069481049,    0.05401173533902428,
             0.7329888214065166,   0.010505018819241962,  0.7553984164057089,
             0.06612245802840343,  0.5823597834782124,    0.14906691012577386,
             0.5772425066507145,   0.011691824674667157,  0.8528896449496688,
             0.014331524778941987, 0.3850440344131637,    0.5245289252324957,
             0.04727614183265175,  0.9402249256838529,    0.30206195771287075,
             0.6439263069481049,   0.2565061597742415,    0.7329888214065166,
             0.17847912556588763,  0.7553984164057089,    0.2685733063960138,
             0.5823597834782124,   0.41106566867461836,   0.5772425066507145,
             0.1327788302713893,   0.8528896449496688,    0.3850440344131637,
             0.09042704035434063,  0.04727614183265175,   0.012498932483495477,
             0.30206195771287075,  0.05401173533902428,   0.2565061597742415,
             0.010505018819241962, 0.17847912556588763,   0.06612245802840343,
             0.2685733063960138,   0.14906691012577386,   0.41106566867461836,
             0.011691824674667157, 0.1327788302713893,    0.014331524778941987,
             0.5245289252324957,   0.3850440344131637,    0.9402249256838529,
             0.04727614183265175,  0.6439263069481049,    0.30206195771287075,
             0.7329888214065166,   0.2565061597742415,    0.7553984164057089,
             0.17847912556588763,  0.5823597834782124,    0.2685733063960138,
             0.5772425066507145,   0.41106566867461836,   0.8528896449496688,
             0.1327788302713893,   0.09042704035434063,   0.5245289252324957,
             0.012498932483495477, 0.9402249256838529,    0.05401173533902428,
             0.6439263069481049,   0.010505018819241962,  0.7329888214065166,
             0.06612245802840343,  0.7553984164057089,    0.14906691012577386,
             0.5823597834782124,   0.011691824674667157,  0.5772425066507145,
             0.014331524778941987, 0.8528896449496688};
      std::vector<T> w = {
          0.01537426061955793,   0.006553513745869378,  0.0101591694227292,
          0.01673599702992395,   0.015558198301003067,  0.0002660028084738903,
          0.006895143302383471,  0.006553513745869378,  0.0101591694227292,
          0.01673599702992395,   0.015558198301003067,  0.0002660028084738903,
          0.006895143302383471,  0.006553513745869378,  0.0101591694227292,
          0.01673599702992395,   0.015558198301003067,  0.0002660028084738903,
          0.006895143302383471,  0.007664129097276571,  0.0021087583873722216,
          0.008182954206993283,  0.0038649176400031137, 0.00845582695874004,
          0.01379644324428974,   0.004793062237180752,  0.0038208524863598183,
          0.007664129097276571,  0.0021087583873722216, 0.008182954206993283,
          0.0038649176400031137, 0.00845582695874004,   0.01379644324428974,
          0.004793062237180752,  0.0038208524863598183, 0.007664129097276571,
          0.0021087583873722216, 0.008182954206993283,  0.0038649176400031137,
          0.00845582695874004,   0.01379644324428974,   0.004793062237180752,
          0.0038208524863598183, 0.007664129097276571,  0.0021087583873722216,
          0.008182954206993283,  0.0038649176400031137, 0.00845582695874004,
          0.01379644324428974,   0.004793062237180752,  0.0038208524863598183,
          0.007664129097276571,  0.0021087583873722216, 0.008182954206993283,
          0.0038649176400031137, 0.00845582695874004,   0.01379644324428974,
          0.004793062237180752,  0.0038208524863598183, 0.007664129097276571,
          0.0021087583873722216, 0.008182954206993283,  0.0038649176400031137,
          0.00845582695874004,   0.01379644324428974,   0.004793062237180752,
          0.0038208524863598183};
      return {std::move(x), std::move(w)};
    }
    else if (m == 19)
    {
      // Xiao Gimbutas, 3 points, degree 19
      std::vector<T> x = {
          0.3333333333333333,    0.3333333333333333,    0.05252627985410363,
          0.05252627985410363,   0.11144805571699878,   0.11144805571699878,
          0.011639027327922657,  0.011639027327922657,  0.25516213315312486,
          0.25516213315312486,   0.4039697179663861,    0.4039697179663861,
          0.17817100607962755,   0.17817100607962755,   0.4591943889568276,
          0.4591943889568276,    0.4925124498658742,    0.4925124498658742,
          0.05252627985410363,   0.8949474402917927,    0.11144805571699878,
          0.7771038885660024,    0.011639027327922657,  0.9767219453441547,
          0.25516213315312486,   0.4896757336937503,    0.4039697179663861,
          0.19206056406722782,   0.17817100607962755,   0.6436579878407449,
          0.4591943889568276,    0.08161122208634475,   0.4925124498658742,
          0.014975100268251551,  0.8949474402917927,    0.05252627985410363,
          0.7771038885660024,    0.11144805571699878,   0.9767219453441547,
          0.011639027327922657,  0.4896757336937503,    0.25516213315312486,
          0.19206056406722782,   0.4039697179663861,    0.6436579878407449,
          0.17817100607962755,   0.08161122208634475,   0.4591943889568276,
          0.014975100268251551,  0.4925124498658742,    0.005005142352350433,
          0.1424222825711269,    0.009777061438676854,  0.06008389996270236,
          0.039142449434608845,  0.13070066996053453,   0.129312809767979,
          0.31131838322398686,   0.07456118930435514,   0.22143394188911344,
          0.04088831446497813,   0.3540259269997119,    0.014923638907438481,
          0.24189410400689262,   0.0020691038491023883, 0.36462041433871,
          0.8525725750765227,    0.005005142352350433,  0.9301390385986208,
          0.009777061438676854,  0.8301568806048566,    0.039142449434608845,
          0.5593688070080342,    0.129312809767979,     0.7040048688065313,
          0.07456118930435514,   0.60508575853531,      0.04088831446497813,
          0.7431822570856689,    0.014923638907438481,  0.6333104818121875,
          0.0020691038491023883, 0.1424222825711269,    0.8525725750765227,
          0.06008389996270236,   0.9301390385986208,    0.13070066996053453,
          0.8301568806048566,    0.31131838322398686,   0.5593688070080342,
          0.22143394188911344,   0.7040048688065313,    0.3540259269997119,
          0.60508575853531,      0.24189410400689262,   0.7431822570856689,
          0.36462041433871,      0.6333104818121875,    0.1424222825711269,
          0.005005142352350433,  0.06008389996270236,   0.009777061438676854,
          0.13070066996053453,   0.039142449434608845,  0.31131838322398686,
          0.129312809767979,     0.22143394188911344,   0.07456118930435514,
          0.3540259269997119,    0.04088831446497813,   0.24189410400689262,
          0.014923638907438481,  0.36462041433871,      0.0020691038491023883,
          0.8525725750765227,    0.1424222825711269,    0.9301390385986208,
          0.06008389996270236,   0.8301568806048566,    0.13070066996053453,
          0.5593688070080342,    0.31131838322398686,   0.7040048688065313,
          0.22143394188911344,   0.60508575853531,      0.3540259269997119,
          0.7431822570856689,    0.24189410400689262,   0.6333104818121875,
          0.36462041433871,      0.005005142352350433,  0.8525725750765227,
          0.009777061438676854,  0.9301390385986208,    0.039142449434608845,
          0.8301568806048566,    0.129312809767979,     0.5593688070080342,
          0.07456118930435514,   0.7040048688065313,    0.04088831446497813,
          0.60508575853531,      0.014923638907438481,  0.7431822570856689,
          0.0020691038491023883, 0.6333104818121875};
      std::vector<T> w = {
          0.017234580425452638,  0.0035546968113974735, 0.007617478258502418,
          0.0008825962091542701, 0.01587642729376499,   0.01576867932261981,
          0.012325990526792415,  0.011491785488561626,  0.005160941091209432,
          0.0035546968113974735, 0.007617478258502418,  0.0008825962091542701,
          0.01587642729376499,   0.01576867932261981,   0.012325990526792415,
          0.011491785488561626,  0.005160941091209432,  0.0035546968113974735,
          0.007617478258502418,  0.0008825962091542701, 0.01587642729376499,
          0.01576867932261981,   0.012325990526792415,  0.011491785488561626,
          0.005160941091209432,  0.0014628462439400358, 0.0016636944202969523,
          0.004847759540812101,  0.013173132353722682,  0.009054037295215252,
          0.008051104730469714,  0.00422796241954674,   0.0016410687574198689,
          0.0014628462439400358, 0.0016636944202969523, 0.004847759540812101,
          0.013173132353722682,  0.009054037295215252,  0.008051104730469714,
          0.00422796241954674,   0.0016410687574198689, 0.0014628462439400358,
          0.0016636944202969523, 0.004847759540812101,  0.013173132353722682,
          0.009054037295215252,  0.008051104730469714,  0.00422796241954674,
          0.0016410687574198689, 0.0014628462439400358, 0.0016636944202969523,
          0.004847759540812101,  0.013173132353722682,  0.009054037295215252,
          0.008051104730469714,  0.00422796241954674,   0.0016410687574198689,
          0.0014628462439400358, 0.0016636944202969523, 0.004847759540812101,
          0.013173132353722682,  0.009054037295215252,  0.008051104730469714,
          0.00422796241954674,   0.0016410687574198689, 0.0014628462439400358,
          0.0016636944202969523, 0.004847759540812101,  0.013173132353722682,
          0.009054037295215252,  0.008051104730469714,  0.00422796241954674,
          0.0016410687574198689};
      return {std::move(x), std::move(w)};
    }
    else if (m == 20)
    {
      // Xiao Gimbutas, 3 points, degree 20
      std::vector<T> x
          = {0.3333333333333333,   0.3333333333333333,   0.18629499774454095,
             0.18629499774454095,  0.037310880598884766, 0.037310880598884766,
             0.476245611540499,    0.476245611540499,    0.4455510569559248,
             0.4455510569559248,   0.25457926767333916,  0.25457926767333916,
             0.39342534781709987,  0.39342534781709987,  0.01097614102839789,
             0.01097614102839789,  0.10938359671171471,  0.10938359671171471,
             0.18629499774454095,  0.6274100045109181,   0.037310880598884766,
             0.9253782388022305,   0.476245611540499,    0.047508776919002016,
             0.4455510569559248,   0.10889788608815043,  0.25457926767333916,
             0.4908414646533217,   0.39342534781709987,  0.21314930436580026,
             0.01097614102839789,  0.9780477179432042,   0.10938359671171471,
             0.7812328065765706,   0.6274100045109181,   0.18629499774454095,
             0.9253782388022305,   0.037310880598884766, 0.047508776919002016,
             0.476245611540499,    0.10889788608815043,  0.4455510569559248,
             0.4908414646533217,   0.25457926767333916,  0.21314930436580026,
             0.39342534781709987,  0.9780477179432042,   0.01097614102839789,
             0.7812328065765706,   0.10938359671171471,  0.004854937607623827,
             0.06409058560843404,  0.10622720472027006,  0.2156070573900944,
             0.007570780504696579, 0.15913370765706722,  0.13980807199179993,
             0.317860123835772,    0.04656036490766434,  0.19851813222878817,
             0.038363684775374655, 0.09995229628813862,  0.009831548292802588,
             0.42002375881622406,  0.05498747914298685,  0.33313481730958744,
             0.01073721285601111,  0.2805814114236652,   0.9310544767839422,
             0.004854937607623827, 0.6781657378896355,   0.10622720472027006,
             0.8332955118382361,   0.007570780504696579, 0.5423318041724281,
             0.13980807199179993,  0.7549215028635474,   0.04656036490766434,
             0.8616840189364867,   0.038363684775374655, 0.5701446928909732,
             0.009831548292802588, 0.6118777035474257,   0.05498747914298685,
             0.7086813757203236,   0.01073721285601111,  0.06409058560843404,
             0.9310544767839422,   0.2156070573900944,   0.6781657378896355,
             0.15913370765706722,  0.8332955118382361,   0.317860123835772,
             0.5423318041724281,   0.19851813222878817,  0.7549215028635474,
             0.09995229628813862,  0.8616840189364867,   0.42002375881622406,
             0.5701446928909732,   0.33313481730958744,  0.6118777035474257,
             0.2805814114236652,   0.7086813757203236,   0.06409058560843404,
             0.004854937607623827, 0.2156070573900944,   0.10622720472027006,
             0.15913370765706722,  0.007570780504696579, 0.317860123835772,
             0.13980807199179993,  0.19851813222878817,  0.04656036490766434,
             0.09995229628813862,  0.038363684775374655, 0.42002375881622406,
             0.009831548292802588, 0.33313481730958744,  0.05498747914298685,
             0.2805814114236652,   0.01073721285601111,  0.9310544767839422,
             0.06409058560843404,  0.6781657378896355,   0.2156070573900944,
             0.8332955118382361,   0.15913370765706722,  0.5423318041724281,
             0.317860123835772,    0.7549215028635474,   0.19851813222878817,
             0.8616840189364867,   0.09995229628813862,  0.5701446928909732,
             0.42002375881622406,  0.6118777035474257,   0.33313481730958744,
             0.7086813757203236,   0.2805814114236652,   0.004854937607623827,
             0.9310544767839422,   0.10622720472027006,  0.6781657378896355,
             0.007570780504696579, 0.8332955118382361,   0.13980807199179993,
             0.5423318041724281,   0.04656036490766434,  0.7549215028635474,
             0.038363684775374655, 0.8616840189364867,   0.009831548292802588,
             0.5701446928909732,   0.05498747914298685,  0.6118777035474257,
             0.01073721285601111,  0.7086813757203236};
      std::vector<T> w = {
          0.013910110701453116,  0.009173462974252915,  0.0021612754106655778,
          0.007101825303408441,  0.009452399933232448,  0.014083201307520249,
          0.013788050629070459,  0.00079884079106662,   0.007830230776074535,
          0.009173462974252915,  0.0021612754106655778, 0.007101825303408441,
          0.009452399933232448,  0.014083201307520249,  0.013788050629070459,
          0.00079884079106662,   0.007830230776074535,  0.009173462974252915,
          0.0021612754106655778, 0.007101825303408441,  0.009452399933232448,
          0.014083201307520249,  0.013788050629070459,  0.00079884079106662,
          0.007830230776074535,  0.0011298696021258656, 0.007722607822099231,
          0.002202897418558498,  0.011691745731827735,  0.00598639857895469,
          0.004145711527613858,  0.003695681500255298,  0.008667225567219335,
          0.0035782002384576856, 0.0011298696021258656, 0.007722607822099231,
          0.002202897418558498,  0.011691745731827735,  0.00598639857895469,
          0.004145711527613858,  0.003695681500255298,  0.008667225567219335,
          0.0035782002384576856, 0.0011298696021258656, 0.007722607822099231,
          0.002202897418558498,  0.011691745731827735,  0.00598639857895469,
          0.004145711527613858,  0.003695681500255298,  0.008667225567219335,
          0.0035782002384576856, 0.0011298696021258656, 0.007722607822099231,
          0.002202897418558498,  0.011691745731827735,  0.00598639857895469,
          0.004145711527613858,  0.003695681500255298,  0.008667225567219335,
          0.0035782002384576856, 0.0011298696021258656, 0.007722607822099231,
          0.002202897418558498,  0.011691745731827735,  0.00598639857895469,
          0.004145711527613858,  0.003695681500255298,  0.008667225567219335,
          0.0035782002384576856, 0.0011298696021258656, 0.007722607822099231,
          0.002202897418558498,  0.011691745731827735,  0.00598639857895469,
          0.004145711527613858,  0.003695681500255298,  0.008667225567219335,
          0.0035782002384576856};
      return {std::move(x), std::move(w)};
    }
    else if (m == 21)
    {
      // Xiao Gimbutas, 3 points, degree 21
      std::vector<T> x
          = {0.2989362353149826,   0.2989362353149826,   0.4970078754686856,
             0.4970078754686856,   0.40361758654638513,  0.40361758654638513,
             0.11898857762271953,  0.11898857762271953,  0.19028871809127856,
             0.19028871809127856,  0.4815978686532166,   0.4815978686532166,
             0.4498127917753624,   0.4498127917753624,   0.053627575546145,
             0.053627575546145,    0.010742456432828507, 0.010742456432828507,
             0.2989362353149826,   0.4021275293700348,   0.4970078754686856,
             0.005984249062628844, 0.40361758654638513,  0.19276482690722974,
             0.11898857762271953,  0.762022844754561,    0.19028871809127856,
             0.6194225638174429,   0.4815978686532166,   0.03680426269356685,
             0.4498127917753624,   0.10037441644927525,  0.053627575546145,
             0.89274484890771,     0.010742456432828507, 0.978515087134343,
             0.4021275293700348,   0.2989362353149826,   0.005984249062628844,
             0.4970078754686856,   0.19276482690722974,  0.40361758654638513,
             0.762022844754561,    0.11898857762271953,  0.6194225638174429,
             0.19028871809127856,  0.03680426269356685,  0.4815978686532166,
             0.10037441644927525,  0.4498127917753624,   0.89274484890771,
             0.053627575546145,    0.978515087134343,    0.010742456432828507,
             0.20529555933516153,  0.28918949607859473,  0.006931809031468116,
             0.23787338259799398,  0.12377940040549276,  0.31886531079482827,
             0.03899136262322033,  0.23187362537040096,  0.009536247529710598,
             0.1331671229413703,   0.05305219170121682,  0.34680797980991107,
             0.10045802007411446,  0.21659962318998252,  0.04945106556854055,
             0.12882980796205154,  0.010254635872924515, 0.3609534080189222,
             0.010301903643423904, 0.055719565072371954, 0.5055149445862437,
             0.20529555933516153,  0.755194808370538,    0.006931809031468116,
             0.557355288799679,    0.12377940040549276,  0.7291350120063786,
             0.03899136262322033,  0.857296629528919,    0.009536247529710598,
             0.6001398284888722,   0.05305219170121682,  0.6829423567359031,
             0.10045802007411446,  0.8217191264694079,   0.04945106556854055,
             0.6287919561081533,   0.010254635872924515, 0.9339785312842042,
             0.010301903643423904, 0.28918949607859473,  0.5055149445862437,
             0.23787338259799398,  0.755194808370538,    0.31886531079482827,
             0.557355288799679,    0.23187362537040096,  0.7291350120063786,
             0.1331671229413703,   0.857296629528919,    0.34680797980991107,
             0.6001398284888722,   0.21659962318998252,  0.6829423567359031,
             0.12882980796205154,  0.8217191264694079,   0.3609534080189222,
             0.6287919561081533,   0.055719565072371954, 0.9339785312842042,
             0.28918949607859473,  0.20529555933516153,  0.23787338259799398,
             0.006931809031468116, 0.31886531079482827,  0.12377940040549276,
             0.23187362537040096,  0.03899136262322033,  0.1331671229413703,
             0.009536247529710598, 0.34680797980991107,  0.05305219170121682,
             0.21659962318998252,  0.10045802007411446,  0.12882980796205154,
             0.04945106556854055,  0.3609534080189222,   0.010254635872924515,
             0.055719565072371954, 0.010301903643423904, 0.5055149445862437,
             0.28918949607859473,  0.755194808370538,    0.23787338259799398,
             0.557355288799679,    0.31886531079482827,  0.7291350120063786,
             0.23187362537040096,  0.857296629528919,    0.1331671229413703,
             0.6001398284888722,   0.34680797980991107,  0.6829423567359031,
             0.21659962318998252,  0.8217191264694079,   0.12882980796205154,
             0.6287919561081533,   0.3609534080189222,   0.9339785312842042,
             0.055719565072371954, 0.20529555933516153,  0.5055149445862437,
             0.006931809031468116, 0.755194808370538,    0.12377940040549276,
             0.557355288799679,    0.03899136262322033,  0.7291350120063786,
             0.009536247529710598, 0.857296629528919,    0.05305219170121682,
             0.6001398284888722,   0.10045802007411446,  0.6829423567359031,
             0.04945106556854055,  0.8217191264694079,   0.010254635872924515,
             0.6287919561081533,   0.010301903643423904, 0.9339785312842042};
      std::vector<T> w = {
          0.01072556096456617,   0.0022189148485329394, 0.011500352326641932,
          0.006828016226115099,  0.009727620930375354,  0.006107205081692191,
          0.009807237613912011,  0.0035760425506418257, 0.0007543496361893447,
          0.01072556096456617,   0.0022189148485329394, 0.011500352326641932,
          0.006828016226115099,  0.009727620930375354,  0.006107205081692191,
          0.009807237613912011,  0.0035760425506418257, 0.0007543496361893447,
          0.01072556096456617,   0.0022189148485329394, 0.011500352326641932,
          0.006828016226115099,  0.009727620930375354,  0.006107205081692191,
          0.009807237613912011,  0.0035760425506418257, 0.0007543496361893447,
          0.008747708077881562,  0.002103060144074865,  0.009223742423966418,
          0.005234952092662423,  0.002240406560950738,  0.007250152959485511,
          0.007952018352713986,  0.004905985911275205,  0.0034199424289671526,
          0.0016327142920220426, 0.008747708077881562,  0.002103060144074865,
          0.009223742423966418,  0.005234952092662423,  0.002240406560950738,
          0.007250152959485511,  0.007952018352713986,  0.004905985911275205,
          0.0034199424289671526, 0.0016327142920220426, 0.008747708077881562,
          0.002103060144074865,  0.009223742423966418,  0.005234952092662423,
          0.002240406560950738,  0.007250152959485511,  0.007952018352713986,
          0.004905985911275205,  0.0034199424289671526, 0.0016327142920220426,
          0.008747708077881562,  0.002103060144074865,  0.009223742423966418,
          0.005234952092662423,  0.002240406560950738,  0.007250152959485511,
          0.007952018352713986,  0.004905985911275205,  0.0034199424289671526,
          0.0016327142920220426, 0.008747708077881562,  0.002103060144074865,
          0.009223742423966418,  0.005234952092662423,  0.002240406560950738,
          0.007250152959485511,  0.007952018352713986,  0.004905985911275205,
          0.0034199424289671526, 0.0016327142920220426, 0.008747708077881562,
          0.002103060144074865,  0.009223742423966418,  0.005234952092662423,
          0.002240406560950738,  0.007250152959485511,  0.007952018352713986,
          0.004905985911275205,  0.0034199424289671526, 0.0016327142920220426};
      return {std::move(x), std::move(w)};
    }
    else if (m == 22)
    {
      // Xiao Gimbutas, 3 points, degree 22
      std::vector<T> x
          = {0.3851845246273021,   0.3851845246273021,    0.4577694113676721,
             0.4577694113676721,   0.29455825902995014,   0.29455825902995014,
             0.18851052363028398,  0.18851052363028398,   0.42198188879353493,
             0.42198188879353493,  0.49616117840970864,   0.49616117840970864,
             0.029108470670807574, 0.029108470670807574,  0.11543153821920499,
             0.11543153821920499,  0.3851845246273021,    0.22963095074539575,
             0.4577694113676721,   0.08446117726465585,   0.29455825902995014,
             0.4108834819400997,   0.18851052363028398,   0.622978952739432,
             0.42198188879353493,  0.15603622241293014,   0.49616117840970864,
             0.007677643180582727, 0.029108470670807574,  0.9417830586583849,
             0.11543153821920499,  0.76913692356159,      0.22963095074539575,
             0.3851845246273021,   0.08446117726465585,   0.4577694113676721,
             0.4108834819400997,   0.29455825902995014,   0.622978952739432,
             0.18851052363028398,  0.15603622241293014,   0.42198188879353493,
             0.007677643180582727, 0.49616117840970864,   0.9417830586583849,
             0.029108470670807574, 0.76913692356159,      0.11543153821920499,
             0.007876282221582374, 0.06984216946744362,   0.04475228434833587,
             0.09039883116640775,  0.038275234700863824,  0.4113417640205587,
             0.10274707598693139,  0.3321061050074464,    0.007400241234710751,
             0.36257628043246726,  0.19108129796672008,   0.29006682411666884,
             0.04399164539345585,  0.28793180282417186,   0.10868994186267199,
             0.21678693336494115,  0.009144711374964054,  0.14587371987352518,
             0.048254924114641384, 0.17629743482450005,   0.009163909248185229,
             0.24399064603949305,  0.0017984649889483744, 0.017934321052938986,
             0.9222815483109741,   0.007876282221582374,  0.8648488844852563,
             0.04475228434833587,  0.5503830012785775,    0.038275234700863824,
             0.5651468190056222,   0.10274707598693139,   0.630023478332822,
             0.007400241234710751, 0.5188518779166111,    0.19108129796672008,
             0.6680765517823722,   0.04399164539345585,   0.6745231247723869,
             0.10868994186267199,  0.8449815687515108,    0.009144711374964054,
             0.7754476410608586,   0.048254924114641384,  0.7468454447123217,
             0.009163909248185229, 0.9802672139581126,    0.0017984649889483744,
             0.06984216946744362,  0.9222815483109741,    0.09039883116640775,
             0.8648488844852563,   0.4113417640205587,    0.5503830012785775,
             0.3321061050074464,   0.5651468190056222,    0.36257628043246726,
             0.630023478332822,    0.29006682411666884,   0.5188518779166111,
             0.28793180282417186,  0.6680765517823722,    0.21678693336494115,
             0.6745231247723869,   0.14587371987352518,   0.8449815687515108,
             0.17629743482450005,  0.7754476410608586,    0.24399064603949305,
             0.7468454447123217,   0.017934321052938986,  0.9802672139581126,
             0.06984216946744362,  0.007876282221582374,  0.09039883116640775,
             0.04475228434833587,  0.4113417640205587,    0.038275234700863824,
             0.3321061050074464,   0.10274707598693139,   0.36257628043246726,
             0.007400241234710751, 0.29006682411666884,   0.19108129796672008,
             0.28793180282417186,  0.04399164539345585,   0.21678693336494115,
             0.10868994186267199,  0.14587371987352518,   0.009144711374964054,
             0.17629743482450005,  0.048254924114641384,  0.24399064603949305,
             0.009163909248185229, 0.017934321052938986,  0.0017984649889483744,
             0.9222815483109741,   0.06984216946744362,   0.8648488844852563,
             0.09039883116640775,  0.5503830012785775,    0.4113417640205587,
             0.5651468190056222,   0.3321061050074464,    0.630023478332822,
             0.36257628043246726,  0.5188518779166111,    0.29006682411666884,
             0.6680765517823722,   0.28793180282417186,   0.6745231247723869,
             0.21678693336494115,  0.8449815687515108,    0.14587371987352518,
             0.7754476410608586,   0.17629743482450005,   0.7468454447123217,
             0.24399064603949305,  0.9802672139581126,    0.017934321052938986,
             0.007876282221582374, 0.9222815483109741,    0.04475228434833587,
             0.8648488844852563,   0.038275234700863824,  0.5503830012785775,
             0.10274707598693139,  0.5651468190056222,    0.007400241234710751,
             0.630023478332822,    0.19108129796672008,   0.5188518779166111,
             0.04399164539345585,  0.6680765517823722,    0.10868994186267199,
             0.6745231247723869,   0.009144711374964054,  0.8449815687515108,
             0.048254924114641384, 0.7754476410608586,    0.009163909248185229,
             0.7468454447123217,   0.0017984649889483744, 0.9802672139581126};
      std::vector<T> w = {
          0.006746541941805331,  0.006930699762117096,  0.010537881978726092,
          0.008010649562574445,  0.009426546276920644,  0.002644669832992209,
          0.0017845545829281882, 0.007207856564052301,  0.006746541941805331,
          0.006930699762117096,  0.010537881978726092,  0.008010649562574445,
          0.009426546276920644,  0.002644669832992209,  0.0017845545829281882,
          0.007207856564052301,  0.006746541941805331,  0.006930699762117096,
          0.010537881978726092,  0.008010649562574445,  0.009426546276920644,
          0.002644669832992209,  0.0017845545829281882, 0.007207856564052301,
          0.0012977192371156389, 0.003758788908894188,  0.005598656735981385,
          0.00885954674475511,   0.0024521301987784822, 0.01085320977775448,
          0.005831111433671501,  0.007855081311285159,  0.002053343535787778,
          0.005281792483873449,  0.0025270384487923007, 0.0003202142655857129,
          0.0012977192371156389, 0.003758788908894188,  0.005598656735981385,
          0.00885954674475511,   0.0024521301987784822, 0.01085320977775448,
          0.005831111433671501,  0.007855081311285159,  0.002053343535787778,
          0.005281792483873449,  0.0025270384487923007, 0.0003202142655857129,
          0.0012977192371156389, 0.003758788908894188,  0.005598656735981385,
          0.00885954674475511,   0.0024521301987784822, 0.01085320977775448,
          0.005831111433671501,  0.007855081311285159,  0.002053343535787778,
          0.005281792483873449,  0.0025270384487923007, 0.0003202142655857129,
          0.0012977192371156389, 0.003758788908894188,  0.005598656735981385,
          0.00885954674475511,   0.0024521301987784822, 0.01085320977775448,
          0.005831111433671501,  0.007855081311285159,  0.002053343535787778,
          0.005281792483873449,  0.0025270384487923007, 0.0003202142655857129,
          0.0012977192371156389, 0.003758788908894188,  0.005598656735981385,
          0.00885954674475511,   0.0024521301987784822, 0.01085320977775448,
          0.005831111433671501,  0.007855081311285159,  0.002053343535787778,
          0.005281792483873449,  0.0025270384487923007, 0.0003202142655857129,
          0.0012977192371156389, 0.003758788908894188,  0.005598656735981385,
          0.00885954674475511,   0.0024521301987784822, 0.01085320977775448,
          0.005831111433671501,  0.007855081311285159,  0.002053343535787778,
          0.005281792483873449,  0.0025270384487923007, 0.0003202142655857129};
      return {std::move(x), std::move(w)};
    }
    else if (m == 23)
    {
      // Xiao Gimbutas, 3 points, degree 23
      std::vector<T> x
          = {0.3333333333333333,    0.3333333333333333,   0.0390072687570322,
             0.0390072687570322,    0.4803288773373085,   0.4803288773373085,
             0.08684104820763322,   0.08684104820763322,  0.39432350601154154,
             0.39432350601154154,   0.2662513178772473,   0.2662513178772473,
             0.1371293873116477,    0.1371293873116477,   0.4989594312095863,
             0.4989594312095863,    0.4446924421277275,   0.4446924421277275,
             0.19874980639653628,   0.19874980639653628,  0.009016440205598442,
             0.009016440205598442,  0.0390072687570322,   0.9219854624859356,
             0.4803288773373085,    0.039342245325382996, 0.08684104820763322,
             0.8263179035847336,    0.39432350601154154,  0.21135298797691693,
             0.2662513178772473,    0.46749736424550536,  0.1371293873116477,
             0.7257412253767046,    0.4989594312095863,   0.002081137580827397,
             0.4446924421277275,    0.11061511574454497,  0.19874980639653628,
             0.6025003872069274,    0.009016440205598442, 0.9819671195888031,
             0.9219854624859356,    0.0390072687570322,   0.039342245325382996,
             0.4803288773373085,    0.8263179035847336,   0.08684104820763322,
             0.21135298797691693,   0.39432350601154154,  0.46749736424550536,
             0.2662513178772473,    0.7257412253767046,   0.1371293873116477,
             0.002081137580827397,  0.4989594312095863,   0.11061511574454497,
             0.4446924421277275,    0.6025003872069274,   0.19874980639653628,
             0.9819671195888031,    0.009016440205598442, 0.02387025365435361,
             0.15950379892475722,   0.005189821760844536, 0.11410136032236454,
             0.0327410291887064,    0.0955398781717349,   0.0024475998559663793,
             0.31116226805170194,   0.008725289585308535, 0.20561723205805207,
             0.007162539910244482,  0.0472616294497253,   0.068526954187213,
             0.3585095935696251,    0.10172832932728422,  0.2404827720350127,
             0.05835157523751544,   0.17293230312922397,  0.1548301554055162,
             0.3163043076538381,    0.014758969729945169, 0.39775857680300764,
             0.03299370819253279,   0.27879416981410227,  0.8166259474208892,
             0.02387025365435361,   0.880708817916791,    0.005189821760844536,
             0.8717190926395587,    0.0327410291887064,   0.6863901320923316,
             0.0024475998559663793, 0.7856574783566395,   0.008725289585308535,
             0.9455758306400301,    0.007162539910244482, 0.5729634522431619,
             0.068526954187213,     0.6577888986377031,   0.10172832932728422,
             0.7687161216332605,    0.05835157523751544,  0.5288655369406456,
             0.1548301554055162,    0.5874824534670472,   0.014758969729945169,
             0.688212121993365,     0.03299370819253279,  0.15950379892475722,
             0.8166259474208892,    0.11410136032236454,  0.880708817916791,
             0.0955398781717349,    0.8717190926395587,   0.31116226805170194,
             0.6863901320923316,    0.20561723205805207,  0.7856574783566395,
             0.0472616294497253,    0.9455758306400301,   0.3585095935696251,
             0.5729634522431619,    0.2404827720350127,   0.6577888986377031,
             0.17293230312922397,   0.7687161216332605,   0.3163043076538381,
             0.5288655369406456,    0.39775857680300764,  0.5874824534670472,
             0.27879416981410227,   0.688212121993365,    0.15950379892475722,
             0.02387025365435361,   0.11410136032236454,  0.005189821760844536,
             0.0955398781717349,    0.0327410291887064,   0.31116226805170194,
             0.0024475998559663793, 0.20561723205805207,  0.008725289585308535,
             0.0472616294497253,    0.007162539910244482, 0.3585095935696251,
             0.068526954187213,     0.2404827720350127,   0.10172832932728422,
             0.17293230312922397,   0.05835157523751544,  0.3163043076538381,
             0.1548301554055162,    0.39775857680300764,  0.014758969729945169,
             0.27879416981410227,   0.03299370819253279,  0.8166259474208892,
             0.15950379892475722,   0.880708817916791,    0.11410136032236454,
             0.8717190926395587,    0.0955398781717349,   0.6863901320923316,
             0.31116226805170194,   0.7856574783566395,   0.20561723205805207,
             0.9455758306400301,    0.0472616294497253,   0.5729634522431619,
             0.3585095935696251,    0.6577888986377031,   0.2404827720350127,
             0.7687161216332605,    0.17293230312922397,  0.5288655369406456,
             0.3163043076538381,    0.5874824534670472,   0.39775857680300764,
             0.688212121993365,     0.27879416981410227,  0.02387025365435361,
             0.8166259474208892,    0.005189821760844536, 0.880708817916791,
             0.0327410291887064,    0.8717190926395587,   0.0024475998559663793,
             0.6863901320923316,    0.008725289585308535, 0.7856574783566395,
             0.007162539910244482,  0.9455758306400301,   0.068526954187213,
             0.5729634522431619,    0.10172832932728422,  0.6577888986377031,
             0.05835157523751544,   0.7687161216332605,   0.1548301554055162,
             0.5288655369406456,    0.014758969729945169, 0.5874824534670472,
             0.03299370819253279,   0.688212121993365};
      std::vector<T> w = {
          0.012626530161518105,  0.001957870129516468,  0.00569894463390038,
          0.004479958512756771,  0.011837304231564011,  0.011903931443749882,
          0.007279724696370875,  0.0012037723020907048, 0.009475975334669443,
          0.009967638940052512,  0.0005326806164146575, 0.001957870129516468,
          0.00569894463390038,   0.004479958512756771,  0.011837304231564011,
          0.011903931443749882,  0.007279724696370875,  0.0012037723020907048,
          0.009475975334669443,  0.009967638940052512,  0.0005326806164146575,
          0.001957870129516468,  0.00569894463390038,   0.004479958512756771,
          0.011837304231564011,  0.011903931443749882,  0.007279724696370875,
          0.0012037723020907048, 0.009475975334669443,  0.009967638940052512,
          0.0005326806164146575, 0.0012640830276911316, 0.0011125098648622574,
          0.0026640152155973924, 0.0011405518381279172, 0.002057375172208046,
          0.0009762956639453631, 0.007490556696599583,  0.008060620818508576,
          0.0052351282465650335, 0.010422197929484405,  0.0035488894172609124,
          0.005087787328353519,  0.0012640830276911316, 0.0011125098648622574,
          0.0026640152155973924, 0.0011405518381279172, 0.002057375172208046,
          0.0009762956639453631, 0.007490556696599583,  0.008060620818508576,
          0.0052351282465650335, 0.010422197929484405,  0.0035488894172609124,
          0.005087787328353519,  0.0012640830276911316, 0.0011125098648622574,
          0.0026640152155973924, 0.0011405518381279172, 0.002057375172208046,
          0.0009762956639453631, 0.007490556696599583,  0.008060620818508576,
          0.0052351282465650335, 0.010422197929484405,  0.0035488894172609124,
          0.005087787328353519,  0.0012640830276911316, 0.0011125098648622574,
          0.0026640152155973924, 0.0011405518381279172, 0.002057375172208046,
          0.0009762956639453631, 0.007490556696599583,  0.008060620818508576,
          0.0052351282465650335, 0.010422197929484405,  0.0035488894172609124,
          0.005087787328353519,  0.0012640830276911316, 0.0011125098648622574,
          0.0026640152155973924, 0.0011405518381279172, 0.002057375172208046,
          0.0009762956639453631, 0.007490556696599583,  0.008060620818508576,
          0.0052351282465650335, 0.010422197929484405,  0.0035488894172609124,
          0.005087787328353519,  0.0012640830276911316, 0.0011125098648622574,
          0.0026640152155973924, 0.0011405518381279172, 0.002057375172208046,
          0.0009762956639453631, 0.007490556696599583,  0.008060620818508576,
          0.0052351282465650335, 0.010422197929484405,  0.0035488894172609124,
          0.005087787328353519};
      return {std::move(x), std::move(w)};
    }
    else if (m == 24)
    {
      // Xiao Gimbutas, 3 points, degree 24
      std::vector<T> x
          = {0.3333333333333333,   0.3333333333333333,   0.4188909749106028,
             0.4188909749106028,   0.16236063371692644,  0.16236063371692644,
             0.04098562900111713,  0.04098562900111713,  0.006731270887888441,
             0.006731270887888441, 0.49625527767573513,  0.49625527767573513,
             0.2642313154382726,   0.2642313154382726,   0.4806125617925032,
             0.4806125617925032,   0.0963284955992153,   0.0963284955992153,
             0.3753529267020863,   0.3753529267020863,   0.4188909749106028,
             0.16221805017879443,  0.16236063371692644,  0.6752787325661471,
             0.04098562900111713,  0.9180287419977657,   0.006731270887888441,
             0.9865374582242231,   0.49625527767573513,  0.007489444648529742,
             0.2642313154382726,   0.4715373691234548,   0.4806125617925032,
             0.03877487641499355,  0.0963284955992153,   0.8073430088015694,
             0.3753529267020863,   0.24929414659582738,  0.16221805017879443,
             0.4188909749106028,   0.6752787325661471,   0.16236063371692644,
             0.9180287419977657,   0.04098562900111713,  0.9865374582242231,
             0.006731270887888441, 0.007489444648529742, 0.49625527767573513,
             0.4715373691234548,   0.2642313154382726,   0.03877487641499355,
             0.4806125617925032,   0.8073430088015694,   0.0963284955992153,
             0.24929414659582738,  0.3753529267020863,   0.17036728246244368,
             0.241479760073594,    0.169759795860736,    0.3289758089242264,
             0.03831822582101938,  0.09316740977988115,  0.09265648152075752,
             0.39452027980019433,  0.041188714248475373, 0.16267741639447741,
             0.03957090497015804,  0.25358901421887947,  0.038592700174896126,
             0.36225224131779127,  0.09453496173659899,  0.28162257770616084,
             0.007387994632294238, 0.3832726649926592,   0.007546003162312815,
             0.2737503525162605,   0.007234558457782137, 0.09412134279736603,
             0.09556626952736523,  0.18039615188676572,  0.007987921880847964,
             0.17473734628280568,  0.008074910870208776, 0.03729147205129122,
             0.5881529574639623,   0.17036728246244368,  0.5012643952150376,
             0.169759795860736,    0.8685143643990995,   0.03831822582101938,
             0.5128232386790481,   0.09265648152075752,  0.7961338693570472,
             0.041188714248475373, 0.7068400808109625,   0.03957090497015804,
             0.5991550585073127,   0.038592700174896126, 0.6238424605572401,
             0.09453496173659899,  0.6093393403750466,   0.007387994632294238,
             0.7187036443214267,   0.007546003162312815, 0.8986440987448518,
             0.007234558457782137, 0.724037578585869,    0.09556626952736523,
             0.8172747318363464,   0.007987921880847964, 0.9546336170785,
             0.008074910870208776, 0.241479760073594,    0.5881529574639623,
             0.3289758089242264,   0.5012643952150376,   0.09316740977988115,
             0.8685143643990995,   0.39452027980019433,  0.5128232386790481,
             0.16267741639447741,  0.7961338693570472,   0.25358901421887947,
             0.7068400808109625,   0.36225224131779127,  0.5991550585073127,
             0.28162257770616084,  0.6238424605572401,   0.3832726649926592,
             0.6093393403750466,   0.2737503525162605,   0.7187036443214267,
             0.09412134279736603,  0.8986440987448518,   0.18039615188676572,
             0.724037578585869,    0.17473734628280568,  0.8172747318363464,
             0.03729147205129122,  0.9546336170785,      0.241479760073594,
             0.17036728246244368,  0.3289758089242264,   0.169759795860736,
             0.09316740977988115,  0.03831822582101938,  0.39452027980019433,
             0.09265648152075752,  0.16267741639447741,  0.041188714248475373,
             0.25358901421887947,  0.03957090497015804,  0.36225224131779127,
             0.038592700174896126, 0.28162257770616084,  0.09453496173659899,
             0.3832726649926592,   0.007387994632294238, 0.2737503525162605,
             0.007546003162312815, 0.09412134279736603,  0.007234558457782137,
             0.18039615188676572,  0.09556626952736523,  0.17473734628280568,
             0.007987921880847964, 0.03729147205129122,  0.008074910870208776,
             0.5881529574639623,   0.241479760073594,    0.5012643952150376,
             0.3289758089242264,   0.8685143643990995,   0.09316740977988115,
             0.5128232386790481,   0.39452027980019433,  0.7961338693570472,
             0.16267741639447741,  0.7068400808109625,   0.25358901421887947,
             0.5991550585073127,   0.36225224131779127,  0.6238424605572401,
             0.28162257770616084,  0.6093393403750466,   0.3832726649926592,
             0.7187036443214267,   0.2737503525162605,   0.8986440987448518,
             0.09412134279736603,  0.724037578585869,    0.18039615188676572,
             0.8172747318363464,   0.17473734628280568,  0.9546336170785,
             0.03729147205129122,  0.17036728246244368,  0.5881529574639623,
             0.169759795860736,    0.5012643952150376,   0.03831822582101938,
             0.8685143643990995,   0.09265648152075752,  0.5128232386790481,
             0.041188714248475373, 0.7961338693570472,   0.03957090497015804,
             0.7068400808109625,   0.038592700174896126, 0.5991550585073127,
             0.09453496173659899,  0.6238424605572401,   0.007387994632294238,
             0.6093393403750466,   0.007546003162312815, 0.7187036443214267,
             0.007234558457782137, 0.8986440987448518,   0.09556626952736523,
             0.724037578585869,    0.007987921880847964, 0.8172747318363464,
             0.008074910870208776, 0.9546336170785};
      std::vector<T> w = {
          0.00627284492280016,   0.006555266350942619,  0.0051895080282000966,
          0.0019168498654645917, 0.0003086272527483216, 0.002171623361085349,
          0.010260004335754922,  0.005176247385426301,  0.005013696533694453,
          0.009497293258676329,  0.006555266350942619,  0.0051895080282000966,
          0.0019168498654645917, 0.0003086272527483216, 0.002171623361085349,
          0.010260004335754922,  0.005176247385426301,  0.005013696533694453,
          0.009497293258676329,  0.006555266350942619,  0.0051895080282000966,
          0.0019168498654645917, 0.0003086272527483216, 0.002171623361085349,
          0.010260004335754922,  0.005176247385426301,  0.005013696533694453,
          0.009497293258676329,  0.007072522903242423,  0.007637221300662313,
          0.002683135727083884,  0.0075159271748706695, 0.0036020670873987137,
          0.004452438464081782,  0.004973625937841209,  0.007176175789078727,
          0.0021210746334018845, 0.0020406375385582255, 0.0012946061911989924,
          0.005921781071271555,  0.0018536133821231541, 0.0008984737927232883,
          0.007072522903242423,  0.007637221300662313,  0.002683135727083884,
          0.0075159271748706695, 0.0036020670873987137, 0.004452438464081782,
          0.004973625937841209,  0.007176175789078727,  0.0021210746334018845,
          0.0020406375385582255, 0.0012946061911989924, 0.005921781071271555,
          0.0018536133821231541, 0.0008984737927232883, 0.007072522903242423,
          0.007637221300662313,  0.002683135727083884,  0.0075159271748706695,
          0.0036020670873987137, 0.004452438464081782,  0.004973625937841209,
          0.007176175789078727,  0.0021210746334018845, 0.0020406375385582255,
          0.0012946061911989924, 0.005921781071271555,  0.0018536133821231541,
          0.0008984737927232883, 0.007072522903242423,  0.007637221300662313,
          0.002683135727083884,  0.0075159271748706695, 0.0036020670873987137,
          0.004452438464081782,  0.004973625937841209,  0.007176175789078727,
          0.0021210746334018845, 0.0020406375385582255, 0.0012946061911989924,
          0.005921781071271555,  0.0018536133821231541, 0.0008984737927232883,
          0.007072522903242423,  0.007637221300662313,  0.002683135727083884,
          0.0075159271748706695, 0.0036020670873987137, 0.004452438464081782,
          0.004973625937841209,  0.007176175789078727,  0.0021210746334018845,
          0.0020406375385582255, 0.0012946061911989924, 0.005921781071271555,
          0.0018536133821231541, 0.0008984737927232883, 0.007072522903242423,
          0.007637221300662313,  0.002683135727083884,  0.0075159271748706695,
          0.0036020670873987137, 0.004452438464081782,  0.004973625937841209,
          0.007176175789078727,  0.0021210746334018845, 0.0020406375385582255,
          0.0012946061911989924, 0.005921781071271555,  0.0018536133821231541,
          0.0008984737927232883};
      return {std::move(x), std::move(w)};
    }
    else if (m == 25)
    {
      // Xiao Gimbutas, 3 points, degree 25
      std::vector<T> x = {
          0.3876420304045634,    0.3876420304045634,    0.21100450806149668,
          0.21100450806149668,   0.2994923158045085,    0.2994923158045085,
          0.03722292599244087,   0.03722292599244087,   0.1451092435745004,
          0.1451092435745004,    0.42475930454057476,   0.42475930454057476,
          0.4622087087487061,    0.4622087087487061,    0.09294970170076994,
          0.09294970170076994,   0.007835344282603851,  0.007835344282603851,
          0.48903936966039546,   0.48903936966039546,   0.3876420304045634,
          0.22471593919087318,   0.21100450806149668,   0.5779909838770066,
          0.2994923158045085,    0.40101536839098295,   0.03722292599244087,
          0.9255541480151183,    0.1451092435745004,    0.7097815128509992,
          0.42475930454057476,   0.1504813909188505,    0.4622087087487061,
          0.07558258250258776,   0.09294970170076994,   0.8141005965984601,
          0.007835344282603851,  0.9843293114347923,    0.48903936966039546,
          0.021921260679209076,  0.22471593919087318,   0.3876420304045634,
          0.5779909838770066,    0.21100450806149668,   0.40101536839098295,
          0.2994923158045085,    0.9255541480151183,    0.03722292599244087,
          0.7097815128509992,    0.1451092435745004,    0.1504813909188505,
          0.42475930454057476,   0.07558258250258776,   0.4622087087487061,
          0.8141005965984601,    0.09294970170076994,   0.9843293114347923,
          0.007835344282603851,  0.021921260679209076,  0.48903936966039546,
          0.0018188666342743875, 0.4404169274793433,    0.03696014157967147,
          0.15900790619732788,   0.07885806800563527,   0.1773537967572529,
          0.06884752943149791,   0.2700667358209594,    0.11599980764096017,
          0.34139103302114987,   0.04831743428737695,   0.3739379797195844,
          0.007128314501257424,  0.09913306334168219,   0.20369291058425096,
          0.29950641862967453,   0.007236161747948156,  0.17862984860361625,
          0.012913883250032529,  0.362068801895972,     0.037687949784259066,
          0.08879291548936656,   0.13700669408707095,   0.23362281014171524,
          0.02454006024752439,   0.2565954097090198,    0.007188828261693038,
          0.041068819111784644,  0.0008914643174981278, 0.2794161886492607,
          0.5577642058863823,    0.0018188666342743875, 0.8040319522230007,
          0.03696014157967147,   0.7437881352371118,    0.07885806800563527,
          0.6610857347475427,    0.06884752943149791,   0.5426091593378899,
          0.11599980764096017,   0.5777445859930387,    0.04831743428737695,
          0.8937386221570605,    0.007128314501257424,  0.4968006707860745,
          0.20369291058425096,   0.8141339896484356,    0.007236161747948156,
          0.6250173148539955,    0.012913883250032529,  0.8735191347263744,
          0.037687949784259066,  0.6293704957712138,    0.13700669408707095,
          0.7188645300434557,    0.02454006024752439,   0.9517423526265223,
          0.007188828261693038,  0.7196923470332413,    0.0008914643174981278,
          0.4404169274793433,    0.5577642058863823,    0.15900790619732788,
          0.8040319522230007,    0.1773537967572529,    0.7437881352371118,
          0.2700667358209594,    0.6610857347475427,    0.34139103302114987,
          0.5426091593378899,    0.3739379797195844,    0.5777445859930387,
          0.09913306334168219,   0.8937386221570605,    0.29950641862967453,
          0.4968006707860745,    0.17862984860361625,   0.8141339896484356,
          0.362068801895972,     0.6250173148539955,    0.08879291548936656,
          0.8735191347263744,    0.23362281014171524,   0.6293704957712138,
          0.2565954097090198,    0.7188645300434557,    0.041068819111784644,
          0.9517423526265223,    0.2794161886492607,    0.7196923470332413,
          0.4404169274793433,    0.0018188666342743875, 0.15900790619732788,
          0.03696014157967147,   0.1773537967572529,    0.07885806800563527,
          0.2700667358209594,    0.06884752943149791,   0.34139103302114987,
          0.11599980764096017,   0.3739379797195844,    0.04831743428737695,
          0.09913306334168219,   0.007128314501257424,  0.29950641862967453,
          0.20369291058425096,   0.17862984860361625,   0.007236161747948156,
          0.362068801895972,     0.012913883250032529,  0.08879291548936656,
          0.037687949784259066,  0.23362281014171524,   0.13700669408707095,
          0.2565954097090198,    0.02454006024752439,   0.041068819111784644,
          0.007188828261693038,  0.2794161886492607,    0.0008914643174981278,
          0.5577642058863823,    0.4404169274793433,    0.8040319522230007,
          0.15900790619732788,   0.7437881352371118,    0.1773537967572529,
          0.6610857347475427,    0.2700667358209594,    0.5426091593378899,
          0.34139103302114987,   0.5777445859930387,    0.3739379797195844,
          0.8937386221570605,    0.09913306334168219,   0.4968006707860745,
          0.29950641862967453,   0.8141339896484356,    0.17862984860361625,
          0.6250173148539955,    0.362068801895972,     0.8735191347263744,
          0.08879291548936656,   0.6293704957712138,    0.23362281014171524,
          0.7188645300434557,    0.2565954097090198,    0.9517423526265223,
          0.041068819111784644,  0.7196923470332413,    0.2794161886492607,
          0.0018188666342743875, 0.5577642058863823,    0.03696014157967147,
          0.8040319522230007,    0.07885806800563527,   0.7437881352371118,
          0.06884752943149791,   0.6610857347475427,    0.11599980764096017,
          0.5426091593378899,    0.04831743428737695,   0.5777445859930387,
          0.007128314501257424,  0.8937386221570605,    0.20369291058425096,
          0.4968006707860745,    0.007236161747948156,  0.8141339896484356,
          0.012913883250032529,  0.6250173148539955,    0.037687949784259066,
          0.8735191347263744,    0.13700669408707095,   0.6293704957712138,
          0.02454006024752439,   0.7188645300434557,    0.007188828261693038,
          0.9517423526265223,    0.0008914643174981278, 0.7196923470332413};
      std::vector<T> w = {
          0.006844925774136122,  0.005793631618005297,  0.009008820350850738,
          0.001698648860952368,  0.005745762931282399,  0.00795565506872921,
          0.006827137593764007,  0.004591410629910018,  0.0004032551441623084,
          0.004222042973260539,  0.006844925774136122,  0.005793631618005297,
          0.009008820350850738,  0.001698648860952368,  0.005745762931282399,
          0.00795565506872921,   0.006827137593764007,  0.004591410629910018,
          0.0004032551441623084, 0.004222042973260539,  0.006844925774136122,
          0.005793631618005297,  0.009008820350850738,  0.001698648860952368,
          0.005745762931282399,  0.00795565506872921,   0.006827137593764007,
          0.004591410629910018,  0.0004032551441623084, 0.004222042973260539,
          0.0008374089159673526, 0.003155739012379637,  0.004757510783727886,
          0.00544219680621846,   0.007920176143949218,  0.0053200853477543926,
          0.0012726358126745072, 0.00895691044613803,   0.0016318698410246215,
          0.0027273191839872145, 0.00263628096071471,   0.006870041296011276,
          0.0036571704539664234, 0.0008464918170636662, 0.0007558510392294402,
          0.0008374089159673526, 0.003155739012379637,  0.004757510783727886,
          0.00544219680621846,   0.007920176143949218,  0.0053200853477543926,
          0.0012726358126745072, 0.00895691044613803,   0.0016318698410246215,
          0.0027273191839872145, 0.00263628096071471,   0.006870041296011276,
          0.0036571704539664234, 0.0008464918170636662, 0.0007558510392294402,
          0.0008374089159673526, 0.003155739012379637,  0.004757510783727886,
          0.00544219680621846,   0.007920176143949218,  0.0053200853477543926,
          0.0012726358126745072, 0.00895691044613803,   0.0016318698410246215,
          0.0027273191839872145, 0.00263628096071471,   0.006870041296011276,
          0.0036571704539664234, 0.0008464918170636662, 0.0007558510392294402,
          0.0008374089159673526, 0.003155739012379637,  0.004757510783727886,
          0.00544219680621846,   0.007920176143949218,  0.0053200853477543926,
          0.0012726358126745072, 0.00895691044613803,   0.0016318698410246215,
          0.0027273191839872145, 0.00263628096071471,   0.006870041296011276,
          0.0036571704539664234, 0.0008464918170636662, 0.0007558510392294402,
          0.0008374089159673526, 0.003155739012379637,  0.004757510783727886,
          0.00544219680621846,   0.007920176143949218,  0.0053200853477543926,
          0.0012726358126745072, 0.00895691044613803,   0.0016318698410246215,
          0.0027273191839872145, 0.00263628096071471,   0.006870041296011276,
          0.0036571704539664234, 0.0008464918170636662, 0.0007558510392294402,
          0.0008374089159673526, 0.003155739012379637,  0.004757510783727886,
          0.00544219680621846,   0.007920176143949218,  0.0053200853477543926,
          0.0012726358126745072, 0.00895691044613803,   0.0016318698410246215,
          0.0027273191839872145, 0.00263628096071471,   0.006870041296011276,
          0.0036571704539664234, 0.0008464918170636662, 0.0007558510392294402};
      return {std::move(x), std::move(w)};
    }
    else if (m == 26)
    {
      // Xiao Gimbutas, 3 points, degree 26
      std::vector<T> x = {
          0.3333333333333333,    0.3333333333333333,    0.06673712257646625,
          0.06673712257646625,   0.0063401164920769415, 0.0063401164920769415,
          0.4937530328963848,    0.4937530328963848,    0.388787497107594,
          0.388787497107594,     0.2731471009290788,    0.2731471009290788,
          0.471828563321166,     0.471828563321166,     0.1542014303645443,
          0.1542014303645443,    0.21204316330220568,   0.21204316330220568,
          0.4359854193843832,    0.4359854193843832,    0.06673712257646625,
          0.8665257548470675,    0.0063401164920769415, 0.9873197670158461,
          0.4937530328963848,    0.01249393420723044,   0.388787497107594,
          0.22242500578481195,   0.2731471009290788,    0.4537057981418424,
          0.471828563321166,     0.056342873357667966,  0.1542014303645443,
          0.6915971392709114,    0.21204316330220568,   0.5759136733955886,
          0.4359854193843832,    0.12802916123123365,   0.8665257548470675,
          0.06673712257646625,   0.9873197670158461,    0.0063401164920769415,
          0.01249393420723044,   0.4937530328963848,    0.22242500578481195,
          0.388787497107594,     0.4537057981418424,    0.2731471009290788,
          0.056342873357667966,  0.471828563321166,     0.6915971392709114,
          0.1542014303645443,    0.5759136733955886,    0.21204316330220568,
          0.12802916123123365,   0.4359854193843832,    0.004794660975436677,
          0.08007165494031654,   0.029155196206835834,  0.031643611571530776,
          0.02620936402249865,   0.07538004751539866,   0.005698117916875216,
          0.03310003433603227,   0.041724722742120926,  0.13248618961456732,
          0.10004565910652752,   0.10868713291440213,   0.120614402205249,
          0.25027231329052646,   0.029537942516907823,  0.3890220620427618,
          0.08737846516384448,   0.35850929642766155,   0.07631190151295938,
          0.18686917947622156,   0.002057530965370865,  0.4147059095903063,
          0.1704787284972489,    0.31941530538343876,   0.007999608091484301,
          0.14373762619976402,   0.05116587368513777,   0.2837881388594704,
          0.02278459925089566,   0.21654666647347712,   0.009473297912213558,
          0.31289850307488,      0.0004640077321756526, 0.22643479740771752,
          0.9151336840842468,    0.004794660975436677,  0.9392011922216335,
          0.029155196206835834,  0.8984105884621028,    0.02620936402249865,
          0.9612018477470925,    0.005698117916875216,  0.8257890876433118,
          0.041724722742120926,  0.7912672079790704,    0.10004565910652752,
          0.6291132845042245,    0.120614402205249,     0.5814399954403304,
          0.029537942516907823,  0.554112238408494,     0.08737846516384448,
          0.736818919010819,     0.07631190151295938,   0.5832365594443228,
          0.002057530965370865,  0.5101059661193124,    0.1704787284972489,
          0.8482627657087517,    0.007999608091484301,  0.6650459874553918,
          0.05116587368513777,   0.7606687342756272,    0.02278459925089566,
          0.6776281990129065,    0.009473297912213558,  0.7731011948601068,
          0.0004640077321756526, 0.08007165494031654,   0.9151336840842468,
          0.031643611571530776,  0.9392011922216335,    0.07538004751539866,
          0.8984105884621028,    0.03310003433603227,   0.9612018477470925,
          0.13248618961456732,   0.8257890876433118,    0.10868713291440213,
          0.7912672079790704,    0.25027231329052646,   0.6291132845042245,
          0.3890220620427618,    0.5814399954403304,    0.35850929642766155,
          0.554112238408494,     0.18686917947622156,   0.736818919010819,
          0.4147059095903063,    0.5832365594443228,    0.31941530538343876,
          0.5101059661193124,    0.14373762619976402,   0.8482627657087517,
          0.2837881388594704,    0.6650459874553918,    0.21654666647347712,
          0.7606687342756272,    0.31289850307488,      0.6776281990129065,
          0.22643479740771752,   0.7731011948601068,    0.08007165494031654,
          0.004794660975436677,  0.031643611571530776,  0.029155196206835834,
          0.07538004751539866,   0.02620936402249865,   0.03310003433603227,
          0.005698117916875216,  0.13248618961456732,   0.041724722742120926,
          0.10868713291440213,   0.10004565910652752,   0.25027231329052646,
          0.120614402205249,     0.3890220620427618,    0.029537942516907823,
          0.35850929642766155,   0.08737846516384448,   0.18686917947622156,
          0.07631190151295938,   0.4147059095903063,    0.002057530965370865,
          0.31941530538343876,   0.1704787284972489,    0.14373762619976402,
          0.007999608091484301,  0.2837881388594704,    0.05116587368513777,
          0.21654666647347712,   0.02278459925089566,   0.31289850307488,
          0.009473297912213558,  0.22643479740771752,   0.0004640077321756526,
          0.9151336840842468,    0.08007165494031654,   0.9392011922216335,
          0.031643611571530776,  0.8984105884621028,    0.07538004751539866,
          0.9612018477470925,    0.03310003433603227,   0.8257890876433118,
          0.13248618961456732,   0.7912672079790704,    0.10868713291440213,
          0.6291132845042245,    0.25027231329052646,   0.5814399954403304,
          0.3890220620427618,    0.554112238408494,     0.35850929642766155,
          0.736818919010819,     0.18686917947622156,   0.5832365594443228,
          0.4147059095903063,    0.5101059661193124,    0.31941530538343876,
          0.8482627657087517,    0.14373762619976402,   0.6650459874553918,
          0.2837881388594704,    0.7606687342756272,    0.21654666647347712,
          0.6776281990129065,    0.31289850307488,      0.7731011948601068,
          0.22643479740771752,   0.004794660975436677,  0.9151336840842468,
          0.029155196206835834,  0.9392011922216335,    0.02620936402249865,
          0.8984105884621028,    0.005698117916875216,  0.9612018477470925,
          0.041724722742120926,  0.8257890876433118,    0.10004565910652752,
          0.7912672079790704,    0.120614402205249,     0.6291132845042245,
          0.029537942516907823,  0.5814399954403304,    0.08737846516384448,
          0.554112238408494,     0.07631190151295938,   0.736818919010819,
          0.002057530965370865,  0.5832365594443228,    0.1704787284972489,
          0.5101059661193124,    0.007999608091484301,  0.8482627657087517,
          0.05116587368513777,   0.6650459874553918,    0.02278459925089566,
          0.7606687342756272,    0.009473297912213558,  0.6776281990129065,
          0.0004640077321756526, 0.7731011948601068};
      std::vector<T> w = {
          0.010243331294611621,  0.002456912651483009,  0.00026347655834093594,
          0.002651079590933673,  0.00973403391859144,   0.00976782346162377,
          0.005764251817328446,  0.006627629724272634,  0.008472172539264045,
          0.008206200301293952,  0.002456912651483009,  0.00026347655834093594,
          0.002651079590933673,  0.00973403391859144,   0.00976782346162377,
          0.005764251817328446,  0.006627629724272634,  0.008472172539264045,
          0.008206200301293952,  0.002456912651483009,  0.00026347655834093594,
          0.002651079590933673,  0.00973403391859144,   0.00976782346162377,
          0.005764251817328446,  0.006627629724272634,  0.008472172539264045,
          0.008206200301293952,  0.0006992632240801362, 0.0006027823868584428,
          0.0016527723564838351, 0.0005428536714983775, 0.0032017989498564093,
          0.002307105538189159,  0.00718973661379937,   0.00412988360854342,
          0.006863979108042852,  0.005198822764087162,  0.000928573735499042,
          0.008799583590347606,  0.0014833808313282528, 0.005053562216044342,
          0.0031346689230402846, 0.0022957791936993187, 0.0005697744579341068,
          0.0006992632240801362, 0.0006027823868584428, 0.0016527723564838351,
          0.0005428536714983775, 0.0032017989498564093, 0.002307105538189159,
          0.00718973661379937,   0.00412988360854342,   0.006863979108042852,
          0.005198822764087162,  0.000928573735499042,  0.008799583590347606,
          0.0014833808313282528, 0.005053562216044342,  0.0031346689230402846,
          0.0022957791936993187, 0.0005697744579341068, 0.0006992632240801362,
          0.0006027823868584428, 0.0016527723564838351, 0.0005428536714983775,
          0.0032017989498564093, 0.002307105538189159,  0.00718973661379937,
          0.00412988360854342,   0.006863979108042852,  0.005198822764087162,
          0.000928573735499042,  0.008799583590347606,  0.0014833808313282528,
          0.005053562216044342,  0.0031346689230402846, 0.0022957791936993187,
          0.0005697744579341068, 0.0006992632240801362, 0.0006027823868584428,
          0.0016527723564838351, 0.0005428536714983775, 0.0032017989498564093,
          0.002307105538189159,  0.00718973661379937,   0.00412988360854342,
          0.006863979108042852,  0.005198822764087162,  0.000928573735499042,
          0.008799583590347606,  0.0014833808313282528, 0.005053562216044342,
          0.0031346689230402846, 0.0022957791936993187, 0.0005697744579341068,
          0.0006992632240801362, 0.0006027823868584428, 0.0016527723564838351,
          0.0005428536714983775, 0.0032017989498564093, 0.002307105538189159,
          0.00718973661379937,   0.00412988360854342,   0.006863979108042852,
          0.005198822764087162,  0.000928573735499042,  0.008799583590347606,
          0.0014833808313282528, 0.005053562216044342,  0.0031346689230402846,
          0.0022957791936993187, 0.0005697744579341068, 0.0006992632240801362,
          0.0006027823868584428, 0.0016527723564838351, 0.0005428536714983775,
          0.0032017989498564093, 0.002307105538189159,  0.00718973661379937,
          0.00412988360854342,   0.006863979108042852,  0.005198822764087162,
          0.000928573735499042,  0.008799583590347606,  0.0014833808313282528,
          0.005053562216044342,  0.0031346689230402846, 0.0022957791936993187,
          0.0005697744579341068};
      return {std::move(x), std::move(w)};
    }
    else if (m == 27)
    {
      // Xiao Gimbutas, 3 points, degree 27
      std::vector<T> x = {
          0.3807140211811872,    0.3807140211811872,    0.4466678037038646,
          0.4466678037038646,    0.41614137880541213,   0.41614137880541213,
          0.08030464778843843,   0.08030464778843843,   0.23340040666987116,
          0.23340040666987116,   0.3011654651665092,    0.3011654651665092,
          0.17477996635490006,   0.17477996635490006,   0.48556505418516277,
          0.48556505418516277,   0.03257152018018172,   0.03257152018018172,
          0.12757090190467762,   0.12757090190467762,   0.0066392191809588885,
          0.0066392191809588885, 0.3807140211811872,    0.23857195763762562,
          0.4466678037038646,    0.10666439259227078,   0.41614137880541213,
          0.16771724238917574,   0.08030464778843843,   0.8393907044231231,
          0.23340040666987116,   0.5331991866602577,    0.3011654651665092,
          0.39766906966698157,   0.17477996635490006,   0.6504400672901999,
          0.48556505418516277,   0.02886989162967446,   0.03257152018018172,
          0.9348569596396366,    0.12757090190467762,   0.7448581961906448,
          0.0066392191809588885, 0.9867215616380822,    0.23857195763762562,
          0.3807140211811872,    0.10666439259227078,   0.4466678037038646,
          0.16771724238917574,   0.41614137880541213,   0.8393907044231231,
          0.08030464778843843,   0.5331991866602577,    0.23340040666987116,
          0.39766906966698157,   0.3011654651665092,    0.6504400672901999,
          0.17477996635490006,   0.02886989162967446,   0.48556505418516277,
          0.9348569596396366,    0.03257152018018172,   0.7448581961906448,
          0.12757090190467762,   0.9867215616380822,    0.0066392191809588885,
          0.030730604727272855,  0.2870421965934966,    0.12915264006344968,
          0.3450878417155684,    0.028033486095250002,  0.3759301570486618,
          0.20913092113766868,   0.31694558893313196,   0.06603891284973865,
          0.4072283930427199,    0.041030576819181826,  0.21355359845782393,
          0.005299640371799034,  0.32885287806889263,   0.06307399541495087,
          0.13929530614214874,   0.1489628509382401,    0.25524625469697804,
          0.09469708243313069,   0.20837601560037405,   0.005580717015260116,
          0.44001055194621547,   0.07507690243319622,   0.3022209412278211,
          0.0069825293244590156, 0.08194680258353369,   0.0060935694037648315,
          0.03436496991214199,   0.03503442252769738,   0.08011207384710112,
          0.019352001318038967,  0.14721343189892247,   0.007332472549040455,
          0.22971965325784321,   0.0004903284434629743, 0.1476555211198698,
          0.6822271986792305,    0.030730604727272855,  0.5257595182209819,
          0.12915264006344968,   0.5960363568560882,    0.028033486095250002,
          0.4739234899291994,    0.20913092113766868,   0.5267326941075414,
          0.06603891284973865,   0.7454158247229942,    0.041030576819181826,
          0.6658474815593083,    0.005299640371799034,  0.7976306984429005,
          0.06307399541495087,   0.5957908943647818,    0.1489628509382401,
          0.6969269019664952,    0.09469708243313069,   0.5544087310385244,
          0.005580717015260116,  0.6227021563389827,    0.07507690243319622,
          0.9110706680920073,    0.0069825293244590156, 0.9595414606840932,
          0.0060935694037648315, 0.8848535036252014,    0.03503442252769738,
          0.8334345667830386,    0.019352001318038967,  0.7629478741931164,
          0.007332472549040455,  0.8518541504366672,    0.0004903284434629743,
          0.2870421965934966,    0.6822271986792305,    0.3450878417155684,
          0.5257595182209819,    0.3759301570486618,    0.5960363568560882,
          0.31694558893313196,   0.4739234899291994,    0.4072283930427199,
          0.5267326941075414,    0.21355359845782393,   0.7454158247229942,
          0.32885287806889263,   0.6658474815593083,    0.13929530614214874,
          0.7976306984429005,    0.25524625469697804,   0.5957908943647818,
          0.20837601560037405,   0.6969269019664952,    0.44001055194621547,
          0.5544087310385244,    0.3022209412278211,    0.6227021563389827,
          0.08194680258353369,   0.9110706680920073,    0.03436496991214199,
          0.9595414606840932,    0.08011207384710112,   0.8848535036252014,
          0.14721343189892247,   0.8334345667830386,    0.22971965325784321,
          0.7629478741931164,    0.1476555211198698,    0.8518541504366672,
          0.2870421965934966,    0.030730604727272855,  0.3450878417155684,
          0.12915264006344968,   0.3759301570486618,    0.028033486095250002,
          0.31694558893313196,   0.20913092113766868,   0.4072283930427199,
          0.06603891284973865,   0.21355359845782393,   0.041030576819181826,
          0.32885287806889263,   0.005299640371799034,  0.13929530614214874,
          0.06307399541495087,   0.25524625469697804,   0.1489628509382401,
          0.20837601560037405,   0.09469708243313069,   0.44001055194621547,
          0.005580717015260116,  0.3022209412278211,    0.07507690243319622,
          0.08194680258353369,   0.0069825293244590156, 0.03436496991214199,
          0.0060935694037648315, 0.08011207384710112,   0.03503442252769738,
          0.14721343189892247,   0.019352001318038967,  0.22971965325784321,
          0.007332472549040455,  0.1476555211198698,    0.0004903284434629743,
          0.6822271986792305,    0.2870421965934966,    0.5257595182209819,
          0.3450878417155684,    0.5960363568560882,    0.3759301570486618,
          0.4739234899291994,    0.31694558893313196,   0.5267326941075414,
          0.4072283930427199,    0.7454158247229942,    0.21355359845782393,
          0.6658474815593083,    0.32885287806889263,   0.7976306984429005,
          0.13929530614214874,   0.5957908943647818,    0.25524625469697804,
          0.6969269019664952,    0.20837601560037405,   0.5544087310385244,
          0.44001055194621547,   0.6227021563389827,    0.3022209412278211,
          0.9110706680920073,    0.08194680258353369,   0.9595414606840932,
          0.03436496991214199,   0.8848535036252014,    0.08011207384710112,
          0.8334345667830386,    0.14721343189892247,   0.7629478741931164,
          0.22971965325784321,   0.8518541504366672,    0.1476555211198698,
          0.030730604727272855,  0.6822271986792305,    0.12915264006344968,
          0.5257595182209819,    0.028033486095250002,  0.5960363568560882,
          0.20913092113766868,   0.4739234899291994,    0.06603891284973865,
          0.5267326941075414,    0.041030576819181826,  0.7454158247229942,
          0.005299640371799034,  0.6658474815593083,    0.06307399541495087,
          0.7976306984429005,    0.1489628509382401,    0.5957908943647818,
          0.09469708243313069,   0.6969269019664952,    0.005580717015260116,
          0.5544087310385244,    0.07507690243319622,   0.6227021563389827,
          0.0069825293244590156, 0.9110706680920073,    0.0060935694037648315,
          0.9595414606840932,    0.03503442252769738,   0.8848535036252014,
          0.019352001318038967,  0.8334345667830386,    0.007332472549040455,
          0.7629478741931164,    0.0004903284434629743, 0.8518541504366672};
      std::vector<T> w = {
          0.00478004248372996,   0.004705079904727113,  0.006025113512075217,
          0.0026063109364009383, 0.006735657699024688,  0.007873982890681327,
          0.00564122127234919,   0.003558618706437321,  0.0013886697644770905,
          0.004871622461408866,  0.0002877212028352512, 0.00478004248372996,
          0.004705079904727113,  0.006025113512075217,  0.0026063109364009383,
          0.006735657699024688,  0.007873982890681327,  0.00564122127234919,
          0.003558618706437321,  0.0013886697644770905, 0.004871622461408866,
          0.0002877212028352512, 0.00478004248372996,   0.004705079904727113,
          0.006025113512075217,  0.0026063109364009383, 0.006735657699024688,
          0.007873982890681327,  0.00564122127234919,   0.003558618706437321,
          0.0013886697644770905, 0.004871622461408866,  0.0002877212028352512,
          0.0027658974168833657, 0.006278718102018268,  0.0031975763497272013,
          0.00685769661527542,   0.004931135059494785,  0.003227686452464845,
          0.0014639131808995513, 0.003565317655243512,  0.006173831565430682,
          0.005346850294808132,  0.0016212337988196705, 0.005465305546456643,
          0.0010075615636448512, 0.0005983868042365814, 0.0021635790176803564,
          0.002311193555890555,  0.0016971268694035136, 0.00042330306788192527,
          0.0027658974168833657, 0.006278718102018268,  0.0031975763497272013,
          0.00685769661527542,   0.004931135059494785,  0.003227686452464845,
          0.0014639131808995513, 0.003565317655243512,  0.006173831565430682,
          0.005346850294808132,  0.0016212337988196705, 0.005465305546456643,
          0.0010075615636448512, 0.0005983868042365814, 0.0021635790176803564,
          0.002311193555890555,  0.0016971268694035136, 0.00042330306788192527,
          0.0027658974168833657, 0.006278718102018268,  0.0031975763497272013,
          0.00685769661527542,   0.004931135059494785,  0.003227686452464845,
          0.0014639131808995513, 0.003565317655243512,  0.006173831565430682,
          0.005346850294808132,  0.0016212337988196705, 0.005465305546456643,
          0.0010075615636448512, 0.0005983868042365814, 0.0021635790176803564,
          0.002311193555890555,  0.0016971268694035136, 0.00042330306788192527,
          0.0027658974168833657, 0.006278718102018268,  0.0031975763497272013,
          0.00685769661527542,   0.004931135059494785,  0.003227686452464845,
          0.0014639131808995513, 0.003565317655243512,  0.006173831565430682,
          0.005346850294808132,  0.0016212337988196705, 0.005465305546456643,
          0.0010075615636448512, 0.0005983868042365814, 0.0021635790176803564,
          0.002311193555890555,  0.0016971268694035136, 0.00042330306788192527,
          0.0027658974168833657, 0.006278718102018268,  0.0031975763497272013,
          0.00685769661527542,   0.004931135059494785,  0.003227686452464845,
          0.0014639131808995513, 0.003565317655243512,  0.006173831565430682,
          0.005346850294808132,  0.0016212337988196705, 0.005465305546456643,
          0.0010075615636448512, 0.0005983868042365814, 0.0021635790176803564,
          0.002311193555890555,  0.0016971268694035136, 0.00042330306788192527,
          0.0027658974168833657, 0.006278718102018268,  0.0031975763497272013,
          0.00685769661527542,   0.004931135059494785,  0.003227686452464845,
          0.0014639131808995513, 0.003565317655243512,  0.006173831565430682,
          0.005346850294808132,  0.0016212337988196705, 0.005465305546456643,
          0.0010075615636448512, 0.0005983868042365814, 0.0021635790176803564,
          0.002311193555890555,  0.0016971268694035136, 0.00042330306788192527};
      return {std::move(x), std::move(w)};
    }
    else if (m == 28)
    {
      // Xiao Gimbutas, 3 points, degree 28
      std::vector<T> x = {
          0.3039829225164842,    0.3039829225164842,    0.004804126196658098,
          0.004804126196658098,  0.45827990424041193,   0.45827990424041193,
          0.38626797357004206,   0.38626797357004206,   0.2582640721504622,
          0.2582640721504622,    0.10589584417862768,   0.10589584417862768,
          0.42955220211889933,   0.42955220211889933,   0.4848411325625894,
          0.4848411325625894,    0.15863768886305973,   0.15863768886305973,
          0.06083919239275881,   0.06083919239275881,   0.3039829225164842,
          0.39203415496703165,   0.004804126196658098,  0.9903917476066838,
          0.45827990424041193,   0.08344019151917614,   0.38626797357004206,
          0.2274640528599159,    0.2582640721504622,    0.4834718556990756,
          0.10589584417862768,   0.7882083116427446,    0.42955220211889933,
          0.14089559576220134,   0.4848411325625894,    0.030317734874821145,
          0.15863768886305973,   0.6827246222738805,    0.06083919239275881,
          0.8783216152144824,    0.39203415496703165,   0.3039829225164842,
          0.9903917476066838,    0.004804126196658098,  0.08344019151917614,
          0.45827990424041193,   0.2274640528599159,    0.38626797357004206,
          0.4834718556990756,    0.2582640721504622,    0.7882083116427446,
          0.10589584417862768,   0.14089559576220134,   0.42955220211889933,
          0.030317734874821145,  0.4848411325625894,    0.6827246222738805,
          0.15863768886305973,   0.8783216152144824,    0.06083919239275881,
          0.02152438536945612,   0.0455054005583464,    0.04906966935755949,
          0.2133944547670873,    0.17765845029637026,   0.24210251191931964,
          0.1898123562927368,    0.32719073201917004,   0.0044583820232893204,
          0.14199816693317424,   0.08767797648435202,   0.17539639319146172,
          0.06318032763441064,   0.39213961333441455,   0.004149464133923672,
          0.33414561503592133,   0.022794804925916238,  0.17414619605118214,
          0.022700844371797004,  0.2758390080718242,    0.006149648542663968,
          0.026222667164652273,  0.11203362934227094,   0.24304720236592617,
          0.004781489772987132,  0.22999298405790716,   0.062448742179632866,
          0.29560828087240165,   0.050211185913428096,  0.12139345075409119,
          0.025727998742878733,  0.3738238003102097,    0.005646565993466159,
          0.44278340652024356,   0.11808906971509502,   0.3377986632005823,
          0.018242291012294715,  0.09222918919528221,   0.0012002556014871519,
          0.07029074047813273,   0.9329702140721975,    0.02152438536945612,
          0.7375358758753532,    0.04906966935755949,   0.5802390377843101,
          0.17765845029637026,   0.4829969116880932,    0.1898123562927368,
          0.8535434510435365,    0.0044583820232893204, 0.7369256303241862,
          0.08767797648435202,   0.5446800590311749,    0.06318032763441064,
          0.661704920830155,     0.004149464133923672,  0.8030589990229016,
          0.022794804925916238,  0.7014601475563789,    0.022700844371797004,
          0.9676276842926838,    0.006149648542663968,  0.644919168291803,
          0.11203362934227094,   0.7652255261691058,    0.004781489772987132,
          0.6419429769479654,    0.062448742179632866,  0.8283953633324808,
          0.050211185913428096,  0.6004482009469116,    0.025727998742878733,
          0.5515700274862902,    0.005646565993466159,  0.5441122670843226,
          0.11808906971509502,   0.8895285197924231,    0.018242291012294715,
          0.92850900392038,      0.0012002556014871519, 0.0455054005583464,
          0.9329702140721975,    0.2133944547670873,    0.7375358758753532,
          0.24210251191931964,   0.5802390377843101,    0.32719073201917004,
          0.4829969116880932,    0.14199816693317424,   0.8535434510435365,
          0.17539639319146172,   0.7369256303241862,    0.39213961333441455,
          0.5446800590311749,    0.33414561503592133,   0.661704920830155,
          0.17414619605118214,   0.8030589990229016,    0.2758390080718242,
          0.7014601475563789,    0.026222667164652273,  0.9676276842926838,
          0.24304720236592617,   0.644919168291803,     0.22999298405790716,
          0.7652255261691058,    0.29560828087240165,   0.6419429769479654,
          0.12139345075409119,   0.8283953633324808,    0.3738238003102097,
          0.6004482009469116,    0.44278340652024356,   0.5515700274862902,
          0.3377986632005823,    0.5441122670843226,    0.09222918919528221,
          0.8895285197924231,    0.07029074047813273,   0.92850900392038,
          0.0455054005583464,    0.02152438536945612,   0.2133944547670873,
          0.04906966935755949,   0.24210251191931964,   0.17765845029637026,
          0.32719073201917004,   0.1898123562927368,    0.14199816693317424,
          0.0044583820232893204, 0.17539639319146172,   0.08767797648435202,
          0.39213961333441455,   0.06318032763441064,   0.33414561503592133,
          0.004149464133923672,  0.17414619605118214,   0.022794804925916238,
          0.2758390080718242,    0.022700844371797004,  0.026222667164652273,
          0.006149648542663968,  0.24304720236592617,   0.11203362934227094,
          0.22999298405790716,   0.004781489772987132,  0.29560828087240165,
          0.062448742179632866,  0.12139345075409119,   0.050211185913428096,
          0.3738238003102097,    0.025727998742878733,  0.44278340652024356,
          0.005646565993466159,  0.3377986632005823,    0.11808906971509502,
          0.09222918919528221,   0.018242291012294715,  0.07029074047813273,
          0.0012002556014871519, 0.9329702140721975,    0.0455054005583464,
          0.7375358758753532,    0.2133944547670873,    0.5802390377843101,
          0.24210251191931964,   0.4829969116880932,    0.32719073201917004,
          0.8535434510435365,    0.14199816693317424,   0.7369256303241862,
          0.17539639319146172,   0.5446800590311749,    0.39213961333441455,
          0.661704920830155,     0.33414561503592133,   0.8030589990229016,
          0.17414619605118214,   0.7014601475563789,    0.2758390080718242,
          0.9676276842926838,    0.026222667164652273,  0.644919168291803,
          0.24304720236592617,   0.7652255261691058,    0.22999298405790716,
          0.6419429769479654,    0.29560828087240165,   0.8283953633324808,
          0.12139345075409119,   0.6004482009469116,    0.3738238003102097,
          0.5515700274862902,    0.44278340652024356,   0.5441122670843226,
          0.3377986632005823,    0.8895285197924231,    0.09222918919528221,
          0.92850900392038,      0.07029074047813273,   0.02152438536945612,
          0.9329702140721975,    0.04906966935755949,   0.7375358758753532,
          0.17765845029637026,   0.5802390377843101,    0.1898123562927368,
          0.4829969116880932,    0.0044583820232893204, 0.8535434510435365,
          0.08767797648435202,   0.7369256303241862,    0.06318032763441064,
          0.5446800590311749,    0.004149464133923672,  0.661704920830155,
          0.022794804925916238,  0.8030589990229016,    0.022700844371797004,
          0.7014601475563789,    0.006149648542663968,  0.9676276842926838,
          0.11203362934227094,   0.644919168291803,     0.004781489772987132,
          0.7652255261691058,    0.062448742179632866,  0.6419429769479654,
          0.050211185913428096,  0.8283953633324808,    0.025727998742878733,
          0.6004482009469116,    0.005646565993466159,  0.5515700274862902,
          0.11808906971509502,   0.5441122670843226,    0.018242291012294715,
          0.8895285197924231,    0.0012002556014871519, 0.92850900392038};
      std::vector<T> w = {0.007181233150323068,  0.00015556760434074816,
                          0.0044258525054465805, 0.007105293195224093,
                          0.006546374294544483,  0.0039049716855431705,
                          0.006773797801662755,  0.003841055289297512,
                          0.005971334820124114,  0.0025641260023390343,
                          0.007181233150323068,  0.00015556760434074816,
                          0.0044258525054465805, 0.007105293195224093,
                          0.006546374294544483,  0.0039049716855431705,
                          0.006773797801662755,  0.003841055289297512,
                          0.005971334820124114,  0.0025641260023390343,
                          0.007181233150323068,  0.00015556760434074816,
                          0.0044258525054465805, 0.007105293195224093,
                          0.006546374294544483,  0.0039049716855431705,
                          0.006773797801662755,  0.003841055289297512,
                          0.005971334820124114,  0.0025641260023390343,
                          0.0010587697788404492, 0.002947836117257123,
                          0.006278628108338439,  0.006553057062049693,
                          0.0008894477096277566, 0.004005964643347482,
                          0.004232347867138302,  0.0011883821338438917,
                          0.0021384711675089727, 0.0025656386910186484,
                          0.0004742702129447003, 0.0051229932808521,
                          0.0011622726822320252, 0.004462558070138275,
                          0.0029589418861769453, 0.003130267321484276,
                          0.0015700888217440182, 0.0063935692114779,
                          0.0016024580953965175, 0.0003625672974930415,
                          0.0010587697788404492, 0.002947836117257123,
                          0.006278628108338439,  0.006553057062049693,
                          0.0008894477096277566, 0.004005964643347482,
                          0.004232347867138302,  0.0011883821338438917,
                          0.0021384711675089727, 0.0025656386910186484,
                          0.0004742702129447003, 0.0051229932808521,
                          0.0011622726822320252, 0.004462558070138275,
                          0.0029589418861769453, 0.003130267321484276,
                          0.0015700888217440182, 0.0063935692114779,
                          0.0016024580953965175, 0.0003625672974930415,
                          0.0010587697788404492, 0.002947836117257123,
                          0.006278628108338439,  0.006553057062049693,
                          0.0008894477096277566, 0.004005964643347482,
                          0.004232347867138302,  0.0011883821338438917,
                          0.0021384711675089727, 0.0025656386910186484,
                          0.0004742702129447003, 0.0051229932808521,
                          0.0011622726822320252, 0.004462558070138275,
                          0.0029589418861769453, 0.003130267321484276,
                          0.0015700888217440182, 0.0063935692114779,
                          0.0016024580953965175, 0.0003625672974930415,
                          0.0010587697788404492, 0.002947836117257123,
                          0.006278628108338439,  0.006553057062049693,
                          0.0008894477096277566, 0.004005964643347482,
                          0.004232347867138302,  0.0011883821338438917,
                          0.0021384711675089727, 0.0025656386910186484,
                          0.0004742702129447003, 0.0051229932808521,
                          0.0011622726822320252, 0.004462558070138275,
                          0.0029589418861769453, 0.003130267321484276,
                          0.0015700888217440182, 0.0063935692114779,
                          0.0016024580953965175, 0.0003625672974930415,
                          0.0010587697788404492, 0.002947836117257123,
                          0.006278628108338439,  0.006553057062049693,
                          0.0008894477096277566, 0.004005964643347482,
                          0.004232347867138302,  0.0011883821338438917,
                          0.0021384711675089727, 0.0025656386910186484,
                          0.0004742702129447003, 0.0051229932808521,
                          0.0011622726822320252, 0.004462558070138275,
                          0.0029589418861769453, 0.003130267321484276,
                          0.0015700888217440182, 0.0063935692114779,
                          0.0016024580953965175, 0.0003625672974930415,
                          0.0010587697788404492, 0.002947836117257123,
                          0.006278628108338439,  0.006553057062049693,
                          0.0008894477096277566, 0.004005964643347482,
                          0.004232347867138302,  0.0011883821338438917,
                          0.0021384711675089727, 0.0025656386910186484,
                          0.0004742702129447003, 0.0051229932808521,
                          0.0011622726822320252, 0.004462558070138275,
                          0.0029589418861769453, 0.003130267321484276,
                          0.0015700888217440182, 0.0063935692114779,
                          0.0016024580953965175, 0.0003625672974930415};
      return {std::move(x), std::move(w)};
    }
    else if (m == 29)
    {
      // Xiao Gimbutas, 3 points, degree 29
      std::vector<T> x = {
          0.49891482463768616,  0.49891482463768616,    0.4343804267617306,
          0.4343804267617306,   0.0410973356271182,     0.0410973356271182,
          0.2084053051324009,   0.2084053051324009,     0.16074588443196364,
          0.16074588443196364,  0.48840160293260276,    0.48840160293260276,
          0.3023864112151285,   0.3023864112151285,     0.11442681299442559,
          0.11442681299442559,  0.46476243108073895,    0.46476243108073895,
          0.07372188139009989,  0.07372188139009989,    0.39061917878326374,
          0.39061917878326374,  0.49891482463768616,    0.0021703507246276788,
          0.4343804267617306,   0.13123914647653878,    0.0410973356271182,
          0.9178053287457636,   0.2084053051324009,     0.5831893897351982,
          0.16074588443196364,  0.6785082311360727,     0.48840160293260276,
          0.023196794134794474, 0.3023864112151285,     0.395227177569743,
          0.11442681299442559,  0.7711463740111488,     0.46476243108073895,
          0.0704751378385221,   0.07372188139009989,    0.8525562372198002,
          0.39061917878326374,  0.21876164243347251,    0.0021703507246276788,
          0.49891482463768616,  0.13123914647653878,    0.4343804267617306,
          0.9178053287457636,   0.0410973356271182,     0.5831893897351982,
          0.2084053051324009,   0.6785082311360727,     0.16074588443196364,
          0.023196794134794474, 0.48840160293260276,    0.395227177569743,
          0.3023864112151285,   0.7711463740111488,     0.11442681299442559,
          0.0704751378385221,   0.46476243108073895,    0.8525562372198002,
          0.07372188139009989,  0.21876164243347251,    0.39061917878326374,
          0.002728743247921069, 0.058942108840229206,   0.15717769986719343,
          0.34978801000933185,  0.0021009666448275587,  0.32300182354355017,
          0.06816580881374641,  0.15814585424951613,    0.010830958603609348,
          0.029549468261353372, 0.21893234198017247,    0.29181917342653707,
          0.02128689624073325,  0.0755221785129958,     0.040847216576102435,
          0.11711666950889889,  0.001603496496043763,   0.011166218108169191,
          0.10154598522683399,  0.20804564927908714,    0.04152706126882266,
          0.39221011498043484,  0.0938490411451324,     0.3597112755099759,
          0.008686029804384135, 0.24587469948287544,    0.01758912404404562,
          0.16700273817492314,  0.005523524512212553,   0.11500859863194643,
          0.0238589269426556,   0.31539539811731915,    0.04029533454477179,
          0.2232226502248206,   0.06787840431144707,    0.2883958599187324,
          0.13953560718108263,  0.2673663502727756,     0.008066585704166612,
          0.40576539529889155,  0.00012344681228740494, 0.18632072767535954,
          0.9383291479118497,   0.002728743247921069,   0.4930342901234747,
          0.15717769986719343,  0.6748972098116224,     0.0021009666448275587,
          0.7736883369367374,   0.06816580881374641,    0.9596195731350372,
          0.010830958603609348, 0.4892484845932904,     0.21893234198017247,
          0.903190925246271,    0.02128689624073325,    0.8420361139149987,
          0.040847216576102435, 0.987230285395787,      0.001603496496043763,
          0.6904083654940789,   0.10154598522683399,    0.5662628237507425,
          0.04152706126882266,  0.5464396833448917,     0.0938490411451324,
          0.7454392707127404,   0.008686029804384135,   0.8154081377810312,
          0.01758912404404562,  0.8794678768558409,     0.005523524512212553,
          0.6607456749400253,   0.0238589269426556,     0.7364820152304077,
          0.04029533454477179,  0.6437257357698205,     0.06787840431144707,
          0.5930980425461418,   0.13953560718108263,    0.5861680189969418,
          0.008066585704166612, 0.8135558255123531,     0.00012344681228740494,
          0.058942108840229206, 0.9383291479118497,     0.34978801000933185,
          0.4930342901234747,   0.32300182354355017,    0.6748972098116224,
          0.15814585424951613,  0.7736883369367374,     0.029549468261353372,
          0.9596195731350372,   0.29181917342653707,    0.4892484845932904,
          0.0755221785129958,   0.903190925246271,      0.11711666950889889,
          0.8420361139149987,   0.011166218108169191,   0.987230285395787,
          0.20804564927908714,  0.6904083654940789,     0.39221011498043484,
          0.5662628237507425,   0.3597112755099759,     0.5464396833448917,
          0.24587469948287544,  0.7454392707127404,     0.16700273817492314,
          0.8154081377810312,   0.11500859863194643,    0.8794678768558409,
          0.31539539811731915,  0.6607456749400253,     0.2232226502248206,
          0.7364820152304077,   0.2883958599187324,     0.6437257357698205,
          0.2673663502727756,   0.5930980425461418,     0.40576539529889155,
          0.5861680189969418,   0.18632072767535954,    0.8135558255123531,
          0.058942108840229206, 0.002728743247921069,   0.34978801000933185,
          0.15717769986719343,  0.32300182354355017,    0.0021009666448275587,
          0.15814585424951613,  0.06816580881374641,    0.029549468261353372,
          0.010830958603609348, 0.29181917342653707,    0.21893234198017247,
          0.0755221785129958,   0.02128689624073325,    0.11711666950889889,
          0.040847216576102435, 0.011166218108169191,   0.001603496496043763,
          0.20804564927908714,  0.10154598522683399,    0.39221011498043484,
          0.04152706126882266,  0.3597112755099759,     0.0938490411451324,
          0.24587469948287544,  0.008686029804384135,   0.16700273817492314,
          0.01758912404404562,  0.11500859863194643,    0.005523524512212553,
          0.31539539811731915,  0.0238589269426556,     0.2232226502248206,
          0.04029533454477179,  0.2883958599187324,     0.06787840431144707,
          0.2673663502727756,   0.13953560718108263,    0.40576539529889155,
          0.008066585704166612, 0.18632072767535954,    0.00012344681228740494,
          0.9383291479118497,   0.058942108840229206,   0.4930342901234747,
          0.34978801000933185,  0.6748972098116224,     0.32300182354355017,
          0.7736883369367374,   0.15814585424951613,    0.9596195731350372,
          0.029549468261353372, 0.4892484845932904,     0.29181917342653707,
          0.903190925246271,    0.0755221785129958,     0.8420361139149987,
          0.11711666950889889,  0.987230285395787,      0.011166218108169191,
          0.6904083654940789,   0.20804564927908714,    0.5662628237507425,
          0.39221011498043484,  0.5464396833448917,     0.3597112755099759,
          0.7454392707127404,   0.24587469948287544,    0.8154081377810312,
          0.16700273817492314,  0.8794678768558409,     0.11500859863194643,
          0.6607456749400253,   0.31539539811731915,    0.7364820152304077,
          0.2232226502248206,   0.6437257357698205,     0.2883958599187324,
          0.5930980425461418,   0.2673663502727756,     0.5861680189969418,
          0.40576539529889155,  0.8135558255123531,     0.18632072767535954,
          0.002728743247921069, 0.9383291479118497,     0.15717769986719343,
          0.4930342901234747,   0.0021009666448275587,  0.6748972098116224,
          0.06816580881374641,  0.7736883369367374,     0.010830958603609348,
          0.9596195731350372,   0.21893234198017247,    0.4892484845932904,
          0.02128689624073325,  0.903190925246271,      0.040847216576102435,
          0.8420361139149987,   0.001603496496043763,   0.987230285395787,
          0.10154598522683399,  0.6904083654940789,     0.04152706126882266,
          0.5662628237507425,   0.0938490411451324,     0.5464396833448917,
          0.008686029804384135, 0.7454392707127404,     0.01758912404404562,
          0.8154081377810312,   0.005523524512212553,   0.8794678768558409,
          0.0238589269426556,   0.6607456749400253,     0.04029533454477179,
          0.7364820152304077,   0.06787840431144707,    0.6437257357698205,
          0.13953560718108263,  0.5930980425461418,     0.008066585704166612,
          0.5861680189969418,   0.00012344681228740494, 0.8135558255123531};
      std::vector<T> w = {
          0.0007582310515784511,  0.005585550147583929,  0.0014302457530784435,
          0.006269601721311614,   0.005285708929113789,  0.0030672005823594553,
          0.008155119403871603,   0.004086439220613566,  0.00515655831762925,
          0.002804423566415652,   0.007956607642044452,  0.0007582310515784511,
          0.005585550147583929,   0.0014302457530784435, 0.006269601721311614,
          0.005285708929113789,   0.0030672005823594553, 0.008155119403871603,
          0.004086439220613566,   0.00515655831762925,   0.002804423566415652,
          0.007956607642044452,   0.0007582310515784511, 0.005585550147583929,
          0.0014302457530784435,  0.006269601721311614,  0.005285708929113789,
          0.0030672005823594553,  0.008155119403871603,  0.004086439220613566,
          0.00515655831762925,    0.002804423566415652,  0.007956607642044452,
          0.00038462648573812436, 0.005226107348461218,  0.0006764119746262043,
          0.0032195683537669033,  0.0006360972185858155, 0.006772623286003788,
          0.001380765374795222,   0.002256343422743665,  0.0001347461159293506,
          0.004693165338141832,   0.00391145922915324,   0.005315198181598307,
          0.0015256194116725953,  0.001912683335865006,  0.0008708476156508746,
          0.002834576740832729,   0.003290998881926103,  0.004589462082463799,
          0.006259183249417213,   0.0018206702056403684, 0.0003443363125208805,
          0.00038462648573812436, 0.005226107348461218,  0.0006764119746262043,
          0.0032195683537669033,  0.0006360972185858155, 0.006772623286003788,
          0.001380765374795222,   0.002256343422743665,  0.0001347461159293506,
          0.004693165338141832,   0.00391145922915324,   0.005315198181598307,
          0.0015256194116725953,  0.001912683335865006,  0.0008708476156508746,
          0.002834576740832729,   0.003290998881926103,  0.004589462082463799,
          0.006259183249417213,   0.0018206702056403684, 0.0003443363125208805,
          0.00038462648573812436, 0.005226107348461218,  0.0006764119746262043,
          0.0032195683537669033,  0.0006360972185858155, 0.006772623286003788,
          0.001380765374795222,   0.002256343422743665,  0.0001347461159293506,
          0.004693165338141832,   0.00391145922915324,   0.005315198181598307,
          0.0015256194116725953,  0.001912683335865006,  0.0008708476156508746,
          0.002834576740832729,   0.003290998881926103,  0.004589462082463799,
          0.006259183249417213,   0.0018206702056403684, 0.0003443363125208805,
          0.00038462648573812436, 0.005226107348461218,  0.0006764119746262043,
          0.0032195683537669033,  0.0006360972185858155, 0.006772623286003788,
          0.001380765374795222,   0.002256343422743665,  0.0001347461159293506,
          0.004693165338141832,   0.00391145922915324,   0.005315198181598307,
          0.0015256194116725953,  0.001912683335865006,  0.0008708476156508746,
          0.002834576740832729,   0.003290998881926103,  0.004589462082463799,
          0.006259183249417213,   0.0018206702056403684, 0.0003443363125208805,
          0.00038462648573812436, 0.005226107348461218,  0.0006764119746262043,
          0.0032195683537669033,  0.0006360972185858155, 0.006772623286003788,
          0.001380765374795222,   0.002256343422743665,  0.0001347461159293506,
          0.004693165338141832,   0.00391145922915324,   0.005315198181598307,
          0.0015256194116725953,  0.001912683335865006,  0.0008708476156508746,
          0.002834576740832729,   0.003290998881926103,  0.004589462082463799,
          0.006259183249417213,   0.0018206702056403684, 0.0003443363125208805,
          0.00038462648573812436, 0.005226107348461218,  0.0006764119746262043,
          0.0032195683537669033,  0.0006360972185858155, 0.006772623286003788,
          0.001380765374795222,   0.002256343422743665,  0.0001347461159293506,
          0.004693165338141832,   0.00391145922915324,   0.005315198181598307,
          0.0015256194116725953,  0.001912683335865006,  0.0008708476156508746,
          0.002834576740832729,   0.003290998881926103,  0.004589462082463799,
          0.006259183249417213,   0.0018206702056403684, 0.0003443363125208805};
      return {std::move(x), std::move(w)};
    }
    else if (m == 30)
    {
      // Xiao Gimbutas, 3 points, degree 30
      std::vector<T> x
          = {0.003318724936644646, 0.003318724936644646, 0.07237240722467797,
             0.07237240722467797,  0.047157910242171974, 0.047157910242171974,
             0.4680301736511254,   0.4680301736511254,   0.01268660467446775,
             0.01268660467446775,  0.12159150822272807,  0.12159150822272807,
             0.18240956151745308,  0.18240956151745308,  0.36228727935294985,
             0.36228727935294985,  0.4367432485484602,   0.4367432485484602,
             0.2724280407839283,   0.2724280407839283,   0.49731933900030856,
             0.49731933900030856,  0.003318724936644646, 0.9933625501267107,
             0.07237240722467797,  0.8552551855506441,   0.047157910242171974,
             0.905684179515656,    0.4680301736511254,   0.06393965269774915,
             0.01268660467446775,  0.9746267906510645,   0.12159150822272807,
             0.7568169835545439,   0.18240956151745308,  0.6351808769650938,
             0.36228727935294985,  0.2754254412941003,   0.4367432485484602,
             0.1265135029030796,   0.2724280407839283,   0.4551439184321434,
             0.49731933900030856,  0.005361321999382884, 0.9933625501267107,
             0.003318724936644646, 0.8552551855506441,   0.07237240722467797,
             0.905684179515656,    0.047157910242171974, 0.06393965269774915,
             0.4680301736511254,   0.9746267906510645,   0.01268660467446775,
             0.7568169835545439,   0.12159150822272807,  0.6351808769650938,
             0.18240956151745308,  0.2754254412941003,   0.36228727935294985,
             0.1265135029030796,   0.4367432485484602,   0.4551439184321434,
             0.2724280407839283,   0.005361321999382884, 0.49731933900030856,
             0.047835123140772554, 0.25905388452106737,  0.07965952693160062,
             0.39157218829125634,  0.05769340127387423,  0.35614028329623354,
             0.0772614375768841,   0.2830124973495888,   0.022758384295000066,
             0.2416137624515244,   0.1238119787706746,   0.25278124718793293,
             0.11588196723610056,  0.18490610638391713,  0.0665174447818816,
             0.19370437715364014,  0.00443878137706136,  0.07640124843938755,
             0.004663579392688625, 0.20904808745268963,  0.004703681764477044,
             0.2985429940592427,   0.025182066703868706, 0.33437904003403107,
             0.06577657382474286,  0.12323080238069513,  0.12612409498498942,
             0.3385141624298457,   0.19487095092351842,  0.35440360218068745,
             0.19100142457228309,  0.2630682977578083,   0.027533406124549888,
             0.4345661739646696,   0.028063921981372968, 0.16407098706987833,
             0.015902416268934703, 0.0426819997060804,   0.027294230652095765,
             0.09395979465272987,  0.005691211445416102, 0.13540885351993445,
             0.005162347016621321, 0.3962215147396591,   0.000533708660694491,
             0.02948404259767394,  0.6931109923381601,   0.047835123140772554,
             0.5287682847771431,   0.07965952693160062,  0.5861663154298922,
             0.05769340127387423,  0.6397260650735271,   0.0772614375768841,
             0.7356278532534755,   0.022758384295000066, 0.6234067740413924,
             0.1238119787706746,   0.6992119263799823,   0.11588196723610056,
             0.7397781780644783,   0.0665174447818816,   0.9191599701835511,
             0.00443878137706136,  0.7862883331546218,   0.004663579392688625,
             0.6967533241762803,   0.004703681764477044, 0.6404388932621002,
             0.025182066703868706, 0.8109926237945619,   0.06577657382474286,
             0.5353617425851649,   0.12612409498498942,  0.4507254468957941,
             0.19487095092351842,  0.5459302776699086,   0.19100142457228309,
             0.5379004199107804,   0.027533406124549888, 0.8078650909487487,
             0.028063921981372968, 0.9414155840249848,   0.015902416268934703,
             0.8787459746951743,   0.027294230652095765, 0.8588999350346495,
             0.005691211445416102, 0.5986161382437196,   0.005162347016621321,
             0.9699822487416316,   0.000533708660694491, 0.25905388452106737,
             0.6931109923381601,   0.39157218829125634,  0.5287682847771431,
             0.35614028329623354,  0.5861663154298922,   0.2830124973495888,
             0.6397260650735271,   0.2416137624515244,   0.7356278532534755,
             0.25278124718793293,  0.6234067740413924,   0.18490610638391713,
             0.6992119263799823,   0.19370437715364014,  0.7397781780644783,
             0.07640124843938755,  0.9191599701835511,   0.20904808745268963,
             0.7862883331546218,   0.2985429940592427,   0.6967533241762803,
             0.33437904003403107,  0.6404388932621002,   0.12323080238069513,
             0.8109926237945619,   0.3385141624298457,   0.5353617425851649,
             0.35440360218068745,  0.4507254468957941,   0.2630682977578083,
             0.5459302776699086,   0.4345661739646696,   0.5379004199107804,
             0.16407098706987833,  0.8078650909487487,   0.0426819997060804,
             0.9414155840249848,   0.09395979465272987,  0.8787459746951743,
             0.13540885351993445,  0.8588999350346495,   0.3962215147396591,
             0.5986161382437196,   0.02948404259767394,  0.9699822487416316,
             0.25905388452106737,  0.047835123140772554, 0.39157218829125634,
             0.07965952693160062,  0.35614028329623354,  0.05769340127387423,
             0.2830124973495888,   0.0772614375768841,   0.2416137624515244,
             0.022758384295000066, 0.25278124718793293,  0.1238119787706746,
             0.18490610638391713,  0.11588196723610056,  0.19370437715364014,
             0.0665174447818816,   0.07640124843938755,  0.00443878137706136,
             0.20904808745268963,  0.004663579392688625, 0.2985429940592427,
             0.004703681764477044, 0.33437904003403107,  0.025182066703868706,
             0.12323080238069513,  0.06577657382474286,  0.3385141624298457,
             0.12612409498498942,  0.35440360218068745,  0.19487095092351842,
             0.2630682977578083,   0.19100142457228309,  0.4345661739646696,
             0.027533406124549888, 0.16407098706987833,  0.028063921981372968,
             0.0426819997060804,   0.015902416268934703, 0.09395979465272987,
             0.027294230652095765, 0.13540885351993445,  0.005691211445416102,
             0.3962215147396591,   0.005162347016621321, 0.02948404259767394,
             0.000533708660694491, 0.6931109923381601,   0.25905388452106737,
             0.5287682847771431,   0.39157218829125634,  0.5861663154298922,
             0.35614028329623354,  0.6397260650735271,   0.2830124973495888,
             0.7356278532534755,   0.2416137624515244,   0.6234067740413924,
             0.25278124718793293,  0.6992119263799823,   0.18490610638391713,
             0.7397781780644783,   0.19370437715364014,  0.9191599701835511,
             0.07640124843938755,  0.7862883331546218,   0.20904808745268963,
             0.6967533241762803,   0.2985429940592427,   0.6404388932621002,
             0.33437904003403107,  0.8109926237945619,   0.12323080238069513,
             0.5353617425851649,   0.3385141624298457,   0.4507254468957941,
             0.35440360218068745,  0.5459302776699086,   0.2630682977578083,
             0.5379004199107804,   0.4345661739646696,   0.8078650909487487,
             0.16407098706987833,  0.9414155840249848,   0.0426819997060804,
             0.8787459746951743,   0.09395979465272987,  0.8588999350346495,
             0.13540885351993445,  0.5986161382437196,   0.3962215147396591,
             0.9699822487416316,   0.02948404259767394,  0.047835123140772554,
             0.6931109923381601,   0.07965952693160062,  0.5287682847771431,
             0.05769340127387423,  0.5861663154298922,   0.0772614375768841,
             0.6397260650735271,   0.022758384295000066, 0.7356278532534755,
             0.1238119787706746,   0.6234067740413924,   0.11588196723610056,
             0.6992119263799823,   0.0665174447818816,   0.7397781780644783,
             0.00443878137706136,  0.9191599701835511,   0.004663579392688625,
             0.7862883331546218,   0.004703681764477044, 0.6967533241762803,
             0.025182066703868706, 0.6404388932621002,   0.06577657382474286,
             0.8109926237945619,   0.12612409498498942,  0.5353617425851649,
             0.19487095092351842,  0.4507254468957941,   0.19100142457228309,
             0.5459302776699086,   0.027533406124549888, 0.5379004199107804,
             0.028063921981372968, 0.8078650909487487,   0.015902416268934703,
             0.9414155840249848,   0.027294230652095765, 0.8787459746951743,
             0.005691211445416102, 0.8588999350346495,   0.005162347016621321,
             0.5986161382437196,   0.000533708660694491, 0.9699822487416316};
      std::vector<T> w = {8.586495068552472e-05,  0.001900966334207759,
                          0.0014222168338437478,  0.0038920643661654237,
                          0.0004232750726644445,  0.0036934559174571445,
                          0.005376425482716406,   0.007798045662912898,
                          0.006202479514073004,   0.00745727538134288,
                          0.0013956590598703843,  8.586495068552472e-05,
                          0.001900966334207759,   0.0014222168338437478,
                          0.0038920643661654237,  0.0004232750726644445,
                          0.0036934559174571445,  0.005376425482716406,
                          0.007798045662912898,   0.006202479514073004,
                          0.00745727538134288,    0.0013956590598703843,
                          8.586495068552472e-05,  0.001900966334207759,
                          0.0014222168338437478,  0.0038920643661654237,
                          0.0004232750726644445,  0.0036934559174571445,
                          0.005376425482716406,   0.007798045662912898,
                          0.006202479514073004,   0.00745727538134288,
                          0.0013956590598703843,  0.002107379231956217,
                          0.0029624613725460075,  0.0029721100922122435,
                          0.003438592193767963,   0.0020462494841735996,
                          0.0044363272530086725,  0.003767161464773241,
                          0.003402003378050937,   0.000603348284271059,
                          0.0009794742889466218,  0.0011372229504993392,
                          0.002682342423093155,   0.0029350349015327634,
                          0.005651511771468744,   0.007158242634833795,
                          0.006463011725059148,   0.003140798290445832,
                          0.0023509684970529836,  0.0009152534380744139,
                          0.0019109402735475102,  0.0009599402787344589,
                          0.0013212839615816257,  0.00016781085573320113,
                          0.002107379231956217,   0.0029624613725460075,
                          0.0029721100922122435,  0.003438592193767963,
                          0.0020462494841735996,  0.0044363272530086725,
                          0.003767161464773241,   0.003402003378050937,
                          0.000603348284271059,   0.0009794742889466218,
                          0.0011372229504993392,  0.002682342423093155,
                          0.0029350349015327634,  0.005651511771468744,
                          0.007158242634833795,   0.006463011725059148,
                          0.003140798290445832,   0.0023509684970529836,
                          0.0009152534380744139,  0.0019109402735475102,
                          0.0009599402787344589,  0.0013212839615816257,
                          0.00016781085573320113, 0.002107379231956217,
                          0.0029624613725460075,  0.0029721100922122435,
                          0.003438592193767963,   0.0020462494841735996,
                          0.0044363272530086725,  0.003767161464773241,
                          0.003402003378050937,   0.000603348284271059,
                          0.0009794742889466218,  0.0011372229504993392,
                          0.002682342423093155,   0.0029350349015327634,
                          0.005651511771468744,   0.007158242634833795,
                          0.006463011725059148,   0.003140798290445832,
                          0.0023509684970529836,  0.0009152534380744139,
                          0.0019109402735475102,  0.0009599402787344589,
                          0.0013212839615816257,  0.00016781085573320113,
                          0.002107379231956217,   0.0029624613725460075,
                          0.0029721100922122435,  0.003438592193767963,
                          0.0020462494841735996,  0.0044363272530086725,
                          0.003767161464773241,   0.003402003378050937,
                          0.000603348284271059,   0.0009794742889466218,
                          0.0011372229504993392,  0.002682342423093155,
                          0.0029350349015327634,  0.005651511771468744,
                          0.007158242634833795,   0.006463011725059148,
                          0.003140798290445832,   0.0023509684970529836,
                          0.0009152534380744139,  0.0019109402735475102,
                          0.0009599402787344589,  0.0013212839615816257,
                          0.00016781085573320113, 0.002107379231956217,
                          0.0029624613725460075,  0.0029721100922122435,
                          0.003438592193767963,   0.0020462494841735996,
                          0.0044363272530086725,  0.003767161464773241,
                          0.003402003378050937,   0.000603348284271059,
                          0.0009794742889466218,  0.0011372229504993392,
                          0.002682342423093155,   0.0029350349015327634,
                          0.005651511771468744,   0.007158242634833795,
                          0.006463011725059148,   0.003140798290445832,
                          0.0023509684970529836,  0.0009152534380744139,
                          0.0019109402735475102,  0.0009599402787344589,
                          0.0013212839615816257,  0.00016781085573320113,
                          0.002107379231956217,   0.0029624613725460075,
                          0.0029721100922122435,  0.003438592193767963,
                          0.0020462494841735996,  0.0044363272530086725,
                          0.003767161464773241,   0.003402003378050937,
                          0.000603348284271059,   0.0009794742889466218,
                          0.0011372229504993392,  0.002682342423093155,
                          0.0029350349015327634,  0.005651511771468744,
                          0.007158242634833795,   0.006463011725059148,
                          0.003140798290445832,   0.0023509684970529836,
                          0.0009152534380744139,  0.0019109402735475102,
                          0.0009599402787344589,  0.0013212839615816257,
                          0.00016781085573320113};
      return {std::move(x), std::move(w)};
    }
    else
      throw std::runtime_error("Xiao-Gimbutas not implemented for this order.");
  }
  else if (celltype == cell::type::tetrahedron)
  {
    if (m == 1)
    {
      // Xiao Gimbutas, 4 points, degree 1
      std::vector<T> x = {0.25, 0.25, 0.25};
      std::vector<T> w = {0.16666666666666666};
      return {std::move(x), std::move(w)};
    }
    else if (m == 2)
    {
      // Xiao Gimbutas, 4 points, degree 2
      std::vector<T> x
          = {0.1236668003284584,    0.8215725409676198, 0.03993304864149842,
             0.4574615870855955,    0.155933120499186,  0.3817653560693467,
             0.3653145188146345,    0.1800296935103654, 0.006923235573627467,
             0.0003755150287292757, 0.2160764291848478, 0.4307017070778361};
      std::vector<T> w = {0.016934591412496782, 0.04646292944776137,
                          0.050086823222829334, 0.05318232258357918};
      return {std::move(x), std::move(w)};
    }
    else if (m == 3)
    {
      // Xiao Gimbutas, 4 points, degree 3
      std::vector<T> x
          = {0.6414297914956963,  0.1620014916985245,  0.1838503504920977,
             0.3454441557197307,  0.01090521221118924, 0.2815238021235462,
             0.439858947649275,   0.1901170024392839,  0.01140332944455717,
             0.03787163178235702, 0.170816925164989,   0.1528181430909273,
             0.1248048621652472,  0.1586851632274406,  0.5856628056552158,
             0.1414827519695045,  0.5712260521491151,  0.1469183900871696};
      std::vector<T> w
          = {0.020387000459557516, 0.021344402118457815, 0.022094671190740867,
             0.0234374016100672,   0.0374025278195929,   0.042000663468250383};
      return {std::move(x), std::move(w)};
    }
    else if (m == 4)
    {
      // Xiao Gimbutas, 4 points, degree 4
      std::vector<T> x
          = {0.1746940586972306,  0.04049050672759043, 0.01356070187980288,
             0.08140491840285925, 0.752508507009655,   0.06809937093820666,
             0.7412288820936226,  0.0672232948933834,  0.03518392977359872,
             0.05334123953574518, 0.419266313879513,   0.04778143555908666,
             0.4329534904813556,  0.4507658760912768,  0.05945661629943383,
             0.5380072039161857,  0.1294113737889104,  0.3301904148374645,
             0.00899126009333578, 0.1215419913339278,  0.3064939884296903,
             0.1066041725619936,  0.09720464458758327, 0.68439041545304,
             0.3292329597426469,  0.02956949520647961, 0.3179035602133946,
             0.1038441164109932,  0.4327102390477686,  0.3538232392092971,
             0.3044484024344968,  0.2402766649280726,  0.126801725915392};
      std::vector<T> w
          = {0.006541848487473326, 0.009212228192656149, 0.009232299811929395,
             0.009988864191093254, 0.011578327656272562, 0.012693785874259726,
             0.013237780011337552, 0.01774467235924835,  0.018372372071416284,
             0.02582935266937435,  0.03223513534160575};
      return {std::move(x), std::move(w)};
    }
    else if (m == 5)
    {
      // Xiao Gimbutas, 4 points, degree 5
      std::vector<T> x
          = {0.4544962958743503,  0.4544962958743504,  0.04550370412564962,
             0.04550370412564967, 0.4544962958743504,  0.4544962958743504,
             0.04550370412564973, 0.4544962958743503,  0.04550370412564969,
             0.4544962958743503,  0.04550370412564966, 0.4544962958743504,
             0.4544962958743503,  0.04550370412564968, 0.04550370412564962,
             0.0455037041256497,  0.04550370412564966, 0.4544962958743504,
             0.09273525031089128, 0.7217942490673263,  0.09273525031089122,
             0.721794249067326,   0.09273525031089128, 0.09273525031089129,
             0.09273525031089132, 0.09273525031089114, 0.09273525031089129,
             0.0927352503108913,  0.0927352503108913,  0.7217942490673263,
             0.3108859192633006,  0.06734224221009831, 0.3108859192633006,
             0.06734224221009824, 0.3108859192633006,  0.3108859192633007,
             0.3108859192633006,  0.3108859192633007,  0.3108859192633006,
             0.3108859192633006,  0.3108859192633007,  0.06734224221009814};
      std::vector<T> w
          = {0.0070910034628469025, 0.007091003462846909, 0.007091003462846909,
             0.007091003462846912,  0.007091003462846912, 0.0070910034628469155,
             0.012248840519393652,  0.012248840519393652, 0.012248840519393655,
             0.012248840519393659,  0.018781320953002632, 0.018781320953002632,
             0.018781320953002632,  0.01878132095300265};
      return {std::move(x), std::move(w)};
    }
    else if (m == 6)
    {
      // Xiao Gimbutas, 4 points, degree 6
      std::vector<T> x
          = {0.02431897424814286,  0.03883608434488445,  0.9029287990136113,
             0.02286582381402311,  0.9037700013321819,   0.02933572108317866,
             0.008781957777518898, 0.0405760510668179,   0.08860035046891021,
             0.8411389516623184,   0.05132520616520296,  0.0372647521383555,
             0.2112976585815863,   0.007354523838069352, 0.2511844952775297,
             0.02346779557305456,  0.06477516044710505,  0.3908620506710118,
             0.2130411832361856,   0.06001058302026912,  0.02584268626070331,
             0.2678441981835756,   0.06476943693005288,  0.6367675085585139,
             0.05399614083591447,  0.2757863004698506,   0.06001614916616868,
             0.3293797185491983,   0.3251196585770252,   0.3268335046190458,
             0.6243213635534294,   0.06592492316000995,  0.2535936747432003,
             0.06319998094256953,  0.6174557201472688,   0.2584491489839256,
             0.248449540118895,    0.6265402017088824,   0.06211553318359875,
             0.06373289529499766,  0.277903669330078,    0.5949096890217955,
             0.06517799276337043,  0.5947173018757956,   0.06660329800760315,
             0.08367881406005505,  0.06609866241468051,  0.6300545551109896,
             0.5773457813897267,   0.2877250948264642,   0.06462063807336853,
             0.038288670738245,    0.3283881712312217,   0.3202874336976925,
             0.3519391973347045,   0.05509902249072568,  0.3810843089063102,
             0.5300632754810166,   0.06678959978173812,  0.07699271710096725,
             0.1521038113099309,   0.1246499636374863,   0.201234567364421,
             0.3041692653497818,   0.3191942803489312,   0.04438334435720821,
             0.2558207842649862,   0.2794200529459882,   0.269569929633272};
      std::vector<T> w = {
          0.0011826324752765881, 0.001206879481977829,  0.0017372226206159916,
          0.0026542465308339587, 0.0037609445463571384, 0.0040385478129073915,
          0.00425072071117374,   0.005251568313784406,  0.006619016274847046,
          0.0072065494492455666, 0.007265066343438196,  0.007768855687763452,
          0.007858005078710203,  0.008148345983740361,  0.008294771681919054,
          0.00883888731802823,   0.008989168438051998,  0.009970224610238195,
          0.010435745880218545,  0.010511060314253423,  0.010722336995514588,
          0.011189302702092839,  0.018766567415678};
      return {std::move(x), std::move(w)};
    }
    else if (m == 7)
    {
      // Xiao Gimbutas, 4 points, degree 7
      std::vector<T> x
          = {0.001996825818299818,  0.01920799348858535,  0.6513348958482376,
             0.06092218458545083,   0.3234568417895977,   0.6151709883118704,
             0.0005004334442718418, 0.6355215105837613,   0.0598944722319085,
             0.6279832293585974,    0.293770036523707,    0.02748237819283441,
             0.05213668905801093,   0.06201109193664409,  0.05718215451677883,
             0.8245440666953954,    0.05989419506998693,  0.05677586668994691,
             0.062815072845237,     0.8207453007415948,   0.05891041282560915,
             0.6315484739180046,    0.02583316773173042,  0.280405238101906,
             0.001613532619990097,  0.1991105720528834,   0.2637473753385648,
             0.3196583760970118,    0.5991009436200256,   0.03521379529745457,
             0.5508889781422127,    0.2234702025301428,   0.2255285376972644,
             0.3505284068372833,    0.003929651487087849, 0.2418446130585829,
             0.2307849002376704,    0.01403308447330531,  0.5661630745306973,
             0.063969430325799,     0.06247402252315021,  0.812872555571,
             0.3239480709891824,    0.0624444127129091,   0.5863212858301218,
             0.2343964973359623,    0.528711306413653,    0.2169982993458658,
             0.3501153045071709,    0.2615087691765827,   0.01197587688915757,
             0.2753098106871322,    0.05300013833454678,  0.04497766312688006,
             0.0762414702839689,    0.03316569983103569,  0.2748015348979936,
             0.02253298349383202,   0.2604205879982621,   0.5646501024676913,
             0.04463026790663657,   0.6018235776318118,   0.2899940343655131,
             0.5990192398798975,    0.0528776293788545,   0.05497958289551413,
             0.0680111048992614,    0.2977852343835241,   0.04764944310089234,
             0.1572418559860032,    0.5504559416248597,   0.04374347016073189,
             0.5003786698498154,    0.2582805473674438,   0.08266909560739215,
             0.2898612819086906,    0.3971744029949173,   0.1710488778187786,
             0.1112511334269427,    0.1074624307831534,   0.4758491617393153,
             0.07400870213911578,   0.4103580959949396,   0.2447616509193416,
             0.2175544442163533,    0.2521305306293954,   0.4487392553835752,
             0.4372480897645487,    0.1098959763270211,   0.2765716827388576,
             0.2188126225475045,    0.176438223014948,    0.1757661102664513};
      std::vector<T> w = {
          0.0012846968603334146, 0.002000632031369977,  0.002085684575720105,
          0.002783666843940815,  0.0030095129140263084, 0.0032004686326964665,
          0.0034317247720467565, 0.003452013693960475,  0.0034787841936317,
          0.0035154609736464167, 0.0036967198625352964, 0.003724778580430695,
          0.0037352275658985514, 0.003869468221310365,  0.0038818311595533,
          0.00409373831432887,   0.004554985719607738,  0.0046768630523221786,
          0.00471879793532209,   0.004799380599205763,  0.005235187909249938,
          0.005632644217125163,  0.0061559681852397475, 0.006973088601266729,
          0.007165193660663451,  0.008796944275592277,  0.010050252598534952,
          0.010698139822576646,  0.011118534527621958,  0.011123248236813444,
          0.01372302813009497};
      return {std::move(x), std::move(w)};
    }
    else if (m == 8)
    {
      // Xiao Gimbutas, 4 points, degree 8
      std::vector<T> x
          = {0.0009718783690150193, 0.9573816260583027,   0.01600798423129822,
             0.01760715612687848,   0.008855594278705747, 0.9485639297990048,
             0.9408082028316022,    0.0300522477759322,   0.006719678822153188,
             0.004571184374515704,  0.04868900463902199,  0.0221620710966624,
             0.4618117251682627,    0.4653151604130365,   0.005422183307178087,
             0.03576257087872006,   0.5003296720329762,   0.008112050755706058,
             0.2307434967946494,    0.6113365336918841,   0.005575773030438238,
             0.02580684887253231,   0.02777675799356541,  0.4665405858909191,
             0.4139212834388201,    0.4982549617822043,   0.07975787096690573,
             0.1999067290329215,    0.4794066533528659,   0.3200956067503981,
             0.1643869309848081,    0.03985667792342303,  0.0344820754789425,
             0.03678064973474819,   0.7205111657178718,   0.2043689239073254,
             0.1890744688700332,    0.03620467192851837,  0.7313673600360894,
             0.1506377117814253,    0.7623678849544528,   0.05243838992870327,
             0.2496370830578753,    0.1955460819226021,   0.5417931824618095,
             0.2579166731543092,    0.2209774649988177,   0.01239592336019548,
             0.05045500693172469,   0.03938021868450564,  0.1731956086833511,
             0.7485745607457628,    0.04543453035210961,  0.1632993432993541,
             0.04648612941109276,   0.1564866850400663,   0.7514965361518376,
             0.723709495041749,     0.04620058214866413,  0.04349881445195029,
             0.04908534432749626,   0.04149415901827835,  0.71717735709395,
             0.7016567760643901,    0.2044918137472231,   0.0468686532808731,
             0.04478021549603177,   0.719573330071695,    0.05599064468210205,
             0.4487780077524824,    0.04105029814548231,  0.04568536574611817,
             0.04312238337787981,   0.4150765842699887,   0.4947991693199913,
             0.03095052921613287,   0.4246500540795894,   0.1362062412251983,
             0.2504719219173209,    0.02261090963666046,  0.2224346424927786,
             0.05438301560623344,   0.2162641488906512,   0.05125376513387382,
             0.4721332195496732,    0.04544737426285883,  0.4356669945456628,
             0.03082247209993105,   0.1812621911709453,   0.2593463891596102,
             0.5111060719135769,    0.02882725890093539,  0.2295407130726556,
             0.1239784893278184,    0.06224439946102048,  0.4036547750100588,
             0.3828805011657271,    0.397275457461443,    0.08173362820479114,
             0.4935900808340203,    0.2161396926300791,   0.03919057319753563,
             0.03383673809722395,   0.2125192261175291,   0.5056419509803703,
             0.04243030114562282,   0.4862252458661596,   0.2752360272999065,
             0.2502274462930384,    0.04529074988858361,  0.4931447679590647,
             0.2280801840550788,    0.1425146299088945,   0.1281437397666177,
             0.4835182802910101,    0.2295406150635747,   0.2420175573174804,
             0.1822460492233608,    0.4284077707360187,   0.06194696154684466,
             0.2096423028706461,    0.4853135576380692,   0.1986199776692312,
             0.1646955059003284,    0.2495207660799524,   0.249905550725755,
             0.4118962036484124,    0.1582590933458902,   0.2073917351406235,
             0.2003307068778031,    0.2274620499867547,   0.4489545286636978};
      std::vector<T> w = {
          0.0002523242325191773, 0.000290863922550942,  0.00034651409450314665,
          0.0004244834198904958, 0.0011495713860626099, 0.0012535250586434693,
          0.0016460861324811202, 0.0016881535417914368, 0.00197757011617887,
          0.0021332305621032666, 0.0022892954263809117, 0.002453317862571215,
          0.00251062824751718,   0.002564740902068485,  0.002602731533963437,
          0.0027548944667587136, 0.0028292048763357883, 0.0028982004270852453,
          0.002935422697522245,  0.002983133967882152,  0.0029981483163099916,
          0.0033577851411315802, 0.0034241727240937898, 0.0036540831010869316,
          0.0037507609946701936, 0.0037598731255837213, 0.0037906201057506216,
          0.003916487256448844,  0.00402639065463561,   0.00418984767843144,
          0.004519589747580942,  0.004542489322412185,  0.004697614316667847,
          0.0053735422239879,    0.005382166773474393,  0.0054545457922569145,
          0.006091681784605513,  0.006510451658158449,  0.006683134190221676,
          0.006927915069245575,  0.008629937323168083,  0.008810488117670844,
          0.008892300407975902,  0.009298747966287926};
      return {std::move(x), std::move(w)};
    }
    else if (m == 9)
    {
      // Xiao Gimbutas, 4 points, degree 9
      std::vector<T> x
          = {0.006891366839295159, 0.7203807203359198,  0.2440786635406753,
             0.1555789083027259,   0.7830238380676567,  0.03722731406074435,
             0.8847735952623145,   0.03746715582104113, 0.04313984945003826,
             0.04038577127605769,  0.2233476678136009,  0.723724094807067,
             0.515285061557277,    0.4405214048473231,  0.03307435692542832,
             0.03474022441842899,  0.8739842844692827,  0.04802973388725062,
             0.2737549691522871,   0.5866194063678511,  0.1331737132533926,
             0.03138783669262657,  0.02302476689782933, 0.2010151099960878,
             0.04195473200768755,  0.04206648540987851, 0.8751960265284056,
             0.2133537804622216,   0.01492382397847686, 0.72761122983064,
             0.05137368203860852,  0.737898449375068,   0.01052487006572961,
             0.01167918776735336,  0.03967712051591504, 0.47317164143189,
             0.6123408021820864,   0.1432143403978234,  0.0006662179087437572,
             0.04585901947602267,  0.04589832979249515, 0.03930065125704704,
             0.1030115138063945,   0.2313142488715318,  0.002864577988396536,
             0.00837326895113094,  0.218303550548395,   0.0638376081133778,
             0.7518893444883036,   0.03020971497018418, 0.03931328269724702,
             0.2307731120717099,   0.03171156153538288, 0.02914913102984441,
             0.005807381258551988, 0.4350661767994387,  0.4471841503422259,
             0.3807370349415273,   0.1946216802654709,  0.4205051742028665,
             0.2760774398234713,   0.6335347557774519,  0.02029485866160777,
             0.06107660031735783,  0.4443452478104636,  0.474856882867235,
             0.04011137824044635,  0.03313166695541811, 0.713691124061697,
             0.02545934399020729,  0.1809521324516702,  0.6820586129684939,
             0.6044262999606023,   0.01758681325508173, 0.2037168634602605,
             0.7098027195586388,   0.04483105915111014, 0.211317992359171,
             0.7304989942620459,   0.1871472921145495,  0.03706986589769196,
             0.5007701133228333,   0.02448223828136365, 0.04933213970351709,
             0.03332720320375657,  0.4837323966974713,  0.0372285314523825,
             0.3567521368987188,   0.1328834330395908,  0.02051457140178872,
             0.1838549978176258,   0.02094427630108402, 0.1704389178072653,
             0.2268373199643008,   0.4167820833252522,  0.007798392818596213,
             0.1923717952701477,   0.1242856487305183,  0.6561382235043771,
             0.3619508813743887,   0.01552011599218286, 0.3924336805270703,
             0.01965056042084037,  0.6443583098389682,  0.1340945354491865,
             0.458244669513536,    0.0366754648901354,  0.4635857264004116,
             0.01538959981066759,  0.3394322940952808,  0.2418695119980843,
             0.1242356229882021,   0.02743271521385679, 0.4125431242926855,
             0.5655163520825289,   0.2158455335267193,  0.1861909388269384,
             0.1031571468282741,   0.6454494634370023,  0.2025517153954042,
             0.04580873781449243,  0.1306656875755306,  0.2354771046476909,
             0.2222475737162643,   0.3744969790769868,  0.3639253527626089,
             0.4587456757955503,   0.360156713211527,   0.03689983110212465,
             0.3633566801123643,   0.4147619222990371,  0.1625263637506013,
             0.1839853909173693,   0.05281258992885371, 0.5791440724832208,
             0.03912033521689179,  0.1799045015970503,  0.4765094010040439,
             0.1524683171735235,   0.1476870681014711,  0.09407076219108002,
             0.1668749728706788,   0.5729563685802239,  0.07774426835124648,
             0.1417161598745218,   0.2271598693150327,  0.5175989795285014,
             0.3516091771883741,   0.05503739382968048, 0.1969410619376471,
             0.3364345271215077,   0.2289840090706933,  0.0827335053473291,
             0.10961245832859,     0.3537003584345384,  0.1179344061919236,
             0.5571603219749478,   0.1325043854190557,  0.1224035951881698,
             0.08342378852063631,  0.4222286016719213,  0.3031418645744979,
             0.2709662131081065,   0.3336191905680228,  0.1998230301014543,
             0.3787443941211687,   0.143482922419707,   0.3469408062547597,
             0.1819772234774806,   0.1650628409073057,  0.3092464405790883};
      std::vector<T> w = {
          0.0007162016044045418, 0.0009077771144629987, 0.0009168420416400242,
          0.0009750266138313078, 0.0011001900925731798, 0.0011235667369375677,
          0.001171941882444346,  0.0011744621157262513, 0.0011822813558184364,
          0.001184296682706247,  0.0011900066193666035, 0.0011999643701743605,
          0.001313356829704151,  0.0013368684225426673, 0.0013663833806215103,
          0.0013827059604750302, 0.0015244284474830849, 0.001527532399215677,
          0.0016299004290249847, 0.0016375601841337004, 0.0017185809772670582,
          0.00193749218890168,   0.0019604227565410766, 0.0020954845204123134,
          0.0021625430372239733, 0.0021645885229682614, 0.00218097825718004,
          0.002205183817607957,  0.0022123193411248848, 0.0022143750651612668,
          0.0022795392347588314, 0.0023777410142237997, 0.002482591391785385,
          0.00251979649860075,   0.0025639554658188667, 0.0026195628789069518,
          0.00274589846301462,   0.00331480176776181,   0.0035590499516530514,
          0.00373011850287772,   0.0038349148702950617, 0.004215437802807641,
          0.0042320852343912235, 0.004275313663630222,  0.0043800821139580864,
          0.00439593140299428,   0.00468786977428942,   0.005307550768351452,
          0.005439951471336866,  0.0055201790703796666, 0.005870060438901905,
          0.005879894123846905,  0.005952755449705822,  0.006370180985658904,
          0.006979224618881529,  0.007535230513202575,  0.008183687426958101};
      return {std::move(x), std::move(w)};
    }
    else if (m == 10)
    {
      // Xiao Gimbutas, 4 points, degree 10
      std::vector<T> x
          = {0.004844889527768171, 0.004844889527573282, 0.9854653314167374,
             0.946719848292354,    0.006667080448351054, 0.00806750556190488,
             0.006667080448658992, 0.03854556569737443,  0.008067505561779886,
             0.0385455656974801,   0.9467198482918522,   0.008067505562173938,
             0.00382689193302969,  0.8740251327651677,   0.01855431505435714,
             0.1035936602480028,   0.003826891933043923, 0.01855431505433685,
             0.8740251327649323,   0.1035936602475548,   0.01855431505434555,
             0.01735404744568466,  0.6202583012900509,   0.0324034637009163,
             0.6202583012898868,   0.3299841875634282,   0.03240346370091293,
             0.3299841875637114,   0.01735404744578351,  0.03240346370092632,
             0.03118131875220204,  0.1266394908785819,   0.8114772294984899,
             0.1266394908788713,   0.03070196087069638,  0.8114772294981942,
             0.03070196087074504,  0.03118131875221086,  0.8114772294979709,
             0.7805722458243883,   0.03518958226876812,  0.03307359982554328,
             0.03518958226870847,  0.1511645720816399,   0.03307359982557704,
             0.1511645720815177,   0.7805722458241509,   0.03307359982557292,
             0.8149022829407979,   0.03351563550542005,  0.1193767951360695,
             0.03351563550541749,  0.03220528641772838,  0.1193767951358615,
             0.03220528641771444,  0.8149022829405032,   0.1193767951363346,
             0.5596600888554596,   0.1807602495702272,   0.007438795731167321,
             0.1807602495700562,   0.2521408658433209,   0.007438795731199615,
             0.2521408658430335,   0.5596600888557026,   0.007438795731214367,
             0.03029393775770431,  0.3510309266619137,   0.03266058005326467,
             0.3510309266617719,   0.5860145555272518,   0.03266058005328976,
             0.5860145555274775,   0.03029393775772133,  0.03266058005327201,
             0.03114461249008678,  0.5992738011392268,   0.3365975263830276,
             0.03298405998767925,  0.03114461249007869,  0.3365975263825117,
             0.5992738011395161,   0.03298405998767671,  0.3365975263827067,
             0.01267643164854561,  0.1645472158899154,   0.4090752240119478,
             0.1645472158900111,   0.4137011284494345,   0.4090752240120063,
             0.4137011284495361,   0.01267643164855134,  0.4090752240118714,
             0.03573908441154557,  0.03377912559111834,  0.592532870170714,
             0.3379489198264275,   0.03573908441154142,  0.5925328701708977,
             0.03377912559113959,  0.3379489198260952,   0.5925328701712224,
             0.1336599495323923,   0.1336599495326111,   0.5990201514025275,
             0.1669597693172063,   0.1669597693175803,   0.4991206920481123,
             0.0122804285624344,   0.4126008361791699,   0.3979043621857173,
             0.1772143730727422,   0.01228042856244396,  0.3979043621855937,
             0.4126008361792021,   0.1772143730727065,   0.3979043621855659,
             0.3696126448561478,   0.1145379863160246,   0.02495402206716512,
             0.1145379863158415,   0.4908953467609971,   0.02495402206723796,
             0.4908953467605529,   0.369612644856217,    0.0249540220672398,
             0.08302494688196357,  0.7282094809974271,   0.03849678186899388,
             0.7282094809970688,   0.1502687902517552,   0.03849678186899534,
             0.1502687902517038,   0.08302494688216776,  0.03849678186896475,
             0.01680959146959896,  0.4043770144883662,   0.1613371421710751,
             0.417476251870908,    0.01680959146964861,  0.1613371421710812,
             0.4043770144883684,   0.4174762518708103,   0.1613371421711606,
             0.1696138475604023,   0.6339280343903018,   0.1733608592737587,
             0.02309725877555648,  0.1696138475603947,   0.1733608592737318,
             0.6339280343903569,   0.0230972587755699,   0.1733608592736602,
             0.02434102832026448,  0.6335875653142609,   0.1650530553465075,
             0.1770183510189467,   0.02434102832027353,  0.1650530553463749,
             0.6335875653143981,   0.1770183510189177,   0.1650530553463994,
             0.02935511686037216,  0.1740037654545786,   0.6208415104357218,
             0.1740037654546278,   0.1757996072492151,   0.6208415104357764,
             0.1757996072493023,   0.02935511686041814,  0.6208415104356415,
             0.1121243864649791,   0.5325889236500517,   0.2362419588457611,
             0.1190447310392825,   0.112124386465023,    0.2362419588456988,
             0.5325889236499357,   0.119044731039276,    0.2362419588456608,
             0.1193225934917416,   0.4534859243589937,   0.1285563923599583,
             0.2986350897893351,   0.119322593491917,    0.1285563923597633,
             0.4534859243587895,   0.298635089789364,    0.1285563923599528,
             0.3020019073266451,   0.3283455214738413,   0.3030261289420949,
             0.3283455214738281,   0.0666264422574022,   0.3030261289418549,
             0.06662644225731254,  0.3020019073267441,   0.3030261289420782,
             0.5159168166142745,   0.1324670107447715,   0.1074464615850585,
             0.1324670107446421,   0.244169711055904,    0.1074464615851541,
             0.2441697110556646,   0.5159168166144414,   0.1074464615851919,
             0.318206200820324,    0.3182062008205924,   0.04538139753862875,
             0.1207131116358087,   0.3395130626211381,   0.4110926574948671,
             0.3395130626209017,   0.1286811682481305,   0.4110926574949583,
             0.1286811682481935,   0.1207131116358556,   0.4110926574950111,
             0.2570719807624278,   0.2570719807625507,   0.2287840577124626};
      std::vector<T> w = {6.365271938318e-05,     0.00013168318579095435,
                          0.00013168318579197532, 0.00013168318579277535,
                          0.0002820943065241142,  0.0002820943065245265,
                          0.0002820943065259753,  0.0010231923733042463,
                          0.0010231923733075212,  0.0010231923733079599,
                          0.0011033909837475048,  0.0011033909837480274,
                          0.0011033909837487538,  0.00116416016749959,
                          0.0011641601675003968,  0.0011641601675006052,
                          0.001191461909106609,   0.0011914619091069783,
                          0.0011914619091077897,  0.001368674554135069,
                          0.0013686745541369087,  0.0013686745541372192,
                          0.001522996656033314,   0.00152299665603406,
                          0.0015229966560349054,  0.0016958310152239383,
                          0.0016958310152242648,  0.00169583101522512,
                          0.0018736985584823533,  0.0018736985584836066,
                          0.0018736985584844332,  0.0018987030975030218,
                          0.0018987030975037966,  0.0018987030975040383,
                          0.0019028288945829631,  0.0019591843153447786,
                          0.001969493864818562,   0.001969493864819973,
                          0.001969493864825328,   0.002064660777657845,
                          0.0020646607776616814,  0.0020646607776622417,
                          0.002083913166714165,   0.002083913166715285,
                          0.0020839131667166597,  0.002094685984369893,
                          0.002094685984373115,   0.002094685984375275,
                          0.00219141969443908,    0.0021914196944394283,
                          0.002191419694440678,   0.002284450780288395,
                          0.0022844507802891865,  0.002284450780289795,
                          0.0027774049091755302,  0.00277740490917618,
                          0.0027774049091794984,  0.003445098889335028,
                          0.0034450988893415164,  0.0034450988893420715,
                          0.004234512641858648,   0.004234512641859507,
                          0.004234512641860643,   0.004471157472950426,
                          0.004471157472951478,   0.004471157472952013,
                          0.0045538831991611866,  0.004553883199161758,
                          0.004553883199163369,   0.004651927948037353,
                          0.004682093742486137,   0.004682093742490215,
                          0.004682093742491071,   0.0077630869974032015};
      return {std::move(x), std::move(w)};
    }
    else if (m == 11)
    {
      // Xiao Gimbutas, 4 points, degree 11
      std::vector<T> x
          = {0.02174356161974667,  0.02174356162123492,  0.9347693151393625,
             0.01214887718924207,  0.4733927262830146,   0.5068373557636653,
             0.473392726282396,    0.007621040764793446, 0.5068373557636205,
             0.007621040764693851, 0.01214887718929203,  0.5068373557640636,
             0.8283411311394426,   0.01186743374839475,  0.03349202862194989,
             0.1262994064902889,   0.8283411311393326,   0.03349202862186627,
             0.01186743374856977,  0.1262994064902864,   0.03349202862190801,
             0.03884286462495556,  0.0266245121640002,   0.02791260212206558,
             0.906620021088922,    0.03884286462496065,  0.02791260212209652,
             0.02662451216400466,  0.9066200210888388,   0.02791260212224616,
             0.01779838979231115,  0.02570671787901103,  0.1452959108855631,
             0.8111989814430212,   0.01779838979236296,  0.1452959108856611,
             0.02570671787902293,  0.8111989814422269,   0.1452959108864052,
             0.4336893678654693,   0.03413212384063684,  0.004124974507952623,
             0.5280535337860727,   0.4336893678650325,   0.0041249745082621,
             0.03413212384056524,  0.528053533786363,    0.004124974508272195,
             0.02723844389731109,  0.02928619896418311,  0.8221371579049448,
             0.1213381992336525,   0.02723844389723012,  0.8221371579047962,
             0.0292861989645277,   0.1213381992371994,   0.8221371579007392,
             0.1055219304212634,   0.4058583387143239,   0.4825471524757858,
             0.40585833871604,     0.006072578389145572, 0.4825471524736373,
             0.00607257838916173,  0.1055219304213615,   0.482547152473874,
             0.7375158260909277,   0.09328386025279926,  0.01187574741300857,
             0.09328386025323457,  0.1573245662432324,   0.01187574741303624,
             0.1573245662429647,   0.7375158260902062,   0.01187574741367739,
             0.7570556764755156,   0.1851613866917053,   0.02607843400608577,
             0.1851613866916737,   0.03170450282669195,  0.02607843400609957,
             0.0317045028266906,   0.7570556764755849,   0.02607843400619944,
             0.3085734208212338,   0.6353194772067751,   0.03266098603990054,
             0.635319477206719,    0.02344611593205012,  0.03266098604008431,
             0.02344611593211183,  0.3085734208211992,   0.03266098603998926,
             0.02841949702328586,  0.2801664445485925,   0.6575245517573061,
             0.2801664445453317,   0.03388950667075014,  0.6575245517602716,
             0.03388950667083485,  0.02841949702363653,  0.6575245517604836,
             0.7288732494846533,   0.1160230818544248,   0.1355923645754941,
             0.1160230818543114,   0.01951130408545599,  0.1355923645754723,
             0.01951130408539934,  0.7288732494841443,   0.135592364576106,
             0.144054962296286,    0.680860745134503,    0.1577793967317463,
             0.01730489583748775,  0.1440549622963796,   0.1577793967318802,
             0.6808607451345954,   0.01730489583762223,  0.157779396731644,
             0.1579550425986422,   0.01216008923516082,  0.6749562807643449,
             0.1549285874017353,   0.1579550425979384,   0.6749562807650454,
             0.01216008923534246,  0.1549285874013449,   0.6749562807649698,
             0.01001242225693271,  0.4226210458987284,   0.4504683351507885,
             0.1168981966933887,   0.01001242225711663,  0.4504683351504025,
             0.4226210458990461,   0.1168981966933672,   0.4504683351504458,
             0.5566694034765993,   0.3465053726509192,   0.07740138872102775,
             0.01942383515147987,  0.5566694034766593,   0.07740138872115238,
             0.3465053726511215,   0.01942383515155061,  0.07740138872070887,
             0.613355585987273,    0.03707747991970328,  0.3147249103897943,
             0.037077479919721,    0.03484202370319439,  0.3147249103898168,
             0.0348420237032176,   0.6133555859856407,   0.314724910391352,
             0.3389609703820722,   0.01688864018974739,  0.280802863730256,
             0.01688864018976847,  0.3633475256969008,   0.280802863730816,
             0.3633475256976239,   0.3389609703821428,   0.2808028637303945,
             0.2689933017087768,   0.1524656156298647,   0.02218921323001911,
             0.5563518694315872,   0.2689933017084892,   0.0221892132299926,
             0.1524656156298783,   0.5563518694313629,   0.02218921323010947,
             0.3277711127242067,   0.3277711127240469,   0.01668666182758015,
             0.4945020621274046,   0.02951774250327373,  0.2969980096590332,
             0.02951774250352841,  0.1789821857098627,   0.2969980096594547,
             0.1789821857092536,   0.4945020621265333,   0.2969980096607454,
             0.1978395470663011,   0.02960272586703677,  0.2278827225106128,
             0.02960272586707343,  0.544675004555141,    0.2278827225112254,
             0.5446750045559754,   0.1978395470660405,   0.2278827225108745,
             0.03229881331828368,  0.3371690700776922,   0.1320500662715863,
             0.4984820503323908,   0.03229881331833241,  0.1320500662716444,
             0.3371690700771687,   0.4984820503329197,   0.1320500662716443,
             0.1207512536386751,   0.6778351364529377,   0.1011781353097973,
             0.6778351364538195,   0.1002354745986857,   0.1011781353085159,
             0.1002354745985345,   0.1207512536390304,   0.1011781353086743,
             0.138106941917413,    0.3354607968315433,   0.03273663417213666,
             0.4936956270790593,   0.1381069419173125,   0.03273663417222188,
             0.3354607968314335,   0.4936956270789277,   0.03273663417220138,
             0.2500532412376334,   0.2507616099201166,   0.4585809066512544,
             0.2507616099209062,   0.04060424219100611,  0.4585809066503929,
             0.04060424219112386,  0.2500532412374494,   0.4585809066512888,
             0.09261674502310314,  0.3090936802353301,   0.5079960391158374,
             0.3090936802365318,   0.0902935356263497,   0.5079960391138479,
             0.09029353562658468,  0.09261674502301917,  0.5079960391137064,
             0.1139695348123683,   0.1139695348116085,   0.6580913955636913,
             0.1429946401664338,   0.2384743646845937,   0.1443488665029351,
             0.4741821286458542,   0.1429946401665482,   0.1443488665032992,
             0.2384743646842464,   0.4741821286455784,   0.1443488665038495,
             0.2864949113772423,   0.1115967035331357,   0.1108203324715355,
             0.1115967035330135,   0.4910880526179628,   0.1108203324717666,
             0.4910880526178663,   0.286494911377532,    0.1108203324716498,
             0.125536113167282,    0.1148821967817834,   0.2994703853758982,
             0.4601113046750214,   0.1255361131668398,   0.2994703853761758,
             0.1148821967814947,   0.4601113046733321,   0.299470385377665,
             0.1968268645044536,   0.1968268645044899,   0.4095194064864269,
             0.1275456273615645,   0.3107744236691848,   0.2609901451138755,
             0.3107744236696469,   0.3006898038543989,   0.2609901451144841,
             0.3006898038549698,   0.1275456273614465,   0.260990145113821,
             0.296008469913471,    0.2960084699134847,   0.1119745902596869};
      std::vector<T> w = {0.00018520507764804083, 0.00024392157838708066,
                          0.000243921578397049,   0.000243921578397797,
                          0.0004426544798853663,  0.00044265447988711795,
                          0.000442654479888878,   0.00046562748965273685,
                          0.00046562748965351384, 0.00046562748965519116,
                          0.000527992042630873,   0.000527992042630885,
                          0.000527992042634342,   0.0005450543594569904,
                          0.000545054359463543,   0.0005450543594640881,
                          0.0007232685232691291,  0.0007232685232705268,
                          0.0007232685232867013,  0.000852258543339523,
                          0.0008522585433631999,  0.0008522585433649738,
                          0.0008570829478892283,  0.0008570829478907689,
                          0.0008570829479115868,  0.001012842250592951,
                          0.001012842250593049,   0.0010128422505964068,
                          0.0010128681282489697,  0.0010128681282520911,
                          0.001012868128252403,   0.0010556629969665655,
                          0.001055662996989173,   0.001055662996989902,
                          0.0010621542172256188,  0.0010621542172265183,
                          0.0010621542172285483,  0.0011697598043711175,
                          0.0011697598043717676,  0.0011697598043740865,
                          0.0012002940734976602,  0.001200294073497819,
                          0.0012002940735026382,  0.001204257126245662,
                          0.0012042571262564923,  0.0012042571262568921,
                          0.001384148614025617,   0.001384148614025973,
                          0.001384148614030434,   0.0015051557681943553,
                          0.0015051557681955198,  0.0015051557682016624,
                          0.0016274985904939964,  0.0016274985904981987,
                          0.0016274985905059345,  0.0018737072129849982,
                          0.00187370721298583,    0.0018737072129913232,
                          0.00193958466403963,    0.0019496326390221017,
                          0.0019496326390307033,  0.0019496326390484552,
                          0.00207150412472298,    0.0020715041247248048,
                          0.002071504124725327,   0.0021498066057806515,
                          0.002149806605781448,   0.0021498066057846865,
                          0.00253409922583179,    0.00253409922583231,
                          0.002534099225833395,   0.0025971204712147886,
                          0.002597120471219267,   0.002597120471221267,
                          0.0028928877153485666,  0.002892887715353863,
                          0.00289288771535987,    0.0029018056586740813,
                          0.0029018056586777047,  0.002901805658680232,
                          0.0029993712524892485,  0.003281967182103737,
                          0.0032819671821101765,  0.0032819671821297963,
                          0.0033417252382084,     0.0033417252382101736,
                          0.0033417252382157863,  0.0036081871202086548,
                          0.0036081871202103036,  0.0036081871202163014,
                          0.00450476478426666,    0.004597380383431189,
                          0.0045973803834320915,  0.00459738038343333,
                          0.004960765552103523};
      return {std::move(x), std::move(w)};
    }
    else if (m == 12)
    {
      // Xiao Gimbutas, 4 points, degree 12
      std::vector<T> x
          = {0.005336077019643312, 0.00533607701965122,  0.9839917689410563,
             0.02289609338296681,  0.9546184848667337,   0.01790586600742269,
             0.004579555742875841, 0.0228960933829682,   0.0179058660074233,
             0.9546184848667292,   0.004579555742878448, 0.0179058660074248,
             0.003634038039874281, 0.8589044391845247,   0.01981617795956102,
             0.1176453448160352,   0.003634038039875694, 0.01981617795956089,
             0.858904439184522,    0.1176453448160376,   0.01981617795956242,
             0.03542041822023295,  0.5224715513554079,   0.01083320231284623,
             0.5224715513554018,   0.4312748281115071,   0.01083320231285078,
             0.43127482811151,     0.03542041822023723,  0.01083320231285082,
             0.7939888499122254,   0.003557095377268929, 0.03305477917755645,
             0.003557095377268694, 0.1693992755329503,   0.0330547791775573,
             0.1693992755329477,   0.7939888499122263,   0.0330547791775572,
             0.03490045658756769,  0.002190912398338879, 0.6722668687771673,
             0.2906417622369189,   0.03490045658756785,  0.6722668687771728,
             0.002190912398345305, 0.2906417622369231,   0.6722668687771658,
             0.2341448840740503,   0.692501417588051,    0.0005245712210146334,
             0.07282912711688336,  0.2341448840740487,   0.000524571221016469,
             0.6925014175880473,   0.07282912711688572,  0.0005245712210174548,
             0.3050609665772057,   0.01338905088310755,  0.02647035150983993,
             0.6550796310298418,   0.3050609665772103,   0.02647035150984013,
             0.01338905088311035,  0.6550796310298342,   0.02647035150984075,
             0.004307832160712481, 0.09910993965985174,  0.1523437205162723,
             0.09910993965985478,  0.7442385076631619,   0.1523437205162712,
             0.7442385076631638,   0.004307832160714106, 0.1523437205162691,
             0.05919699437842132,  0.07116099648149783,  0.01765696019083371,
             0.07116099648149686,  0.8519850489492479,   0.01765696019083371,
             0.851985048949244,    0.05919699437842153,  0.01765696019083469,
             0.06123548744419532,  0.2324179989563869,   0.6975189009523872,
             0.00882761264703291,  0.06123548744418655,  0.6975189009523861,
             0.2324179989563872,   0.008827612647035132, 0.6975189009523897,
             0.09851040146030474,  0.02461521681961515,  0.8525144716501452,
             0.02461521681961598,  0.02435991006993637,  0.8525144716501353,
             0.02435991006993612,  0.09851040146031514,  0.8525144716501323,
             0.0299575541295916,   0.4833426975320538,   0.4764936746832871,
             0.0102060736550687,   0.02995755412959167,  0.4764936746832857,
             0.4833426975320467,   0.01020607365507009,  0.4764936746832923,
             0.6941399725295285,   0.02362764131444302,  0.2643810815068189,
             0.0178513046492101,   0.6941399725295325,   0.2643810815068127,
             0.02362764131444548,  0.01785130464921,     0.2643810815068133,
             0.03438582765934707,  0.02349501300003503,  0.0990866132622732,
             0.02349501300003591,  0.8430325460783443,   0.09908661326227276,
             0.8430325460783392,   0.03438582765934772,  0.09908661326227759,
             0.3771063994176174,   0.5748727589136075,   0.0304912794861778,
             0.5748727589136118,   0.01752956218259882,  0.03049127948617824,
             0.01752956218260132,  0.3771063994176133,   0.03049127948617957,
             0.3324969599766356,   0.3324969599766495,   0.002509120070074782,
             0.1624334612403688,   0.3425319979501514,   0.4858624607394064,
             0.3425319979501512,   0.009172080070074445, 0.4858624607394076,
             0.009172080070074595, 0.1624334612403718,   0.48586246073941,
             0.1021695690942893,   0.0288780033877441,   0.2474312436355633,
             0.02887800338774448,  0.6215211838824055,   0.2474312436355594,
             0.6215211838824011,   0.1021695690942831,   0.2474312436355717,
             0.1183429008210087,   0.5459504609989174,   0.3155723028572613,
             0.02013433532281294,  0.1183429008210102,   0.31557230285726,
             0.5459504609989215,   0.02013433532281465,  0.3155723028572566,
             0.4641610608049748,   0.08568157353024372,  0.4268281867514006,
             0.0233291789133806,   0.4641610608049873,   0.4268281867513944,
             0.08568157353024217,  0.02332917891338057,  0.4268281867513945,
             0.4810426322698022,   0.2337921730578832,   0.2697819249825928,
             0.01538326968972469,  0.4810426322698028,   0.2697819249825922,
             0.2337921730578815,   0.01538326968972496,  0.2697819249825896,
             0.08312515611213879,  0.6898395103699005,   0.02408109553674171,
             0.20295423798122,     0.08312515611213996,  0.02408109553674232,
             0.6898395103698945,   0.2029542379812216,   0.02408109553674225,
             0.02742002486773186,  0.1526139134707651,   0.7016489041188455,
             0.1526139134707658,   0.1183171575426507,   0.7016489041188496,
             0.1183171575426545,   0.02742002486773273,  0.7016489041188465,
             0.0256118593510928,   0.6956093147224817,   0.1154478712992354,
             0.1633309546271897,   0.02561185935109283,  0.1154478712992363,
             0.695609314722476,    0.1633309546271908,   0.1154478712992393,
             0.2708902039289692,   0.5606658837700146,   0.1501625601277951,
             0.5606658837700202,   0.01828135217322237,  0.1501625601277912,
             0.01828135217322301,  0.2708902039289658,   0.1501625601277926,
             0.6014420504489564,   0.07656778420928108,  0.06193417797899343,
             0.2600559873627669,   0.6014420504489668,   0.06193417797899366,
             0.0765677842092839,   0.2600559873627518,   0.06193417797899652,
             0.07462605132720868,  0.128036327678861,    0.0802610178929508,
             0.7170766031009705,   0.0746260513272115,   0.08026101789294995,
             0.1280363276788657,   0.7170766031009719,   0.08026101789294988,
             0.1859004086837668,   0.01925270365058326,  0.5025986233550219,
             0.2922482643106227,   0.185900408683766,    0.5025986233550276,
             0.01925270365058409,  0.2922482643106334,   0.5025986233550225,
             0.1176843397145563,   0.1176843397145589,   0.6469469808563281,
             0.2114376345514575,   0.2363780333382985,   0.02542476511605016,
             0.5267595669941891,   0.211437634551462,    0.02542476511605067,
             0.2363780333382918,   0.5267595669941991,   0.02542476511605207,
             0.4463155182649424,   0.4028816246284508,   0.02619956075625247,
             0.402881624628445,    0.1246032963503558,   0.02619956075625247,
             0.1246032963503482,   0.4463155182649512,   0.02619956075625223,
             0.06201821930268873,  0.08004502188860088,  0.5617139637207647,
             0.2962227950879483,   0.06201821930269108,  0.5617139637207541,
             0.08004502188860683,  0.2962227950879417,   0.5617139637207641,
             0.2972901553171066,   0.3803360632262429,   0.297442217416377,
             0.3803360632262462,   0.02493156404027269,  0.2974422174163724,
             0.02493156404027382,  0.2972901553171099,   0.297442217416373,
             0.6021485466361711,   0.0799434865871736,   0.1972600853341152,
             0.120647881442539,    0.6021485466361716,   0.1972600853341229,
             0.07994348658717083,  0.1206478814425371,   0.1972600853341221,
             0.02875584715397042,  0.4891098147268006,   0.1151244443817293,
             0.3670098937375008,   0.02875584715397076,  0.1151244443817276,
             0.4891098147267995,   0.3670098937374994,   0.1151244443817297,
             0.134993927609458,    0.4106933140901517,   0.3647602491933826,
             0.08955250910700893,  0.1349939276094596,   0.3647602491933785,
             0.4106933140901513,   0.08955250910701389,  0.3647602491933695,
             0.3918753971997565,   0.114767516763099,    0.2334539319513102,
             0.2599031540858343,   0.3918753971997636,   0.2334539319513043,
             0.1147675167631061,   0.2599031540858358,   0.2334539319513023,
             0.1301270947620232,   0.3163617622563091,   0.1273831687260135,
             0.3163617622563013,   0.426127974255657,    0.1273831687260152,
             0.4261279742556524,   0.1301270947620246,   0.127383168726021,
             0.4045023685632761,   0.2089114968404676,   0.2918235155672437,
             0.2089114968404671,   0.09476261902901918,  0.2918235155672501,
             0.0947626190290201,   0.4045023685632663,   0.2918235155672511,
             0.2274485368045121,   0.1002497543721358,   0.4514599495099783,
             0.1002497543721362,   0.2208417593133712,   0.4514599495099766,
             0.2208417593133764,   0.2274485368045148,   0.4514599495099708,
             0.5075486051202548,   0.2316476952172538,   0.1266237132343918,
             0.1341799864280956,   0.5075486051202593,   0.126623713234395,
             0.231647695217251,    0.1341799864280973,   0.1266237132343931,
             0.3053295843360563,   0.305329584336061,    0.08401124699181982,
             0.2482123715562289,   0.2482123715562241,   0.2553628853313281};
      std::vector<T> w = {3.336665703934045e-05,  8.429047994003496e-05,
                          8.42904799400373e-05,   8.429047994005044e-05,
                          0.0002149093453616615,  0.000214909345361668,
                          0.00021490934536169885, 0.0003477168822065943,
                          0.0003477168822067472,  0.00034771688220676085,
                          0.00034938896941761667, 0.0003493889694176177,
                          0.00034938896941761986, 0.00036951072812999795,
                          0.00036951072813002267, 0.00036951072813010886,
                          0.00043580724293473854, 0.0004358072429347666,
                          0.00043580724293479047, 0.0004693930811221192,
                          0.00046939308112213947, 0.000469393081122183,
                          0.00047266648548842713, 0.0004726664854884277,
                          0.00047266648548845966, 0.0005211217875707378,
                          0.0005211217875707438,  0.0005211217875707549,
                          0.0005256622670893608,  0.0005256622670894092,
                          0.0005256622670894917,  0.0005271978638527369,
                          0.0005271978638527777,  0.0005271978638527842,
                          0.0005369300725478063,  0.0005369300725478469,
                          0.000536930072547865,   0.0005566045279494747,
                          0.0005566045279495304,  0.000556604527949537,
                          0.0006129569463521267,  0.0006129569463521502,
                          0.0006129569463521515,  0.0007008690706483745,
                          0.0007008690706484037,  0.0007008690706485162,
                          0.0008238259069470653,  0.0008887153590631331,
                          0.0008887153590632215,  0.0008887153590632422,
                          0.0010498461857128857,  0.0010498461857129228,
                          0.0010498461857129625,  0.0012296728163221301,
                          0.0012296728163221822,  0.0012296728163222826,
                          0.0012860526386169384,  0.0012860526386169403,
                          0.0012860526386169514,  0.0013011726613118815,
                          0.0013011726613120303,  0.0013011726613120442,
                          0.0013077514257282738,  0.0013077514257283426,
                          0.0013077514257283478,  0.001333432670675809,
                          0.0013334326706758266,  0.001333432670675831,
                          0.0013859680589182145,  0.0013859680589182338,
                          0.0013859680589183101,  0.0014030871341772979,
                          0.0014030871341773462,  0.0014030871341773807,
                          0.0014306258422900732,  0.0014306258422900986,
                          0.0014306258422901264,  0.0014481262251185862,
                          0.0014481262251186703,  0.0014481262251187074,
                          0.0015201310181193032,  0.0015201310181193215,
                          0.001520131018119335,   0.0018509975122165715,
                          0.0018566467954329133,  0.0018566467954329567,
                          0.00185664679543301,    0.0019099333823615115,
                          0.0019099333823615501,  0.001909933382361565,
                          0.001925296724080295,   0.0019252967240803384,
                          0.0019252967240803549,  0.0019506389267458684,
                          0.0019506389267458799,  0.00195063892674594,
                          0.001981330865268667,   0.001981330865268715,
                          0.0019813308652687485,  0.00202612293640226,
                          0.0020261229364022834,  0.0020261229364023216,
                          0.0021899086547603466,  0.002189908654760405,
                          0.0021899086547604815,  0.0025877797901108766,
                          0.0025877797901111468,  0.0025877797901113883,
                          0.0028180784812232604,  0.0028180784812233983,
                          0.0028180784812235748,  0.0031305630728214965,
                          0.0031305630728215637,  0.0031305630728215767,
                          0.003251426805132802,   0.003251426805132915,
                          0.0032514268051329347,  0.003658706854740727,
                          0.0036587068547407633,  0.003658706854740793,
                          0.004252858800216373,   0.0049174945629996405};
      return {std::move(x), std::move(w)};
    }
    else if (m == 13)
    {
      // Xiao Gimbutas, 4 points, degree 13
      std::vector<T> x
          = {0.01034510725658796,  0.01034510725652366,  0.9689646782302881,
             0.9412811493523093,   0.008913012362466612, 0.03023687451099998,
             0.008913012362483565, 0.01956896377423934,  0.03023687451100695,
             0.01956896377427091,  0.9412811493522358,   0.03023687451100657,
             0.008603145262962948, 0.8844920682248393,   0.02544900390361993,
             0.08145578260860413,  0.008603145262975986, 0.02544900390362636,
             0.8844920682247885,   0.08145578260858234,  0.02544900390363992,
             0.07915594917543556,  0.87881874683591,     0.008412158658388869,
             0.03361314533024481,  0.07915594917539193,  0.008412158658414975,
             0.8788187468359566,   0.03361314533023658,  0.008412158658423473,
             0.02072452926120901,  0.2820938675417852,   0.005896688536536578,
             0.6912849146605238,   0.02072452926121111,  0.005896688536550856,
             0.282093867541703,    0.691284914660533,    0.005896688536556295,
             0.660612504418027,    0.00423850950551799,  0.3059064193536999,
             0.004238509505525088, 0.02924256672275331,  0.305906419353717,
             0.02924256672275465,  0.660612504417956,    0.3059064193537546,
             0.8108726651448125,   0.03145181004767937,  0.1465204590611696,
             0.01115506574634481,  0.8108726651448392,   0.1465204590611652,
             0.03145181004765925,  0.01115506574634996,  0.1465204590611479,
             0.01337142894352578,  0.0611452791183259,   0.1074571297438097,
             0.06114527911829663,  0.8180261621942968,   0.1074571297438493,
             0.8180261621944124,   0.01337142894356406,  0.1074571297437935,
             0.007414460977235957, 0.6955288748491347,   0.02821381240546334,
             0.6955288748491215,   0.268842851768176,    0.02821381240546195,
             0.2688428517681751,   0.007414460977257575, 0.02821381240545042,
             0.02477429690597821,  0.007482222177392445, 0.5213287809539923,
             0.007482222177401224, 0.4464146999625711,   0.5213287809540651,
             0.4464146999626367,   0.02477429690598193,  0.5213287809539731,
             0.02112965969938043,  0.09035183340349828,  0.86581765678534,
             0.09035183340359541,  0.02270085011179018,  0.8658176567852218,
             0.02270085011179335,  0.0211296596993924,   0.8658176567851917,
             0.3417930742320212,   0.002594165265187326, 0.5797685345851115,
             0.002594165265210041, 0.07584422591768267,  0.5797685345850755,
             0.07584422591769545,  0.3417930742320845,   0.579768534585002,
             0.4847746571587412,   0.4810641377028125,   0.01833057078756236,
             0.4810641377027924,   0.01583063435090152,  0.01833057078755179,
             0.0158306343508983,   0.4847746571587827,   0.01833057078756206,
             0.1676137196659997,   0.7086431089234964,   0.002914466641356441,
             0.7086431089235703,   0.1208287047691087,   0.002914466641385641,
             0.1208287047691088,   0.1676137196659543,   0.002914466641410491,
             0.05182981415363299,  0.7671835222425682,   0.01275903790005842,
             0.767183522242537,    0.1682276257037641,   0.01275903790006475,
             0.1682276257037143,   0.05182981415365493,  0.01275903790008807,
             0.1603808640957398,   0.780025722472745,    0.04144316733981209,
             0.01815024609171124,  0.1603808640958211,   0.04144316733977616,
             0.7800257224727031,   0.0181502460917103,   0.04144316733980475,
             0.001030437977338548, 0.3006497694477112,   0.5608693977782155,
             0.3006497694476872,   0.1374503947967453,   0.560869397778206,
             0.1374503947967274,   0.00103043797737723,  0.5608693977782493,
             0.009571364698298758, 0.6068954673005267,   0.3058952472331785,
             0.6068954673005383,   0.07763792076797839,  0.3058952472331658,
             0.07763792076798627,  0.009571364698318083, 0.3058952472331321,
             0.2359081252750476,   0.02712491103418074,  0.7148030256628792,
             0.02712491103418163,  0.02216393802789915,  0.7148030256628535,
             0.02216393802790523,  0.2359081252749586,   0.7148030256629532,
             0.5611854641624137,   0.3497758718670665,   0.01001327495242358,
             0.07902538901810821,  0.5611854641624555,   0.01001327495242865,
             0.3497758718669661,   0.0790253890181477,   0.01001327495243759,
             0.01201591051800159,  0.7035601418166272,   0.1282324135027442,
             0.1561915341626666,   0.01201591051801169,  0.1282324135027487,
             0.7035601418165589,   0.1561915341626685,   0.1282324135027578,
             0.01227254816726416,  0.1520983651999485,   0.1736456691583085,
             0.1520983651998947,   0.6619834174744651,   0.1736456691583576,
             0.6619834174745407,   0.01227254816727948,  0.1736456691582867,
             0.009608952348337692, 0.5009662598450281,   0.2173473462639225,
             0.5009662598449669,   0.2720774415427657,   0.21734734626392,
             0.2720774415427986,   0.009608952348355002, 0.2173473462639029,
             0.117142028395487,    0.1306252415276769,   0.7370755718293556,
             0.01515715824748938,  0.1171420283955028,   0.7370755718293308,
             0.1306252415276693,   0.01515715824751321,  0.7370755718293365,
             0.06596911170718446,  0.4869178929175357,   0.4164421277991943,
             0.03067086757606238,  0.06596911170715442,  0.4164421277993404,
             0.4869178929174335,   0.03067086757608414,  0.4164421277992895,
             0.06180743725428443,  0.7796220928971531,   0.07217289794213186,
             0.08639757190638243,  0.06180743725431112,  0.07217289794220663,
             0.7796220928970106,   0.08639757190646453,  0.07217289794218502,
             0.2124417592665611,   0.3704425559581763,   0.01068152306355014,
             0.4064341617117311,   0.2124417592666155,   0.01068152306356992,
             0.3704425559580568,   0.4064341617118274,   0.01068152306357016,
             0.01757754194097423,  0.3170690280570044,   0.0808011880664319,
             0.3170690280569446,   0.5845522419356186,   0.08080118806646398,
             0.584552241935574,    0.017577541940982,    0.0808011880664406,
             0.4819714524309398,   0.01756230923632635,  0.3414320148667813,
             0.1590342234659364,   0.4819714524309074,   0.3414320148668196,
             0.01756230923633349,  0.1590342234659289,   0.3414320148668064,
             0.08902566699068729,  0.3386042162630997,   0.02525485039503888,
             0.3386042162631206,   0.5471152663511726,   0.02525485039504986,
             0.5471152663511358,   0.08902566699071175,  0.02525485039503578,
             0.05285544098330982,  0.4000979296301265,   0.4641321089536249,
             0.4000979296298913,   0.0829145204330444,   0.4641321089536999,
             0.08291452043303993,  0.05285544098335554,  0.4641321089535966,
             0.01782606090077854,  0.2874706358928882,   0.3856172937149586,
             0.2874706358928806,   0.3090860094913717,   0.3856172937149701,
             0.3090860094913699,   0.01782606090078187,  0.3856172937149664,
             0.3710934105995327,   0.02657234084597485,  0.09690187156313018,
             0.02657234084597537,  0.5054323769913355,   0.09690187156315631,
             0.5054323769913123,   0.371093410599567,    0.09690187156314264,
             0.02148831270206694,  0.3206072988650484,   0.2127349091997354,
             0.4451694792332048,   0.02148831270207087,  0.2127349091997207,
             0.3206072988650124,   0.445169479233177,    0.2127349091997361,
             0.0951170007288927,   0.0951170007290063,   0.7146489978132146,
             0.1735802358450919,   0.02890436352483536,  0.3570824842180014,
             0.4404329164120814,   0.1735802358450641,   0.3570824842179925,
             0.02890436352488776,  0.4404329164119998,   0.357082484218023,
             0.08023850116226482,  0.6174866483249893,   0.07195784522707145,
             0.6174866483249148,   0.2303170052857254,   0.0719578452270639,
             0.2303170052856326,   0.08023850116232735,  0.07195784522713053,
             0.09452881219751588,  0.1751776752009559,   0.07065601264629488,
             0.1751776752009207,   0.6596374999553175,   0.07065601264621812,
             0.6596374999552117,   0.09452881219755312,  0.0706560126462567,
             0.06541766890798655,  0.0754347492130809,   0.2206728628957595,
             0.6384747189831496,   0.0654176689080234,   0.2206728628957641,
             0.07543474921303878,  0.6384747189831916,   0.22067286289572,
             0.5321359768315169,   0.2362423058619533,   0.03066111663210869,
             0.2362423058618978,   0.2009606006744852,   0.03066111663211837,
             0.200960600674406,    0.5321359768315502,   0.03066111663213887,
             0.2434135622407882,   0.1003198536332518,   0.5906059332310526,
             0.1003198536332653,   0.06566065089495007,  0.5906059332310177,
             0.06566065089490132,  0.2434135622409514,   0.590605933230827,
             0.03545951407500253,  0.1645691312152424,   0.5385515468421933,
             0.2614198078675792,   0.03545951407500569,  0.5385515468421994,
             0.1645691312152262,   0.2614198078675275,   0.5385515468422312,
             0.078850996951403,    0.2214791068336956,   0.1663421652944799,
             0.5333277309204181,   0.07885099695144153,  0.1663421652944797,
             0.2214791068336244,   0.533327730920471,    0.1663421652944595,
             0.0773186488433034,   0.5438135338904002,   0.1960861755130517,
             0.1827816417531721,   0.0773186488433082,   0.1960861755131724,
             0.5438135338903273,   0.1827816417532548,   0.1960861755130822,
             0.1668632870416709,   0.1668632870415595,   0.4994101388751889,
             0.1227490628669584,   0.3909296074394038,   0.08756388655247146,
             0.3987574431411315,   0.1227490628669686,   0.0875638865525005,
             0.3909296074393537,   0.3987574431411711,   0.08756388655250151,
             0.276668775818374,    0.2188997970179869,   0.4020005482407241,
             0.2188997970179731,   0.1024308789229114,   0.4020005482406983,
             0.1024308789229648,   0.2766687758182949,   0.4020005482407235,
             0.3095550183430513,   0.3095550183430962,   0.07133494497075156,
             0.3138415213429039,   0.09663461813535858,  0.2305808594445332,
             0.3589430010771437,   0.3138415213429264,   0.230580859444564,
             0.09663461813536428,  0.3589430010771935,   0.2305808594445379,
             0.4418055888803888,   0.2069223626302676,   0.1404348940839113,
             0.210837154405392,    0.4418055888803925,   0.1404348940839597,
             0.206922362630213,    0.2108371544054255,   0.1404348940839574,
             0.1108495022995848,   0.1626179922026075,   0.3263197105354871,
             0.4002127949623002,   0.1108495022995989,   0.3263197105354845,
             0.1626179922025704,   0.4002127949623043,   0.326319710535522,
             0.2505588869395824,   0.2505588869395491,   0.2483233391813068};
      std::vector<T> w = {4.134324458614672e-05,  0.00010841532703717533,
                          0.00010841532703736251, 0.00010841532703751804,
                          0.00021516249282564702, 0.00021516249282587435,
                          0.00021516249282610783, 0.00022347525797821485,
                          0.0002234752579784065,  0.00022347525797851097,
                          0.00023853469157155934, 0.00023853469157180234,
                          0.00023853469157185316, 0.00033613166424808783,
                          0.0003361316642481367,  0.00033613166424835715,
                          0.0003480651020245443,  0.0003480651020245613,
                          0.00034806510202475897, 0.00035512259072977237,
                          0.00035512259073010164, 0.0003551225907301965,
                          0.00035608395080367433, 0.0003560839508037468,
                          0.00035608395080396967, 0.0003691445658330918,
                          0.0003691445658331333,  0.000369144565833457,
                          0.000371173748022265,   0.0003711737480225677,
                          0.00037117374802262765, 0.00040213122869147067,
                          0.000402131228691888,   0.00040213122869227303,
                          0.0004154352296915045,  0.000415435229691703,
                          0.0004154352296917142,  0.000426698597924656,
                          0.0004266985979252545,  0.0004266985979257583,
                          0.0004530237838723606,  0.0004530237838724362,
                          0.0004530237838729462,  0.0005183573574952897,
                          0.0005183573574953276,  0.0005183573574954048,
                          0.0005341727061933349,  0.0005341727061939642,
                          0.0005341727061944354,  0.0006550908587339017,
                          0.0006550908587343363,  0.0006550908587344126,
                          0.0006619235298199111,  0.0006619235298200495,
                          0.0006619235298204403,  0.0006904621960691755,
                          0.0006904621960692998,  0.0006904621960698748,
                          0.0007517374404992122,  0.0007517374404998534,
                          0.0007517374404998906,  0.0007731049469574419,
                          0.0007731049469581392,  0.0007731049469581534,
                          0.0007934548044526308,  0.0007934548044530055,
                          0.0007934548044531542,  0.0008002620463692416,
                          0.000800262046369655,   0.0008002620463703901,
                          0.0008144758584702261,  0.0008144758584703776,
                          0.0008144758584716199,  0.0009212791124099626,
                          0.0009212791124102366,  0.0009212791124106396,
                          0.0009317064412002208,  0.0009317064412010393,
                          0.0009317064412011602,  0.000938800900399439,
                          0.0009388009003994993,  0.0009388009003998087,
                          0.0011277704236273554,  0.0011277704236277617,
                          0.0011277704236277934,  0.0011558578573166221,
                          0.001155857857316737,   0.0011558578573169188,
                          0.001177874388273861,   0.001177874388275692,
                          0.0011778743882767226,  0.00126388507718553,
                          0.0012638850771856667,  0.001263885077185743,
                          0.00130986012194251,    0.0013098601219425701,
                          0.0013098601219427271,  0.0013824952238897048,
                          0.0013824952238897677,  0.0013824952238899353,
                          0.0014385650798460579,  0.0015235584390117577,
                          0.0015235584390132365,  0.001523558439014583,
                          0.001553388089457361,   0.001553388089457734,
                          0.0015533880894588244,  0.0015785619534530034,
                          0.0015785619534532253,  0.0015785619534532372,
                          0.0016097788682893919,  0.001609778868289749,
                          0.0016097788682897544,  0.0016802161392982585,
                          0.00168021613929846,    0.0016802161392987117,
                          0.0017764529129038432,  0.00177645291290503,
                          0.0017764529129061567,  0.0018087386127984568,
                          0.0018087386127986667,  0.0018087386127988634,
                          0.0019104583647188483,  0.0019104583647194969,
                          0.00191045836472014,    0.0021274466568828383,
                          0.00212744665688324,    0.002127446656884127,
                          0.0023400227806175985,  0.00256011618791495,
                          0.0025601161879153333,  0.0025601161879153585,
                          0.00257113625207729,    0.0025711362520778134,
                          0.0025711362520780268,  0.002834280437245222,
                          0.002961688555453913,   0.002961688555454135,
                          0.002961688555454362,   0.0031167609951133716,
                          0.003116760995113398,   0.0031167609951139037,
                          0.003303366568208123,   0.00330336656820916,
                          0.0033033665682094884,  0.004303940769897278};
      return {std::move(x), std::move(w)};
    }
    else if (m == 14)
    {
      // Xiao Gimbutas, 4 points, degree 14
      std::vector<T> x
          = {0.006857703680959708, 0.006857703681460325, 0.9794268889578419,
             0.009252353846064727, 0.004788844763930928, 0.03329961033004269,
             0.9526591910590293,   0.009252353846425697, 0.03329961033024104,
             0.004788844764298505, 0.9526591910584257,   0.03329961033063356,
             0.06141444425249755,  0.9232999651585443,   0.002010058648387674,
             0.9232999651574737,   0.01327553194083024,  0.002010058649115951,
             0.01327553194070744,  0.06141444425278526,  0.002010058649810132,
             0.01773414386230762,  0.8989650605386635,   0.01446762164021698,
             0.8989650605385748,   0.0688331739588598,   0.01446762164046387,
             0.06883317395833545,  0.01773414386215409,  0.01446762164064887,
             0.009247363617832223, 0.3990139062271814,   0.5829037436121112,
             0.399013906225403,    0.008834986543019535, 0.5829037436136701,
             0.008834986543086735, 0.009247363618203858, 0.5829037436168685,
             0.7709900229262808,   0.01952291262766601,  0.01015620879258834,
             0.01952291262769855,  0.1993308556536677,   0.01015620879265931,
             0.1993308556534236,   0.7709900229262199,   0.01015620879270446,
             0.02032554506145879,  0.7220074978401071,   0.2431115808353706,
             0.7220074978408385,   0.01455537626308008,  0.2431115808345355,
             0.01455537626344418,  0.02032554506140792,  0.2431115808356294,
             0.01883099256369943,  0.01956534940768639,  0.8819163274688043,
             0.07968733056157902,  0.01883099256402238,  0.8819163274665061,
             0.01956534940798839,  0.07968733056254174,  0.8819163274654277,
             0.8343838905775879,   0.02228963644433849,  0.1242803391217787,
             0.02228963644438797,  0.0190461338562963,   0.1242803391208842,
             0.01904613385630611,  0.8343838905759399,   0.1242803391233692,
             0.2039231912002361,   0.02012121561803056,  0.01440053742030228,
             0.02012121561804498,  0.7615550557610796,   0.01440053742029756,
             0.7615550557609511,   0.2039231912006661,   0.01440053742036038,
             0.3361509636180879,   0.384645703604847,    0.2764557088059904,
             0.002747623971354991, 0.3361509636165141,   0.2764557088037994,
             0.3846457036073966,   0.002747623971615903, 0.276455708804184,
             0.350859996445968,    0.5522182104345288,   0.004931325226067042,
             0.5522182104342734,   0.09199046789365573,  0.004931325226427101,
             0.09199046789304373,  0.3508599964450814,   0.004931325226952402,
             0.02152082553450803,  0.0160803500126914,   0.379120808464382,
             0.5832780159922265,   0.02152082553456097,  0.3791208084604637,
             0.01608035001313742,  0.5832780159937186,   0.379120808458791,
             0.00323980932956962,  0.08400887661431675,  0.386296150270937,
             0.5264551637848466,   0.003239809329937143, 0.3862961502712207,
             0.0840088766149807,   0.5264551637842712,   0.3862961502705597,
             0.09120718428587543,  0.00411013141124883,  0.3989904928634934,
             0.004110131411517128, 0.5056921914391684,   0.3989904928650359,
             0.5056921914383363,   0.09120718428557456,  0.3989904928644713,
             0.5156439635468597,   0.2349406124884029,   0.2474912783867316,
             0.2349406124899761,   0.001924145578256931, 0.2474912783858402,
             0.001924145578230063, 0.5156439635453917,   0.2474912783874761,
             0.01932237274818349,  0.07638832520527908,  0.05638385444735509,
             0.8479054475994247,   0.01932237274817627,  0.05638385444701339,
             0.07638832520496093,  0.8479054476003125,   0.05638385444661988,
             0.3823009464917305,   0.0167988939596847,   0.02255542412116617,
             0.01679889395970684,  0.5783447354272748,   0.02255542412126642,
             0.5783447354271684,   0.3823009464918677,   0.02255542412122661,
             0.2129425371446761,   0.0212807476383709,   0.7437074210582457,
             0.02206929415866949,  0.2129425371459558,   0.7437074210569653,
             0.02128074763856252,  0.02206929415888909,  0.7437074210601419,
             0.3317239526150014,   0.3317239526139896,   0.004828142155987282,
             0.5404797680657268,   0.007294355381304199, 0.2453763152229318,
             0.007294355381551722, 0.2068495613301014,   0.2453763152216877,
             0.2068495613312503,   0.5404797680654154,   0.2453763152219601,
             0.8058111958219785,   0.09857704079908997,  0.07586181203272872,
             0.01974995134613911,  0.8058111958226273,   0.07586181203231102,
             0.09857704079958525,  0.0197499513461522,   0.07586181203295679,
             0.01748287896089844,  0.3792756315222444,   0.02304572711563008,
             0.3792756315220148,   0.5801957624014105,   0.02304572711563284,
             0.5801957624013359,   0.01748287896096796,  0.02304572711565241,
             0.2874224758303754,   0.005601417694663574, 0.6029413319659248,
             0.1040347745094613,   0.2874224758296852,   0.6029413319661477,
             0.005601417694883342, 0.104034774509157,    0.6029413319657438,
             0.1114551583386971,   0.01030235261682538,  0.7627564092743483,
             0.01030235261707004,  0.1154860797700707,   0.7627564092740204,
             0.1154860797697298,   0.1114551583385124,   0.7627564092743049,
             0.2935488505770338,   0.09761999738538321,  0.6006213514676992,
             0.09761999738502115,  0.008209800569967951, 0.6006213514669282,
             0.008209800569969635, 0.293548850577753,    0.6006213514668876,
             0.08497804426593077,  0.517513525717932,    0.01294633315160566,
             0.5175135257172315,   0.3845620968650357,   0.01294633315184834,
             0.3845620968647721,   0.08497804426566612,  0.01294633315189131,
             0.7877678986243767,   0.09409500064304101,  0.01931078849337184,
             0.09882631223971038,  0.7877678986241535,   0.01931078849337398,
             0.09409500064289474,  0.09882631223940282,  0.0193107884933751,
             0.5497595632204098,   0.01252905980804105,  0.1219406461222866,
             0.01252905980810353,  0.3157707308500686,   0.1219406461215593,
             0.3157707308512963,   0.5497595632185667,   0.1219406461220274,
             0.5303977839664743,   0.2336768622507286,   0.008568966905135528,
             0.2273563868775282,   0.5303977839660673,   0.008568966905105817,
             0.2336768622504743,   0.2273563868768414,   0.008568966905196002,
             0.01341116464697127,  0.6612982036782233,   0.09601282934079287,
             0.2292778023349974,   0.01341116464707565,  0.0960128293404185,
             0.661298203678323,    0.2292778023338679,   0.09601282934065727,
             0.6533880188663116,   0.2247326755186619,   0.01699262706566816,
             0.10488667854954,     0.6533880188669758,   0.01699262706582885,
             0.2247326755185866,   0.1048866785488491,   0.01699262706599834,
             0.2612596694959481,   0.007667507939535023, 0.4304413398190844,
             0.3006314827447682,   0.2612596694952175,   0.430441339820285,
             0.007667507939682639, 0.3006314827445465,   0.4304413398197249,
             0.01940675068598702,  0.1990033699311775,   0.07543662479224182,
             0.7061532545902,      0.01940675068596605,  0.07543662479225494,
             0.1990033699315986,   0.706153254589724,    0.07543662479267595,
             0.2153693225142795,   0.07162380480079734,  0.0661407842999457,
             0.07162380480015625,  0.6468660883848494,   0.06614078429932518,
             0.6468660883845436,   0.2153693225150791,   0.06614078429928018,
             0.077815923168826,    0.07781592316817382,  0.7665522304918784,
             0.01877419874737987,  0.1014405096223781,   0.1843049531492825,
             0.6954803384805778,   0.01877419874750536,  0.1843049531493682,
             0.1014405096223557,   0.6954803384805676,   0.1843049531495559,
             0.09578855271960066,  0.01823203771678103,  0.2102844818945803,
             0.6756949276694415,   0.09578855271908891,  0.2102844818946296,
             0.01823203771688221,  0.6756949276687967,   0.2102844818947134,
             0.4739258549152569,   0.394299156898155,    0.1145335941286789,
             0.394299156900111,    0.01724139405791831,  0.1145335941285089,
             0.01724139405793893,  0.4739258549146191,   0.1145335941289155,
             0.09983516167455454,  0.2235331462354234,   0.02642159417479953,
             0.6502100979152713,   0.09983516167429868,  0.02642159417490681,
             0.2235331462366786,   0.650210097913948,    0.026421594174992,
             0.04604965750902205,  0.3968682475417654,   0.5163227682090831,
             0.3968682475411293,   0.0407593267406208,   0.5163227682092674,
             0.04075932674082398,  0.04604965750936901,  0.5163227682088478,
             0.5706340243899409,   0.08400311872986549,  0.105036530907377,
             0.240326325971861,    0.5706340243897189,   0.1050365309082531,
             0.08400311873016891,  0.240326325974372,    0.1050365309055699,
             0.1932367143291832,   0.0713650912511827,   0.6582528796899906,
             0.07714531472941318,  0.1932367143282536,   0.6582528796907257,
             0.07136509125209903,  0.07714531472977029,  0.658252879686536,
             0.07837424052530441,  0.3761818136931925,   0.07236155120489983,
             0.4730823945784287,   0.07837424052573189,  0.07236155120465555,
             0.3761818136912941,   0.473082394578608,    0.07236155120451863,
             0.08923568468861766,  0.5115423865329087,   0.09281715399980724,
             0.3064047747789605,   0.08923568468759871,  0.09281715400051348,
             0.5115423865325862,   0.3064047747787268,   0.09281715400041667,
             0.09737203363634417,  0.1007344014121848,   0.4929088721328376,
             0.3089846928173871,   0.09737203363532915,  0.492908872135997,
             0.1007344014106231,   0.3089846928177313,   0.4929088721369547,
             0.1775478833143351,   0.3899676091564037,   0.4030241508000952,
             0.3899676091567567,   0.02946035672918423,  0.4030241508003236,
             0.02946035672919241,  0.1775478833136099,   0.403024150799614,
             0.0382221895291049,   0.2034004094915035,   0.5628552072214578,
             0.195522193757886,    0.03822218952882864,  0.5628552072215468,
             0.2034004094916,      0.1955221937576974,   0.5628552072214532,
             0.391993350183357,    0.1914181781007838,   0.03060828252772313,
             0.3859801891875482,   0.3919933501838906,   0.03060828252764542,
             0.1914181781009923,   0.3859801891869217,   0.03060828252764046,
             0.03830546710979171,  0.4143211516037891,   0.3863688156730401,
             0.4143211516027137,   0.1610045656138734,   0.3863688156733209,
             0.1610045656134992,   0.0383054671100454,   0.3863688156731704,
             0.2056632530541998,   0.05411126130896651,  0.2061145172704723,
             0.5341109683662314,   0.2056632530540627,   0.2061145172707037,
             0.05411126130928814,  0.5341109683657477,   0.2061145172697658,
             0.1015171591699777,   0.1020634349618442,   0.1162724881045504,
             0.6801469177635077,   0.1015171591702443,   0.1162724881044248,
             0.102063434961655,    0.6801469177636642,   0.1162724881042698,
             0.06794709903225467,  0.08050368241686043,  0.3036135923650056,
             0.5479356261852021,   0.0679470990326355,   0.3036135923649413,
             0.08050368241747642,  0.5479356261849525,   0.3036135923645354,
             0.2160065443462191,   0.5071777807646763,   0.2145116281103287,
             0.5071777807651786,   0.06230404677880914,  0.2145116281099157,
             0.06230404677908077,  0.2160065443456221,   0.2145116281088852,
             0.1271466762690916,   0.167752470283245,    0.3465563262838002,
             0.1677524702817992,   0.358544527163581,    0.3465563262852937,
             0.3585445271645286,   0.1271466762697583,   0.3465563262840383,
             0.3502614515179938,   0.04637577761701088,  0.2412387153575752,
             0.3621240555078011,   0.3502614515181501,   0.2412387153574093,
             0.04637577761704335,  0.3621240555066351,   0.2412387153578485,
             0.5035445472625852,   0.2096968537717866,   0.07908151868224303,
             0.2076770802830252,   0.5035445472625711,   0.07908151868236299,
             0.2096968537719228,   0.207677080282593,    0.07908151868295447,
             0.09704125425919986,  0.2851510675157584,   0.3639850895836558,
             0.2851510675153835,   0.2538225886425896,   0.363985089582452,
             0.2538225886423787,   0.09704125425935214,  0.3639850895833628,
             0.3029181193293014,   0.3029181193293765,   0.09124564201234239,
             0.4526488719953613,   0.1783434906135457,   0.2141553642703896,
             0.1783434906131041,   0.1548522731197198,   0.2141553642717896,
             0.1548522731205867,   0.4526488719948076,   0.2141553642709648,
             0.165887520255092,    0.165887520255045,    0.5023374392351683,
             0.1538060360803229,   0.3322960833854866,   0.1644327097770143,
             0.332296083385749,    0.3494651707568161,   0.1644327097772045,
             0.3494651707565804,   0.1538060360803426,   0.1644327097771141,
             0.2432210319415151,   0.2432210319416702,   0.2703369041747087};
      std::vector<T> w = {2.2105078592571967e-05, 5.104037863193469e-05,
                          5.1040378634203715e-05, 5.104037863543036e-05,
                          6.442538948289391e-05,  6.442538948628821e-05,
                          6.442538948864303e-05,  0.0001664251559695554,
                          0.00016642515597055306, 0.00016642515597261462,
                          0.00016751105250402333, 0.00016751105250589703,
                          0.00016751105250894984, 0.0002517305490836875,
                          0.00025173054908493396, 0.000251730549085241,
                          0.0002595061872553382,  0.00025950618726126,
                          0.0002595061872718605,  0.00026235085677183735,
                          0.00026235085677843987, 0.0002623508567806215,
                          0.0002896234674114693,  0.00028962346741212385,
                          0.00028962346741429767, 0.00032793268910043645,
                          0.0003279326891004587,  0.0003279326891011032,
                          0.00033555287853835234, 0.0003355528785454428,
                          0.0003355528785475206,  0.00033972253178854585,
                          0.0003397225317983497,  0.00033972253180672413,
                          0.0003441188150776556,  0.000344118815078894,
                          0.00034411881507963863, 0.0003761105389994102,
                          0.0003761105390067088,  0.00037611053901267236,
                          0.0003795403780601865,  0.00037954037806408534,
                          0.00037954037806870296, 0.0004344437592961113,
                          0.00043444375930333496, 0.00043444375930493687,
                          0.00043682384903025753, 0.000436823849034062,
                          0.00043682384903561935, 0.0004725137441075373,
                          0.00047251374410967945, 0.0004725137441096982,
                          0.0004780758505185763,  0.0004780758505195665,
                          0.0004780758505243325,  0.0004793254756053175,
                          0.000505119620061448,   0.0005051196200666875,
                          0.0005051196200669975,  0.0005119272825804478,
                          0.0005119272825807408,  0.0005119272825811382,
                          0.0005134274075967697,  0.0005134274075983005,
                          0.0005134274075996252,  0.000515837164893621,
                          0.0005158371648967786,  0.0005158371649003289,
                          0.0005239737595642078,  0.000523973759571332,
                          0.0005239737595797443,  0.0005773932407917512,
                          0.0005773932407934662,  0.0005773932407945836,
                          0.0005790788253746123,  0.0005790788253774553,
                          0.0005790788253779831,  0.0006040034240449698,
                          0.000604003424045841,   0.0006040034240459377,
                          0.0006139296700493458,  0.00061392967005271,
                          0.00061392967005407,    0.0006336953465478912,
                          0.000633695346548282,   0.0006336953465503019,
                          0.0006348790199076738,  0.0006348790199103522,
                          0.0006348790199122382,  0.0006452115628213609,
                          0.0006452115628249428,  0.0006452115628287022,
                          0.0006583108630969198,  0.0006583108630977938,
                          0.0006583108631030258,  0.0006802125901020557,
                          0.0006802125901023802,  0.0006802125901092864,
                          0.0007200311129131547,  0.0007200311129157382,
                          0.0007200311129206584,  0.0007932872325677078,
                          0.0008076554699327818,  0.0008076554699409858,
                          0.0008076554699433273,  0.0008218351357600976,
                          0.0008218351357606448,  0.0008218351357671569,
                          0.0009040682164432678,  0.0009040682164456141,
                          0.0009040682164485006,  0.0009101648818636473,
                          0.0009101648818741396,  0.0009101648818818196,
                          0.0009545436306200294,  0.0009545436306268908,
                          0.0009545436306312901,  0.001116608586184538,
                          0.0011166085861889575,  0.0011166085861902028,
                          0.001224916907034844,   0.0012249169070348833,
                          0.0012249169070411472,  0.0013681663198938994,
                          0.0013681663199072208,  0.0013681663199158513,
                          0.0014250918814340463,  0.0014250918814431518,
                          0.0014250918814463257,  0.001517977935285033,
                          0.0015179779352933332,  0.0015179779352944818,
                          0.001521769855485114,   0.001521769855489261,
                          0.0015217698554902874,  0.0015499925409594095,
                          0.0015499925409598898,  0.0015499925409610817,
                          0.0015744566170081877,  0.0015744566170082241,
                          0.0015744566170109685,  0.0016030264549198375,
                          0.0016030264549219807,  0.0016030264549262445,
                          0.0016739002367449065,  0.00167390023675795,
                          0.0016739002367584183,  0.0016764283713183198,
                          0.0016764283713183751,  0.0016764283713199717,
                          0.001679265691424395,   0.0016792656914285016,
                          0.0016792656914316033,  0.001744383239019985,
                          0.001744383239027185,   0.0017443832390316317,
                          0.0018482705191982616,  0.0018482705192122333,
                          0.0018482705192153315,  0.00191279023267428,
                          0.0019127902326753633,  0.0019127902326829017,
                          0.00220492904019695,    0.0022049290401975953,
                          0.002204929040208505,   0.00245029757423239,
                          0.0024502975742393385,  0.0024502975742462865,
                          0.002481588450245397,   0.0025078657023787183,
                          0.0025078657023794634,  0.0025078657023869535,
                          0.002703211730841763,   0.002983445658017808,
                          0.00298344565801923,    0.0029834456580193486,
                          0.0031781597181588447};
      return {std::move(x), std::move(w)};
    }
    else if (m == 15)
    {
      // Xiao Gimbutas, 4 points, degree 15
      std::vector<T> x
          = {0.02484175292278384,   0.008848046489432548,  0.9372660613009632,
             0.008848046489437518,  0.02904413928681676,   0.9372660613009608,
             0.02904413928681213,   0.02484175292278018,   0.9372660613009554,
             0.003073684779436487,  0.2792517608130759,    0.007826636168369566,
             0.709847918239126,     0.003073684779447072,  0.007826636168374292,
             0.2792517608130549,    0.7098479182391227,    0.007826636168374732,
             0.8851662080559259,    0.01941685525440453,   0.09225347133747239,
             0.01941685525440571,   0.003163465352201491,  0.09225347133747905,
             0.003163465352203353,  0.8851662080559214,    0.09225347133747011,
             0.02062270623222065,   0.9433792085073393,    0.01608597865131358,
             0.943379208507337,     0.01991210660912664,   0.01608597865131474,
             0.01991210660912658,   0.02062270623222047,   0.01608597865131562,
             0.1095755475481473,    0.01464021932495235,   0.8630127527658966,
             0.01277148036100595,   0.1095755475481476,    0.8630127527658954,
             0.01464021932495353,   0.012771480361006,     0.8630127527659015,
             0.01034990660067171,   0.7426915834872547,    0.01167275911784623,
             0.2352857507942327,    0.01034990660067426,   0.0116727591178434,
             0.7426915834872502,    0.2352857507942241,    0.01167275911784928,
             0.0133390399676393,    0.3718583817617002,    0.6057046365583734,
             0.3718583817617083,    0.009097941712290891,  0.6057046365583626,
             0.009097941712294314,  0.01333903996763977,   0.6057046365583639,
             0.5249771861414798,    0.01367226401253988,   0.01029208189692955,
             0.01367226401254362,   0.4510584679490587,    0.01029208189693136,
             0.4510584679490486,    0.5249771861414734,    0.01029208189693373,
             0.0745911424654928,    0.09342069694332726,   0.004110311725985602,
             0.8278778488651856,    0.07459114246549663,   0.004110311725986349,
             0.0934206969433283,    0.8278778488651842,    0.004110311725989205,
             0.0122514981411758,    0.11081079369917,      0.01880574820318091,
             0.1108107936991686,    0.8581319599564727,    0.01880574820318023,
             0.85813195995647,      0.01225149814117781,   0.01880574820318224,
             0.01511843542458239,   0.04765944606068975,   0.0750995589695265,
             0.04765944606069021,   0.8621225595452037,    0.07509955896952418,
             0.8621225595452005,    0.01511843542458308,   0.0750995589695319,
             0.1361581213833478,    0.001922023464433764,  0.2260822299666647,
             0.6358376251855543,    0.13615812138335,      0.2260822299666622,
             0.001922023464443689,  0.6358376251855318,    0.2260822299666766,
             0.01376037881840447,   0.7415160297079681,    0.2282126571617323,
             0.741516029707964,     0.01651093431189725,   0.2282126571617314,
             0.01651093431189804,   0.01376037881840719,   0.2282126571617405,
             0.5287234935734707,    0.3829175522239497,    0.08835752618992757,
             1.428012658284353e-06, 0.5287234935734723,    0.08835752618992139,
             0.382917552223954,     1.428012662889333e-06, 0.0883575261899238,
             0.8603078190745503,    0.1006669391238628,    0.02182336879483741,
             0.1006669391238637,    0.01720187300674862,   0.02182336879483792,
             0.01720187300674926,   0.8603078190745502,    0.0218233687948384,
             0.213749749548486,     0.003009152671000964,  0.6883673506451037,
             0.003009152671001727,  0.09487374713541114,   0.6883673506451042,
             0.09487374713541051,   0.213749749548485,     0.6883673506450955,
             0.01776885891901024,   0.5596471440071771,    0.4090155222479943,
             0.5596471440071775,    0.01356847482581952,   0.4090155222479928,
             0.01356847482581988,   0.01776885891900956,   0.4090155222479974,
             0.5561861612025368,    0.07185958939197906,   0.3631251266370949,
             0.07185958939199275,   0.008829122768390513,  0.3631251266370967,
             0.008829122768396735,  0.556186161202529,     0.3631251266370916,
             0.02153556644467808,   0.01948609324379375,   0.7368319066698201,
             0.01948609324379522,   0.2221464336417116,    0.7368319066698122,
             0.2221464336417159,    0.02153556644468099,   0.736831906669808,
             0.7706018404420292,    0.01488093241522564,   0.1457507819888409,
             0.01488093241522547,   0.06876644515390219,   0.1457507819888461,
             0.06876644515390468,   0.7706018404420206,    0.1457507819888496,
             0.2632140692833737,    0.5416447947584936,    0.1940096424521396,
             0.5416447947584961,    0.001131493505992412,  0.1940096424521405,
             0.001131493505999361,  0.2632140692833789,    0.1940096424521402,
             0.5866866785337993,    0.3700027679954775,    0.01920458791743641,
             0.02410596555328862,   0.5866866785337975,    0.01920458791743763,
             0.3700027679954874,    0.02410596555328817,   0.01920458791743828,
             0.1007245253786872,    0.007839833208067415,  0.681257697378422,
             0.007839833208073724,  0.2101779440348198,    0.6812576973784202,
             0.2101779440348183,    0.1007245253786925,    0.6812576973784144,
             0.2678058473170444,    0.009653727838177417,  0.3203870289323185,
             0.009653727838177781,  0.4021533959124907,    0.3203870289323026,
             0.4021533959124549,    0.2678058473170257,    0.3203870289323365,
             0.01871950766556472,   0.07877407681846821,   0.8174647095402787,
             0.07877407681846901,   0.08504170597569094,   0.8174647095402758,
             0.08504170597569295,   0.01871950766556529,   0.8174647095402776,
             0.2289622580755599,    0.7059180690514001,    0.01703231872933662,
             0.7059180690513981,    0.04808735414370217,   0.01703231872933689,
             0.04808735414368957,   0.2289622580755674,    0.01703231872933699,
             0.01343952012215375,   0.188978322150464,     0.07671766832162848,
             0.1889783221504602,    0.7208644894057589,    0.07671766832162524,
             0.7208644894057502,    0.01343952012215593,   0.07671766832162946,
             0.1856695714205305,    0.00940743474102812,   0.498715106027506,
             0.3062078878109278,    0.1856695714205285,    0.4987151060275142,
             0.009407434741032658,  0.306207887810909,     0.4987151060275297,
             0.07134313304985408,   0.7168988880207371,    0.01586351574919136,
             0.7168988880207331,    0.1958944631802114,    0.01586351574919187,
             0.1958944631802093,    0.07134313304985869,   0.01586351574919585,
             0.6081230921397938,    0.01132540328384632,   0.2960917880883974,
             0.01132540328384961,   0.08445971648796122,   0.2960917880884005,
             0.08445971648796277,   0.6081230921397877,    0.2960917880884,
             0.7831901908047053,    0.08946628745362917,   0.1093980943900156,
             0.01794542735164876,   0.7831901908046981,    0.1093980943900213,
             0.08946628745363563,   0.01794542735165081,   0.1093980943900156,
             0.5244117804550043,    0.2084577994081171,    0.01133447823052404,
             0.2084577994081089,    0.255795941906351,     0.01133447823052516,
             0.2557959419063432,    0.5244117804550297,    0.01133447823052761,
             0.01588261437502311,   0.3863456880386881,    0.5325804036187812,
             0.06519129396751876,   0.0158826143750247,    0.5325804036187707,
             0.3863456880386817,    0.0651912939675148,    0.5325804036187769,
             0.03236911071658531,   0.6667349326998179,    0.2320665678040404,
             0.06882938877957372,   0.03236911071658261,   0.2320665678040363,
             0.6667349326998023,    0.06882938877957215,   0.2320665678040363,
             0.2154208621405496,    0.01783785666517354,   0.07794412055941285,
             0.01783785666517584,   0.6887971606348672,    0.07794412055941696,
             0.6887971606348626,    0.2154208621405385,    0.07794412055942193,
             0.4031765079784496,    0.01499093171087654,   0.5021066311640049,
             0.01499093171087712,   0.07972592914667045,   0.5021066311640021,
             0.07972592914666891,   0.4031765079784464,    0.5021066311640048,
             0.1848636811678036,    0.6499516355495718,    0.0182999648331871,
             0.6499516355495523,    0.1468847184494371,    0.01829996483318704,
             0.1468847184494266,    0.184863681167814,     0.01829996483318721,
             0.1804453441648688,    0.451820102412822,     0.3540241023636928,
             0.01371045105861957,   0.1804453441648663,    0.3540241023636804,
             0.4518201024128161,    0.01371045105861805,   0.3540241023636947,
             0.08105206237388583,   0.3766532156247404,    0.01518945086644218,
             0.5271052711349223,    0.08105206237388458,   0.01518945086644177,
             0.3766532156247387,    0.5271052711349294,    0.01518945086644269,
             0.331408497762104,     0.1126435977537889,    0.0134868064234796,
             0.5424610980606107,    0.3314084977621181,    0.01348680642347998,
             0.1126435977537936,    0.5424610980606109,    0.01348680642348059,
             0.5607935139776807,    0.01909773271486895,   0.06348929065765115,
             0.01909773271486825,   0.3566194626497967,    0.06348929065765373,
             0.3566194626497901,    0.5607935139776836,    0.0634892906576574,
             0.07227368768344591,   0.2738988990502447,    0.6053416400059596,
             0.04848577326034448,   0.07227368768344192,   0.6053416400059548,
             0.2738988990502546,    0.04848577326034462,   0.6053416400059433,
             0.3706144545187708,    0.01602307963396246,   0.1897016601820003,
             0.01602307963396413,   0.4236608056652554,    0.1897016601819962,
             0.4236608056652498,    0.3706144545187751,    0.1897016601820099,
             0.07313934036770678,   0.06778253195192757,   0.7047038738820789,
             0.1543742537982787,    0.0731393403677088,    0.7047038738820692,
             0.06778253195194402,   0.1543742537982604,    0.7047038738820758,
             0.2255550687888375,    0.3828261323216449,    0.01264429188566553,
             0.3828261323216364,    0.3789745070038535,    0.01264429188566583,
             0.3789745070038492,    0.2255550687888252,    0.01264429188566767,
             0.01610664762850615,   0.1955976722408662,    0.504888949609758,
             0.1955976722408699,    0.2834067305208563,    0.5048889496097685,
             0.2834067305208526,    0.01610664762850647,   0.5048889496097625,
             0.07394396840975409,   0.08972209215481677,   0.06088720165051376,
             0.7754467377849116,    0.07394396840975978,   0.06088720165051331,
             0.08972209215481254,   0.7754467377849096,    0.06088720165051947,
             0.02198550415554825,   0.4522698873067332,    0.3656528451019294,
             0.4522698873067705,    0.1600917634357737,    0.3656528451019036,
             0.1600917634358059,    0.02198550415555332,   0.3656528451018906,
             0.6425340162574237,    0.03023340240085394,   0.1777342490770398,
             0.1494983322647011,    0.6425340162574126,    0.1777342490770339,
             0.03023340240085156,   0.1494983322646933,    0.1777342490770401,
             0.2982386203137134,    0.3492984375384283,    0.3282766894848435,
             0.02418625266301134,   0.2982386203137263,    0.3282766894848561,
             0.3492984375384166,    0.02418625266301722,   0.3282766894848335,
             0.02492578585848872,   0.5613484323473249,    0.1898351044830521,
             0.2238906773111161,    0.02492578585848644,   0.1898351044830608,
             0.5613484323473145,    0.22389067731113,      0.1898351044830645,
             0.1769461529053036,    0.05090872345953187,   0.6068946490861404,
             0.1652504745490214,    0.1769461529053149,    0.6068946490861201,
             0.05090872345953583,   0.1652504745490102,    0.6068946490861318,
             0.267326785868413,     0.09253312310838845,   0.162728234374953,
             0.09253312310841946,   0.4774118566482484,    0.1627282343749322,
             0.4774118566482272,    0.2673267858684251,    0.162728234374927,
             0.6324086041104882,    0.194821777558372,     0.07976498347271095,
             0.09300463485842678,   0.6324086041104942,    0.0797649834727117,
             0.1948217775583615,    0.09300463485842231,   0.07976498347272819,
             0.1286081121536503,    0.06206882668877994,   0.5069361346327252,
             0.3023869265248393,    0.1286081121536543,    0.5069361346327149,
             0.06206882668879442,   0.3023869265248427,    0.5069361346327274,
             0.0670060501118351,    0.07446918750056565,   0.362874044201665,
             0.4956507181859352,    0.06700605011182939,   0.3628740442016639,
             0.07446918750057789,   0.4956507181859199,    0.3628740442016637,
             0.04935190074328522,   0.3139246227522204,    0.1497252367121568,
             0.3139246227522261,    0.4869982397923315,    0.1497252367121615,
             0.4869982397923459,    0.04935190074327669,   0.149725236712158,
             0.1083529025389752,    0.08656298623704123,   0.1758982241285937,
             0.629185887095386,     0.1083529025389925,    0.1758982241285794,
             0.08656298623703901,   0.629185887095383,     0.1758982241285948,
             0.4947915054658847,    0.3668061114674768,    0.0797762381765776,
             0.05862614489007139,   0.4947915054658749,    0.07977623817658046,
             0.3668061114674683,    0.05862614489007745,   0.07977623817658072,
             0.6107564898848346,    0.09155687274905353,   0.08682921283620239,
             0.2108574245299135,    0.6107564898848316,    0.0868292128362,
             0.09155687274905454,   0.2108574245299094,    0.08682921283620657,
             0.27189840912706,      0.2718984091270191,    0.1843047726189553,
             0.08364528610473303,   0.4598959913496589,    0.2836943220226789,
             0.4598959913496499,    0.1727644005229201,    0.2836943220226703,
             0.1727644005229245,    0.0836452861047611,    0.2836943220226696,
             0.220307180915872,     0.2203071809158613,    0.3390784572525623,
             0.4498492745599813,    0.1402964159526019,    0.06810558755187382,
             0.3417487219355564,    0.4498492745599653,    0.0681055875518764,
             0.1402964159526053,    0.3417487219355464,    0.06810558755187403,
             0.4796245561760771,    0.06988987460087778,   0.2572931400794461,
             0.1931924291436032,    0.4796245561760668,    0.2572931400794509,
             0.06988987460089365,   0.1931924291436036,    0.2572931400794577,
             0.4819852013641325,    0.2605590256904803,    0.06519355736441289,
             0.260559025690468,     0.1922622155809665,    0.06519355736441458,
             0.1922622155809799,    0.481985201364128,     0.06519355736441629,
             0.09109001795840514,   0.343030573977829,     0.2354338687541656,
             0.3304455393096081,    0.09109001795841783,   0.2354338687541547,
             0.343030573977809,     0.3304455393095925,    0.2354338687541694,
             0.08624240167701186,   0.1525520261883117,    0.4323311285083857,
             0.3288744436262873,    0.0862424016770105,    0.4323311285083789,
             0.1525520261883389,    0.3288744436262735,    0.4323311285083771,
             0.3337847727363109,    0.3372700741462638,    0.1536568898623008,
             0.3372700741462386,    0.1752882632551174,    0.1536568898623003,
             0.1752882632551262,    0.333784772736308,     0.1536568898623202,
             0.3113177461109946,    0.3113177461109871,    0.06604676166701309,
             0.2898527003164425,    0.2359060680311884,    0.3891009454304513,
             0.2359060680311902,    0.08514028622192746,   0.3891009454304601,
             0.0851402862219274,    0.2898527003164147,    0.38910094543047,
             0.1915586123296988,    0.4738597830753307,    0.1624688668332693,
             0.4738597830753175,    0.1721127377617093,    0.1624688668332666,
             0.1721127377617219,    0.1915586123296906,    0.1624688668332857,
             0.343066772448071,     0.1725513660667974,    0.2910280425505992,
             0.1933538189344859,    0.3430667724481031,    0.2910280425505916,
             0.1725513660668472,    0.1933538189345398,    0.2910280425506466,
             0.1627284320161738,    0.1627284320161655,    0.5118147039514781};
      std::vector<T> w = {5.0949499461256536e-05, 5.094949946129215e-05,
                          5.0949499461363764e-05, 8.091578025333847e-05,
                          8.09157802534108e-05,   8.091578025341743e-05,
                          9.003858468151016e-05,  9.003858468155147e-05,
                          9.003858468155359e-05,  0.00011096788435521842,
                          0.00011096788435523033, 0.00011096788435523135,
                          0.00014301129642297394, 0.0001430112964229811,
                          0.00014301129642299053, 0.0001435686140277277,
                          0.00014356861402773594, 0.00014356861402778517,
                          0.00017047619231215117, 0.00017047619231217833,
                          0.00017047619231222533, 0.00017795546384389533,
                          0.00017795546384394634, 0.00017795546384398616,
                          0.00018791251008007368, 0.00018791251008008316,
                          0.00018791251008011434, 0.00018853254351190317,
                          0.00018853254351192117, 0.0001885325435119355,
                          0.000203420107255822,   0.00020342010725582285,
                          0.00020342010725585117, 0.00023831202538122532,
                          0.000238312025381286,   0.00023831202538146515,
                          0.00024702490709429967, 0.0002470249070943653,
                          0.00024702490709436765, 0.00025848719013847165,
                          0.00025848719013854435, 0.00025848719013861867,
                          0.000265583230932468,   0.00026558323093246863,
                          0.0002655832309324773,  0.00029768749691803264,
                          0.0002976874969180593,  0.000297687496918183,
                          0.0002989438206765175,  0.0002989438206765232,
                          0.0002989438206765322,  0.0003205499531253637,
                          0.0003205499531254577,  0.0003205499531255287,
                          0.0003351988396668535,  0.00033519883966689985,
                          0.00033519883966691964, 0.000357511187619173,
                          0.000357511187619211,   0.00035751118761923035,
                          0.0003766593641388355,  0.00037665936413884397,
                          0.00037665936413898253, 0.0004170560045706985,
                          0.00041705600457073996, 0.00041705600457075487,
                          0.0004206908093603285,  0.0004206908093604563,
                          0.00042069080936051436, 0.0004403404439190608,
                          0.0004403404439191172,  0.00044034044391935267,
                          0.0004510176089945963,  0.00045101760899459884,
                          0.0004510176089946125,  0.00048092536064853767,
                          0.0004809253606485387,  0.0004809253606485603,
                          0.0004921683513968802,  0.000492168351396902,
                          0.0004921683513969398,  0.0004972651138408006,
                          0.0004972651138408805,  0.0004972651138410044,
                          0.0005046209614997817,  0.0005046209614998177,
                          0.0005046209614999028,  0.0005053606662589742,
                          0.0005053606662590495,  0.0005053606662590833,
                          0.000507488410074817,   0.000507488410074818,
                          0.0005074884100748625,  0.0005296283626829228,
                          0.0005296283626829545,  0.0005296283626830274,
                          0.0005961391122951584,  0.0005961391122951768,
                          0.0005961391122952683,  0.0006063199394915311,
                          0.0006063199394916362,  0.0006063199394917882,
                          0.0006305106887630419,  0.0006305106887630918,
                          0.0006305106887631449,  0.0006490437338464095,
                          0.00064904373384643,    0.0006490437338465401,
                          0.0006784672244327731,  0.0006784672244328733,
                          0.0006784672244328974,  0.0007018738744954993,
                          0.0007018738744955992,  0.0007018738744956027,
                          0.0007044998612670034,  0.0007044998612670082,
                          0.0007044998612670204,  0.0007123210856255668,
                          0.0007123210856256313,  0.0007123210856256491,
                          0.0007180840505148808,  0.0007180840505148981,
                          0.0007180840505149077,  0.0007320095837554124,
                          0.000732009583755433,   0.0007320095837555596,
                          0.0007487529433902918,  0.0007487529433903294,
                          0.0007487529433904035,  0.0007633124179686057,
                          0.0007633124179687115,  0.0007633124179687585,
                          0.0007665117920190594,  0.0007665117920191146,
                          0.0007665117920191253,  0.000782684926528277,
                          0.0007826849265283078,  0.0007826849265283831,
                          0.0008216462765614071,  0.0008216462765614102,
                          0.0008216462765614202,  0.0008401904995806664,
                          0.0008401904995807659,  0.000840190499580874,
                          0.0008649691231126786,  0.0008649691231126842,
                          0.0008649691231126886,  0.0009771082729624124,
                          0.0009771082729625377,  0.0009771082729625865,
                          0.001003226140444279,   0.0010032261404442897,
                          0.001003226140444467,   0.0011107068782117186,
                          0.0011107068782118005,  0.001110706878211849,
                          0.0011541194419159395,  0.0011541194419163337,
                          0.0011541194419164846,  0.0011888869781639686,
                          0.0011888869781640197,  0.0011888869781641177,
                          0.0012039315081215961,  0.0012039315081217672,
                          0.0012039315081217863,  0.0012677548100906834,
                          0.0012677548100907092,  0.0012677548100908191,
                          0.0013075346711990234,  0.001307534671199086,
                          0.0013075346711991051,  0.0013386099794468343,
                          0.0013386099794469022,  0.0013386099794470629,
                          0.0013454689764230332,  0.0013454689764231366,
                          0.0013454689764231468,  0.001398358195255936,
                          0.0013983581952560142,  0.0013983581952561012,
                          0.001511722175784935,   0.001536829401278278,
                          0.0015368294012785917,  0.0015368294012786117,
                          0.001568496765944744,   0.0016371705077032532,
                          0.0016371705077033274,  0.0016371705077033551,
                          0.0016680098648530085,  0.0016680098648530152,
                          0.0016680098648534517,  0.0016801415184265068,
                          0.0016801415184265085,  0.0016801415184265866,
                          0.0016912148002698566,  0.0016912148002698867,
                          0.0016912148002702135,  0.0018093524420684734,
                          0.0018093524420685033,  0.001809352442068565,
                          0.0018270943485695949,  0.0018270943485701834,
                          0.0018270943485707415,  0.0018747167751995899,
                          0.0018892190654910483,  0.0018892190654914516,
                          0.0018892190654915483,  0.0019096117089971417,
                          0.0019096117089973984,  0.0019096117089977185,
                          0.002049458570185687,   0.00204945857018569,
                          0.002049458570187277,   0.0024074895531075533};
      return {std::move(x), std::move(w)};
    }
    else
      throw std::runtime_error("Xiao-Gimbutas not implemented for this order.");
  }
  else
  {
    throw std::runtime_error(
        "Xiao-Gimbutas is only implemented for triangles.");
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::array<std::vector<T>, 2>
make_witherden_vincent_quadrature(cell::type celltype, int m)
{
  switch (celltype)
  {
  case cell::type::interval:
    // The Witherden-Vincent family reduces to Gauss-Legendre on an
    // interval.
    return make_quadrature_line<T>((m + 2) / 2);
  case cell::type::triangle:
  {
    if (m <= 1)
    {
      // Witherden-Vincent, 1 point, degree 1
      std::vector<T> x = {0.3333333333333333, 0.3333333333333333};
      std::vector<T> w = {0.5};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 2)
    {
      // Witherden-Vincent, 3 points, degree 2
      std::vector<T> x
          = {0.16666666666666666, 0.6666666666666666,  0.6666666666666666,
             0.16666666666666666, 0.16666666666666666, 0.16666666666666666};
      std::vector<T> w
          = {0.16666666666666666, 0.16666666666666666, 0.16666666666666666};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 4)
    {
      // Witherden-Vincent, 6 points, degree 4
      std::vector<T> x
          = {0.4459484909159649,  0.10810301816807023, 0.10810301816807023,
             0.4459484909159649,  0.4459484909159649,  0.4459484909159649,
             0.09157621350977074, 0.8168475729804585,  0.8168475729804585,
             0.09157621350977074, 0.09157621350977074, 0.09157621350977074};
      std::vector<T> w
          = {0.11169079483900574,  0.11169079483900574,  0.11169079483900574,
             0.054975871827660935, 0.054975871827660935, 0.054975871827660935};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 5)
    {
      // Witherden-Vincent, 7 points, degree 5
      std::vector<T> x
          = {0.3333333333333333,  0.3333333333333333,  0.10128650732345634,
             0.7974269853530873,  0.7974269853530873,  0.10128650732345634,
             0.10128650732345634, 0.10128650732345634, 0.4701420641051151,
             0.05971587178976982, 0.05971587178976982, 0.4701420641051151,
             0.4701420641051151,  0.4701420641051151};
      std::vector<T> w = {0.1125,
                          0.06296959027241357,
                          0.06296959027241357,
                          0.06296959027241357,
                          0.0661970763942531,
                          0.0661970763942531,
                          0.0661970763942531};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 6)
    {
      // Witherden-Vincent, 12 points, degree 6
      std::vector<T> x
          = {0.06308901449150223,  0.8738219710169955,   0.8738219710169955,
             0.06308901449150223,  0.06308901449150223,  0.06308901449150223,
             0.24928674517091043,  0.5014265096581791,   0.5014265096581791,
             0.24928674517091043,  0.24928674517091043,  0.24928674517091043,
             0.053145049844816945, 0.6365024991213987,   0.6365024991213987,
             0.053145049844816945, 0.3103524510337844,   0.6365024991213987,
             0.6365024991213987,   0.3103524510337844,   0.3103524510337844,
             0.053145049844816945, 0.053145049844816945, 0.3103524510337844};
      std::vector<T> w
          = {0.02542245318510341,  0.02542245318510341,  0.02542245318510341,
             0.058393137863189684, 0.058393137863189684, 0.058393137863189684,
             0.041425537809186785, 0.041425537809186785, 0.041425537809186785,
             0.041425537809186785, 0.041425537809186785, 0.041425537809186785};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 7)
    {
      // Witherden-Vincent, 15 points, degree 7
      std::vector<T> x
          = {0.03373064855458785, 0.9325387028908243,  0.9325387028908243,
             0.03373064855458785, 0.03373064855458785, 0.03373064855458785,
             0.24157738259540357, 0.5168452348091929,  0.5168452348091929,
             0.24157738259540357, 0.24157738259540357, 0.24157738259540357,
             0.4743096925047182,  0.05138061499056353, 0.05138061499056353,
             0.4743096925047182,  0.4743096925047182,  0.4743096925047182,
             0.04703664465259523, 0.7542800405500532,  0.7542800405500532,
             0.04703664465259523, 0.1986833147973516,  0.7542800405500532,
             0.7542800405500532,  0.1986833147973516,  0.1986833147973516,
             0.04703664465259523, 0.04703664465259523, 0.1986833147973516};
      std::vector<T> w
          = {0.008272525055396066, 0.008272525055396066, 0.008272525055396066,
             0.06397208561507779,  0.06397208561507779,  0.06397208561507779,
             0.038543323092993034, 0.038543323092993034, 0.038543323092993034,
             0.02793936645159989,  0.02793936645159989,  0.02793936645159989,
             0.02793936645159989,  0.02793936645159989,  0.02793936645159989};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 8)
    {
      // Witherden-Vincent, 16 points, degree 8
      std::vector<T> x
          = {0.3333333333333333,   0.3333333333333333,  0.4592925882927232,
             0.0814148234145537,   0.0814148234145537,  0.4592925882927232,
             0.4592925882927232,   0.4592925882927232,  0.1705693077517602,
             0.6588613844964796,   0.6588613844964796,  0.1705693077517602,
             0.1705693077517602,   0.1705693077517602,  0.05054722831703098,
             0.8989055433659381,   0.8989055433659381,  0.05054722831703098,
             0.05054722831703098,  0.05054722831703098, 0.008394777409957605,
             0.7284923929554042,   0.7284923929554042,  0.008394777409957605,
             0.2631128296346381,   0.7284923929554042,  0.7284923929554042,
             0.2631128296346381,   0.2631128296346381,  0.008394777409957605,
             0.008394777409957605, 0.2631128296346381};
      std::vector<T> w
          = {0.07215780383889359,  0.04754581713364231,  0.04754581713364231,
             0.04754581713364231,  0.05160868526735912,  0.05160868526735912,
             0.05160868526735912,  0.01622924881159904,  0.01622924881159904,
             0.01622924881159904,  0.013615157087217496, 0.013615157087217496,
             0.013615157087217496, 0.013615157087217496, 0.013615157087217496,
             0.013615157087217496};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 9)
    {
      // Witherden-Vincent, 19 points, degree 9
      std::vector<T> x
          = {0.3333333333333333,   0.3333333333333333,   0.43708959149293664,
             0.12582081701412673,  0.12582081701412673,  0.43708959149293664,
             0.43708959149293664,  0.43708959149293664,  0.18820353561903272,
             0.6235929287619345,   0.6235929287619345,   0.18820353561903272,
             0.18820353561903272,  0.18820353561903272,  0.4896825191987376,
             0.020634961602524746, 0.020634961602524746, 0.4896825191987376,
             0.4896825191987376,   0.4896825191987376,   0.04472951339445271,
             0.9105409732110946,   0.9105409732110946,   0.04472951339445271,
             0.04472951339445271,  0.04472951339445271,  0.036838412054736286,
             0.741198598784498,    0.741198598784498,    0.036838412054736286,
             0.2219629891607657,   0.741198598784498,    0.741198598784498,
             0.2219629891607657,   0.2219629891607657,   0.036838412054736286,
             0.036838412054736286, 0.2219629891607657};
      std::vector<T> w
          = {0.04856789814139942,  0.03891377050238714,  0.03891377050238714,
             0.03891377050238714,  0.039823869463605124, 0.039823869463605124,
             0.039823869463605124, 0.015667350113569536, 0.015667350113569536,
             0.015667350113569536, 0.012788837829349016, 0.012788837829349016,
             0.012788837829349016, 0.021641769688644688, 0.021641769688644688,
             0.021641769688644688, 0.021641769688644688, 0.021641769688644688,
             0.021641769688644688};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 10)
    {
      // Witherden-Vincent, 25 points, degree 10
      std::vector<T> x
          = {0.3333333333333333,  0.3333333333333333,  0.03205537321694351,
             0.935889253566113,   0.935889253566113,   0.03205537321694351,
             0.03205537321694351, 0.03205537321694351, 0.14216110105656438,
             0.7156777978868712,  0.7156777978868712,  0.14216110105656438,
             0.14216110105656438, 0.14216110105656438, 0.32181299528883545,
             0.530054118927344,   0.530054118927344,   0.32181299528883545,
             0.14813288578382056, 0.530054118927344,   0.530054118927344,
             0.14813288578382056, 0.14813288578382056, 0.32181299528883545,
             0.32181299528883545, 0.14813288578382056, 0.02961988948872977,
             0.6012333286834592,  0.6012333286834592,  0.02961988948872977,
             0.36914678182781097, 0.6012333286834592,  0.6012333286834592,
             0.36914678182781097, 0.36914678182781097, 0.02961988948872977,
             0.02961988948872977, 0.36914678182781097, 0.02836766533993844,
             0.807930600922879,   0.807930600922879,   0.02836766533993844,
             0.1637017337371825,  0.807930600922879,   0.807930600922879,
             0.1637017337371825,  0.1637017337371825,  0.02836766533993844,
             0.02836766533993844, 0.1637017337371825};
      std::vector<T> w
          = {0.040871664573142986, 0.006676484406574783, 0.006676484406574783,
             0.006676484406574783, 0.022978981802372365, 0.022978981802372365,
             0.022978981802372365, 0.03195245319821202,  0.03195245319821202,
             0.03195245319821202,  0.03195245319821202,  0.03195245319821202,
             0.03195245319821202,  0.017092324081479714, 0.017092324081479714,
             0.017092324081479714, 0.017092324081479714, 0.017092324081479714,
             0.017092324081479714, 0.012648878853644192, 0.012648878853644192,
             0.012648878853644192, 0.012648878853644192, 0.012648878853644192,
             0.012648878853644192};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 11)
    {
      // Witherden-Vincent, 28 points, degree 11
      std::vector<T> x
          = {0.3333333333333333,   0.3333333333333333,   0.02848541761437191,
             0.9430291647712562,   0.9430291647712562,   0.02848541761437191,
             0.02848541761437191,  0.02848541761437191,  0.21021995670317825,
             0.5795600865936434,   0.5795600865936434,   0.21021995670317825,
             0.21021995670317825,  0.21021995670317825,  0.10263548271224643,
             0.7947290345755071,   0.7947290345755071,   0.10263548271224643,
             0.10263548271224643,  0.10263548271224643,  0.4958919009658909,
             0.008216198068218173, 0.008216198068218173, 0.4958919009658909,
             0.4958919009658909,   0.4958919009658909,   0.4384659267643522,
             0.12306814647129562,  0.12306814647129562,  0.4384659267643522,
             0.4384659267643522,   0.4384659267643522,   0.14932478865208237,
             0.8433497836618532,   0.8433497836618532,   0.14932478865208237,
             0.007325427686064452, 0.8433497836618532,   0.8433497836618532,
             0.007325427686064452, 0.007325427686064452, 0.14932478865208237,
             0.14932478865208237,  0.007325427686064452, 0.04601050016542996,
             0.6644083741968642,   0.6644083741968642,   0.04601050016542996,
             0.2895811256377058,   0.6644083741968642,   0.6644083741968642,
             0.2895811256377058,   0.2895811256377058,   0.04601050016542996,
             0.04601050016542996,  0.2895811256377058};
      std::vector<T> w
          = {0.04288058986611211,  0.005215935256447348, 0.005215935256447348,
             0.005215935256447348, 0.03525784205585829,  0.03525784205585829,
             0.03525784205585829,  0.01931537961850966,  0.01931537961850966,
             0.01931537961850966,  0.008303136527292684, 0.008303136527292684,
             0.008303136527292684, 0.03365807703973415,  0.03365807703973415,
             0.03365807703973415,  0.005145144786476639, 0.005145144786476639,
             0.005145144786476639, 0.005145144786476639, 0.005145144786476639,
             0.005145144786476639, 0.020166238320250277, 0.020166238320250277,
             0.020166238320250277, 0.020166238320250277, 0.020166238320250277,
             0.020166238320250277};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 12)
    {
      // Witherden-Vincent, 33 points, degree 12
      std::vector<T> x
          = {0.4882037509455415,   0.023592498108916896, 0.023592498108916896,
             0.4882037509455415,   0.4882037509455415,   0.4882037509455415,
             0.1092578276593543,   0.7814843446812915,   0.7814843446812915,
             0.1092578276593543,   0.1092578276593543,   0.1092578276593543,
             0.2714625070149261,   0.45707498597014784,  0.45707498597014784,
             0.2714625070149261,   0.2714625070149261,   0.2714625070149261,
             0.024646363436335594, 0.9507072731273288,   0.9507072731273288,
             0.024646363436335594, 0.024646363436335594, 0.024646363436335594,
             0.4401116486585931,   0.11977670268281378,  0.11977670268281378,
             0.4401116486585931,   0.4401116486585931,   0.4401116486585931,
             0.29165567973834094,  0.6853101639063919,   0.6853101639063919,
             0.29165567973834094,  0.02303415635526714,  0.6853101639063919,
             0.6853101639063919,   0.02303415635526714,  0.02303415635526714,
             0.29165567973834094,  0.29165567973834094,  0.02303415635526714,
             0.11629601967792659,  0.628249751683556,    0.628249751683556,
             0.11629601967792659,  0.25545422863851736,  0.628249751683556,
             0.628249751683556,    0.25545422863851736,  0.25545422863851736,
             0.11629601967792659,  0.11629601967792659,  0.25545422863851736,
             0.85133779251024,     0.12727971723358936,  0.12727971723358936,
             0.85133779251024,     0.02138249025617059,  0.12727971723358936,
             0.12727971723358936,  0.02138249025617059,  0.02138249025617059,
             0.85133779251024,     0.85133779251024,     0.02138249025617059};
      std::vector<T> w = {
          0.012133419040726016,  0.012133419040726016,  0.012133419040726016,
          0.014243026034438772,  0.014243026034438772,  0.014243026034438772,
          0.03127060659795138,   0.03127060659795138,   0.03127060659795138,
          0.0039658212549868194, 0.0039658212549868194, 0.0039658212549868194,
          0.02495916746403047,   0.02495916746403047,   0.02495916746403047,
          0.01089179251930378,   0.01089179251930378,   0.01089179251930378,
          0.01089179251930378,   0.01089179251930378,   0.01089179251930378,
          0.021613681829707104,  0.021613681829707104,  0.021613681829707104,
          0.021613681829707104,  0.021613681829707104,  0.021613681829707104,
          0.007541838788255719,  0.007541838788255719,  0.007541838788255719,
          0.007541838788255719,  0.007541838788255719,  0.007541838788255719};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 13)
    {
      // Witherden-Vincent, 37 points, degree 13
      std::vector<T> x
          = {0.3333333333333333,    0.3333333333333333,   0.48907694645253935,
             0.0218461070949213,    0.0218461070949213,   0.48907694645253935,
             0.48907694645253935,   0.48907694645253935,  0.2213722862918329,
             0.5572554274163342,    0.5572554274163342,   0.2213722862918329,
             0.2213722862918329,    0.2213722862918329,   0.4269414142598004,
             0.14611717148039918,   0.14611717148039918,  0.4269414142598004,
             0.4269414142598004,    0.4269414142598004,   0.021509681108843184,
             0.9569806377823137,    0.9569806377823137,   0.021509681108843184,
             0.021509681108843184,  0.021509681108843184, 0.08789548303219732,
             0.7485071158999522,    0.7485071158999522,   0.08789548303219732,
             0.16359740106785048,   0.7485071158999522,   0.7485071158999522,
             0.16359740106785048,   0.16359740106785048,  0.08789548303219732,
             0.08789548303219732,   0.16359740106785048,  0.11092204280346339,
             0.8647077702954428,    0.8647077702954428,   0.11092204280346339,
             0.02437018690109383,   0.8647077702954428,   0.8647077702954428,
             0.02437018690109383,   0.02437018690109383,  0.11092204280346339,
             0.11092204280346339,   0.02437018690109383,  0.3084417608921178,
             0.6235459955536756,    0.6235459955536756,   0.3084417608921178,
             0.06801224355420665,   0.6235459955536756,   0.6235459955536756,
             0.06801224355420665,   0.06801224355420665,  0.3084417608921178,
             0.3084417608921178,    0.06801224355420665,  0.0051263891023823685,
             0.722357793124188,     0.722357793124188,    0.0051263891023823685,
             0.27251581777342965,   0.722357793124188,    0.722357793124188,
             0.27251581777342965,   0.27251581777342965,  0.0051263891023823685,
             0.0051263891023823685, 0.27251581777342965};
      std::vector<T> w = {
          0.03398001829341582,   0.011997200964447365,  0.011997200964447365,
          0.011997200964447365,  0.02913924255959999,   0.02913924255959999,
          0.02913924255959999,   0.027800983765226665,  0.027800983765226665,
          0.027800983765226665,  0.003026168551769586,  0.003026168551769586,
          0.003026168551769586,  0.01208951990579691,   0.01208951990579691,
          0.01208951990579691,   0.01208951990579691,   0.01208951990579691,
          0.01208951990579691,   0.007482700552582834,  0.007482700552582834,
          0.007482700552582834,  0.007482700552582834,  0.007482700552582834,
          0.007482700552582834,  0.017320638070424187,  0.017320638070424187,
          0.017320638070424187,  0.017320638070424187,  0.017320638070424187,
          0.017320638070424187,  0.0047953405017716316, 0.0047953405017716316,
          0.0047953405017716316, 0.0047953405017716316, 0.0047953405017716316,
          0.0047953405017716316};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 14)
    {
      // Witherden-Vincent, 42 points, degree 14
      std::vector<T> x
          = {0.17720553241254344,  0.6455889351749131,   0.6455889351749131,
             0.17720553241254344,  0.17720553241254344,  0.17720553241254344,
             0.41764471934045394,  0.16471056131909215,  0.16471056131909215,
             0.41764471934045394,  0.41764471934045394,  0.41764471934045394,
             0.0617998830908726,   0.8764002338182548,   0.8764002338182548,
             0.0617998830908726,   0.0617998830908726,   0.0617998830908726,
             0.4889639103621786,   0.02207217927564272,  0.02207217927564272,
             0.4889639103621786,   0.4889639103621786,   0.4889639103621786,
             0.27347752830883865,  0.4530449433823227,   0.4530449433823227,
             0.27347752830883865,  0.27347752830883865,  0.27347752830883865,
             0.019390961248701048, 0.9612180775025979,   0.9612180775025979,
             0.019390961248701048, 0.019390961248701048, 0.019390961248701048,
             0.29837288213625773,  0.6869801678080878,   0.6869801678080878,
             0.29837288213625773,  0.01464695005565441,  0.6869801678080878,
             0.6869801678080878,   0.01464695005565441,  0.01464695005565441,
             0.29837288213625773,  0.29837288213625773,  0.01464695005565441,
             0.05712475740364794,  0.7706085547749965,   0.7706085547749965,
             0.05712475740364794,  0.17226668782135557,  0.7706085547749965,
             0.7706085547749965,   0.17226668782135557,  0.17226668782135557,
             0.05712475740364794,  0.05712475740364794,  0.17226668782135557,
             0.336861459796345,    0.5702222908466832,   0.5702222908466832,
             0.336861459796345,    0.09291624935697182,  0.5702222908466832,
             0.5702222908466832,   0.09291624935697182,  0.09291624935697182,
             0.336861459796345,    0.336861459796345,    0.09291624935697182,
             0.001268330932872025, 0.8797571713701712,   0.8797571713701712,
             0.001268330932872025, 0.11897449769695685,  0.8797571713701712,
             0.8797571713701712,   0.11897449769695685,  0.11897449769695685,
             0.001268330932872025, 0.001268330932872025, 0.11897449769695685};
      std::vector<T> w
          = {0.021081294368496508, 0.021081294368496508, 0.021081294368496508,
             0.016394176772062674, 0.016394176772062674, 0.016394176772062674,
             0.007216849834888334, 0.007216849834888334, 0.007216849834888334,
             0.010941790684714445, 0.010941790684714445, 0.010941790684714445,
             0.025887052253645793, 0.025887052253645793, 0.025887052253645793,
             0.002461701801200041, 0.002461701801200041, 0.002461701801200041,
             0.00721815405676692,  0.00721815405676692,  0.00721815405676692,
             0.00721815405676692,  0.00721815405676692,  0.00721815405676692,
             0.012332876606281837, 0.012332876606281837, 0.012332876606281837,
             0.012332876606281837, 0.012332876606281837, 0.012332876606281837,
             0.019285755393530342, 0.019285755393530342, 0.019285755393530342,
             0.019285755393530342, 0.019285755393530342, 0.019285755393530342,
             0.002505114419250336, 0.002505114419250336, 0.002505114419250336,
             0.002505114419250336, 0.002505114419250336, 0.002505114419250336};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 15)
    {
      // Witherden-Vincent, 49 points, degree 15
      std::vector<T> x
          = {0.3333333333333333,   0.3333333333333333,   0.4053622141339755,
             0.18927557173204904,  0.18927557173204904,  0.4053622141339755,
             0.4053622141339755,   0.4053622141339755,   0.07017355289998602,
             0.859652894200028,    0.859652894200028,    0.07017355289998602,
             0.07017355289998602,  0.07017355289998602,  0.47417068143801977,
             0.05165863712396042,  0.05165863712396042,  0.47417068143801977,
             0.47417068143801977,  0.47417068143801977,  0.2263787134203496,
             0.5472425731593008,   0.5472425731593008,   0.2263787134203496,
             0.2263787134203496,   0.2263787134203496,   0.4949969567691262,
             0.010006086461747632, 0.010006086461747632, 0.4949969567691262,
             0.4949969567691262,   0.4949969567691262,   0.015811726250988645,
             0.9683765474980227,   0.9683765474980227,   0.015811726250988645,
             0.015811726250988645, 0.015811726250988645, 0.01837611238568109,
             0.6669756448018681,   0.6669756448018681,   0.01837611238568109,
             0.3146482428124508,   0.6669756448018681,   0.6669756448018681,
             0.3146482428124508,   0.3146482428124508,   0.01837611238568109,
             0.01837611238568109,  0.3146482428124508,   0.009139237037308396,
             0.919912157726236,    0.919912157726236,    0.009139237037308396,
             0.07094860523645553,  0.919912157726236,    0.919912157726236,
             0.07094860523645553,  0.07094860523645553,  0.009139237037308396,
             0.009139237037308396, 0.07094860523645553,  0.19053558947639393,
             0.7152223569314508,   0.7152223569314508,   0.19053558947639393,
             0.09424205359215536,  0.7152223569314508,   0.7152223569314508,
             0.09424205359215536,  0.09424205359215536,  0.19053558947639393,
             0.19053558947639393,  0.09424205359215536,  0.16806864522241435,
             0.8132926410494192,   0.8132926410494192,   0.16806864522241435,
             0.018638713728166376, 0.8132926410494192,   0.8132926410494192,
             0.018638713728166376, 0.018638713728166376, 0.16806864522241435,
             0.16806864522241435,  0.018638713728166376, 0.33895061147527716,
             0.5652526648771142,   0.5652526648771142,   0.33895061147527716,
             0.09579672364760859,  0.5652526648771142,   0.5652526648771142,
             0.09579672364760859,  0.09579672364760859,  0.33895061147527716,
             0.33895061147527716,  0.09579672364760859};
      std::vector<T> w = {
          0.022167693691092036,  0.021356890785730283,  0.021356890785730283,
          0.021356890785730283,  0.008222368781312581,  0.008222368781312581,
          0.008222368781312581,  0.008698074000381707,  0.008698074000381707,
          0.008698074000381707,  0.023391680864354815,  0.023391680864354815,
          0.023391680864354815,  0.004786923091230043,  0.004786923091230043,
          0.004786923091230043,  0.0014803873189526877, 0.0014803873189526877,
          0.0014803873189526877, 0.007801286415287982,  0.007801286415287982,
          0.007801286415287982,  0.007801286415287982,  0.007801286415287982,
          0.007801286415287982,  0.0020149266860090497, 0.0020149266860090497,
          0.0020149266860090497, 0.0020149266860090497, 0.0020149266860090497,
          0.0020149266860090497, 0.014360293462600671,  0.014360293462600671,
          0.014360293462600671,  0.014360293462600671,  0.014360293462600671,
          0.014360293462600671,  0.005836310590787923,  0.005836310590787923,
          0.005836310590787923,  0.005836310590787923,  0.005836310590787923,
          0.005836310590787923,  0.015657738142484643,  0.015657738142484643,
          0.015657738142484643,  0.015657738142484643,  0.015657738142484643,
          0.015657738142484643};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 16)
    {
      // Witherden-Vincent, 55 points, degree 16
      std::vector<T> x = {0.3333333333333333,
                          0.3333333333333333,
                          0.24599007046714175,
                          0.5080198590657166,
                          0.5080198590657166,
                          0.24599007046714175,
                          0.24599007046714175,
                          0.24599007046714175,
                          0.4155848968854205,
                          0.16883020622915898,
                          0.16883020622915898,
                          0.4155848968854205,
                          0.4155848968854205,
                          0.4155848968854205,
                          0.08535556658670032,
                          0.8292888668265993,
                          0.8292888668265993,
                          0.08535556658670032,
                          0.08535556658670032,
                          0.08535556658670032,
                          0.16191864419127122,
                          0.6761627116174576,
                          0.6761627116174576,
                          0.16191864419127122,
                          0.16191864419127122,
                          0.16191864419127122,
                          0.5,
                          0.0,
                          0.0,
                          0.5,
                          0.5,
                          0.5,
                          0.4752807275459421,
                          0.04943854490811579,
                          0.04943854490811579,
                          0.4752807275459421,
                          0.4752807275459421,
                          0.4752807275459421,
                          0.054755174914703116,
                          0.7541700614447678,
                          0.7541700614447678,
                          0.054755174914703116,
                          0.19107476364052908,
                          0.7541700614447678,
                          0.7541700614447678,
                          0.19107476364052908,
                          0.19107476364052908,
                          0.054755174914703116,
                          0.054755174914703116,
                          0.19107476364052908,
                          0.02320342776881371,
                          0.9682443680309587,
                          0.9682443680309587,
                          0.02320342776881371,
                          0.008552204200227611,
                          0.9682443680309587,
                          0.9682443680309587,
                          0.008552204200227611,
                          0.008552204200227611,
                          0.02320342776881371,
                          0.02320342776881371,
                          0.008552204200227611,
                          0.018931778280405914,
                          0.6493036982454464,
                          0.6493036982454464,
                          0.018931778280405914,
                          0.3317645234741477,
                          0.6493036982454464,
                          0.6493036982454464,
                          0.3317645234741477,
                          0.3317645234741477,
                          0.018931778280405914,
                          0.018931778280405914,
                          0.3317645234741477,
                          0.019030129743697447,
                          0.9002737032704297,
                          0.9002737032704297,
                          0.019030129743697447,
                          0.08069616698587292,
                          0.9002737032704297,
                          0.9002737032704297,
                          0.08069616698587292,
                          0.08069616698587292,
                          0.019030129743697447,
                          0.019030129743697447,
                          0.08069616698587292,
                          0.1026061902393981,
                          0.5891488405642479,
                          0.5891488405642479,
                          0.1026061902393981,
                          0.30824496919635397,
                          0.5891488405642479,
                          0.5891488405642479,
                          0.30824496919635397,
                          0.30824496919635397,
                          0.1026061902393981,
                          0.1026061902393981,
                          0.30824496919635397,
                          0.00593635001682227,
                          0.8066218674993957,
                          0.8066218674993957,
                          0.00593635001682227,
                          0.1874417824837821,
                          0.8066218674993957,
                          0.8066218674993957,
                          0.1874417824837821,
                          0.1874417824837821,
                          0.00593635001682227,
                          0.00593635001682227,
                          0.1874417824837821};
      std::vector<T> w = {
          0.022632283036909395,  0.020546461571849476,  0.020546461571849476,
          0.020546461571849476,  0.02035591665621268,   0.02035591665621268,
          0.02035591665621268,   0.007390817345112202,  0.007390817345112202,
          0.007390817345112202,  0.01470920484949405,   0.01470920484949405,
          0.01470920484949405,   0.0022092731560752845, 0.0022092731560752845,
          0.0022092731560752845, 0.01298716664913858,   0.01298716664913858,
          0.01298716664913858,   0.00946913623220785,   0.00946913623220785,
          0.00946913623220785,   0.00946913623220785,   0.00946913623220785,
          0.00946913623220785,   0.0008272333574175241, 0.0008272333574175241,
          0.0008272333574175241, 0.0008272333574175241, 0.0008272333574175241,
          0.0008272333574175241, 0.007504300892142903,  0.007504300892142903,
          0.007504300892142903,  0.007504300892142903,  0.007504300892142903,
          0.007504300892142903,  0.003973796966696249,  0.003973796966696249,
          0.003973796966696249,  0.003973796966696249,  0.003973796966696249,
          0.003973796966696249,  0.015991805039685034,  0.015991805039685034,
          0.015991805039685034,  0.015991805039685034,  0.015991805039685034,
          0.015991805039685034,  0.0026955935584244057, 0.0026955935584244057,
          0.0026955935584244057, 0.0026955935584244057, 0.0026955935584244057,
          0.0026955935584244057};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 17)
    {
      // Witherden-Vincent, 60 points, degree 17
      std::vector<T> x
          = {0.4171034443615992,   0.1657931112768016,   0.1657931112768016,
             0.4171034443615992,   0.4171034443615992,   0.4171034443615992,
             0.014755491660753954, 0.9704890166784921,   0.9704890166784921,
             0.014755491660753954, 0.014755491660753954, 0.014755491660753954,
             0.4655978716188903,   0.06880425676221939,  0.06880425676221939,
             0.4655978716188903,   0.4655978716188903,   0.4655978716188903,
             0.18035811626637063,  0.6392837674672588,   0.6392837674672588,
             0.18035811626637063,  0.18035811626637063,  0.18035811626637063,
             0.06665406347959693,  0.8666918730408062,   0.8666918730408062,
             0.06665406347959693,  0.06665406347959693,  0.06665406347959693,
             0.28570650243658663,  0.42858699512682674,  0.42858699512682674,
             0.28570650243658663,  0.28570650243658663,  0.28570650243658663,
             0.0160176423621193,   0.824790070165088,    0.824790070165088,
             0.0160176423621193,   0.15919228747279268,  0.824790070165088,
             0.824790070165088,    0.15919228747279268,  0.15919228747279268,
             0.0160176423621193,   0.0160176423621193,   0.15919228747279268,
             0.3062815917461865,   0.6263690303864523,   0.6263690303864523,
             0.3062815917461865,   0.0673493778673612,   0.6263690303864523,
             0.6263690303864523,   0.0673493778673612,   0.0673493778673612,
             0.3062815917461865,   0.3062815917461865,   0.0673493778673612,
             0.013229672760086894, 0.5712948679446841,   0.5712948679446841,
             0.013229672760086894, 0.41547545929522905,  0.5712948679446841,
             0.5712948679446841,   0.41547545929522905,  0.41547545929522905,
             0.013229672760086894, 0.013229672760086894, 0.41547545929522905,
             0.07804234056828242,  0.7532351459364581,   0.7532351459364581,
             0.07804234056828242,  0.16872251349525946,  0.7532351459364581,
             0.7532351459364581,   0.16872251349525946,  0.16872251349525946,
             0.07804234056828242,  0.07804234056828242,  0.16872251349525946,
             0.013135870834002694, 0.7150722591106424,   0.7150722591106424,
             0.013135870834002694, 0.2717918700553548,   0.7150722591106424,
             0.7150722591106424,   0.2717918700553548,   0.2717918700553548,
             0.013135870834002694, 0.013135870834002694, 0.2717918700553548,
             0.011575175903180615, 0.9159193532978169,   0.9159193532978169,
             0.011575175903180615, 0.07250547079900242,  0.9159193532978169,
             0.9159193532978169,   0.07250547079900242,  0.07250547079900242,
             0.011575175903180615, 0.011575175903180615, 0.07250547079900242,
             0.1575054779268699,   0.5432755795961598,   0.5432755795961598,
             0.1575054779268699,   0.29921894247697034,  0.5432755795961598,
             0.5432755795961598,   0.29921894247697034,  0.29921894247697034,
             0.1575054779268699,   0.1575054779268699,   0.29921894247697034};
      std::vector<T> w = {
          0.013655463264051053,  0.013655463264051053,  0.013655463264051053,
          0.001386943788818821,  0.001386943788818821,  0.001386943788818821,
          0.012509725475248678,  0.012509725475248678,  0.012509725475248678,
          0.013156315294008993,  0.013156315294008993,  0.013156315294008993,
          0.006229500401152721,  0.006229500401152721,  0.006229500401152721,
          0.01885811857639764,   0.01885811857639764,   0.01885811857639764,
          0.003989150102964797,  0.003989150102964797,  0.003989150102964797,
          0.003989150102964797,  0.003989150102964797,  0.003989150102964797,
          0.011243886273345534,  0.011243886273345534,  0.011243886273345534,
          0.011243886273345534,  0.011243886273345534,  0.011243886273345534,
          0.005199219977919768,  0.005199219977919768,  0.005199219977919768,
          0.005199219977919768,  0.005199219977919768,  0.005199219977919768,
          0.01027894916022726,   0.01027894916022726,   0.01027894916022726,
          0.01027894916022726,   0.01027894916022726,   0.01027894916022726,
          0.004346107250500596,  0.004346107250500596,  0.004346107250500596,
          0.004346107250500596,  0.004346107250500596,  0.004346107250500596,
          0.0022921742008679335, 0.0022921742008679335, 0.0022921742008679335,
          0.0022921742008679335, 0.0022921742008679335, 0.0022921742008679335,
          0.013085812967668494,  0.013085812967668494,  0.013085812967668494,
          0.013085812967668494,  0.013085812967668494,  0.013085812967668494};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 18)
    {
      // Witherden-Vincent, 67 points, degree 18
      std::vector<T> x
          = {0.3333333333333333,    0.3333333333333333,   0.39995562806757623,
             0.20008874386484754,   0.20008874386484754,  0.39995562806757623,
             0.39995562806757623,   0.39995562806757623,  0.48758030157486953,
             0.024839396850260875,  0.024839396850260875, 0.48758030157486953,
             0.48758030157486953,   0.48758030157486953,  0.46180950640644924,
             0.07638098718710154,   0.07638098718710154,  0.46180950640644924,
             0.46180950640644924,   0.46180950640644924,  0.24226470251427196,
             0.5154705949714561,    0.5154705949714561,   0.24226470251427196,
             0.24226470251427196,   0.24226470251427196,  0.038830256088685594,
             0.9223394878226288,    0.9223394878226288,   0.038830256088685594,
             0.038830256088685594,  0.038830256088685594, 0.0919477421216432,
             0.8161045157567136,    0.8161045157567136,   0.0919477421216432,
             0.0919477421216432,    0.0919477421216432,   0.04580491585986078,
             0.7703723762146752,    0.7703723762146752,   0.04580491585986078,
             0.183822707925464,     0.7703723762146752,   0.7703723762146752,
             0.183822707925464,     0.183822707925464,    0.04580491585986078,
             0.04580491585986078,   0.183822707925464,    0.20634925743383795,
             0.6709539851942345,    0.6709539851942345,   0.20634925743383795,
             0.12269675737192755,   0.6709539851942345,   0.6709539851942345,
             0.12269675737192755,   0.12269675737192755,  0.20634925743383795,
             0.20634925743383795,   0.12269675737192755,  0.0038976110334733825,
             0.6004189546342569,    0.6004189546342569,   0.0038976110334733825,
             0.3956834343322697,    0.6004189546342569,   0.6004189546342569,
             0.3956834343322697,    0.3956834343322697,   0.0038976110334733825,
             0.0038976110334733825, 0.3956834343322697,   0.013462016741444989,
             0.8783421894675217,    0.8783421894675217,   0.013462016741444989,
             0.10819579379103329,   0.8783421894675217,   0.8783421894675217,
             0.10819579379103329,   0.10819579379103329,  0.013462016741444989,
             0.013462016741444989,  0.10819579379103329,  0.040260283469908065,
             0.6399880920047146,    0.6399880920047146,   0.040260283469908065,
             0.31975162452537736,   0.6399880920047146,   0.6399880920047146,
             0.31975162452537736,   0.31975162452537736,  0.040260283469908065,
             0.040260283469908065,  0.31975162452537736,  0.005298335186609765,
             0.7589294798551985,    0.7589294798551985,   0.005298335186609765,
             0.23577218495819174,   0.7589294798551985,   0.7589294798551985,
             0.23577218495819174,   0.23577218495819174,  0.005298335186609765,
             0.005298335186609765,  0.23577218495819174,  0.000548360042042319,
             0.9723607289627957,    0.9723607289627957,   0.000548360042042319,
             0.027090910995162015,  0.9723607289627957,   0.9723607289627957,
             0.027090910995162015,  0.027090910995162015, 0.000548360042042319,
             0.000548360042042319,  0.027090910995162015, 0.12058769516392465,
             0.5459187753861946,    0.5459187753861946,   0.12058769516392465,
             0.33349352944988075,   0.5459187753861946,   0.5459187753861946,
             0.33349352944988075,   0.33349352944988075,  0.12058769516392465,
             0.12058769516392465,   0.33349352944988075};
      std::vector<T> w = {
          0.018177867650713334,  0.016652235016695067,  0.016652235016695067,
          0.016652235016695067,  0.0060233238169998555, 0.0060233238169998555,
          0.0060233238169998555, 0.009474585753389433,  0.009474585753389433,
          0.009474585753389433,  0.01823754470447182,   0.01823754470447182,
          0.01823754470447182,   0.0035646630098594852, 0.0035646630098594852,
          0.0035646630098594852, 0.008279579976001624,  0.008279579976001624,
          0.008279579976001624,  0.006879808117471103,  0.006879808117471103,
          0.006879808117471103,  0.006879808117471103,  0.006879808117471103,
          0.006879808117471103,  0.011890955450076415,  0.011890955450076415,
          0.011890955450076415,  0.011890955450076415,  0.011890955450076415,
          0.011890955450076415,  0.0022652672511285325, 0.0022652672511285325,
          0.0022652672511285325, 0.0022652672511285325, 0.0022652672511285325,
          0.0022652672511285325, 0.003420055059803591,  0.003420055059803591,
          0.003420055059803591,  0.003420055059803591,  0.003420055059803591,
          0.003420055059803591,  0.008873744551010202,  0.008873744551010202,
          0.008873744551010202,  0.008873744551010202,  0.008873744551010202,
          0.008873744551010202,  0.002505330437289861,  0.002505330437289861,
          0.002505330437289861,  0.002505330437289861,  0.002505330437289861,
          0.002505330437289861,  0.0006114740634805449, 0.0006114740634805449,
          0.0006114740634805449, 0.0006114740634805449, 0.0006114740634805449,
          0.0006114740634805449, 0.01274108765591222,   0.01274108765591222,
          0.01274108765591222,   0.01274108765591222,   0.01274108765591222,
          0.01274108765591222};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 19)
    {
      // Witherden-Vincent, 73 points, degree 19
      std::vector<T> x
          = {0.3333333333333333,    0.3333333333333333,   0.05252389035120897,
             0.8949522192975821,    0.8949522192975821,   0.05252389035120897,
             0.05252389035120897,   0.05252389035120897,  0.4925126750413369,
             0.014974649917326261,  0.014974649917326261, 0.4925126750413369,
             0.4925126750413369,    0.4925126750413369,   0.11144887332302138,
             0.7771022533539572,    0.7771022533539572,   0.11144887332302138,
             0.11144887332302138,   0.11144887332302138,  0.45919420103954367,
             0.08161159792091269,   0.08161159792091269,  0.45919420103954367,
             0.45919420103954367,   0.45919420103954367,  0.4039697225519012,
             0.1920605548961976,    0.1920605548961976,   0.4039697225519012,
             0.4039697225519012,    0.4039697225519012,   0.17817010478176432,
             0.6436597904364714,    0.6436597904364714,   0.17817010478176432,
             0.17817010478176432,   0.17817010478176432,  0.011639461183789447,
             0.9767210776324211,    0.9767210776324211,   0.011639461183789447,
             0.011639461183789447,  0.011639461183789447, 0.2551616329136077,
             0.4896767341727846,    0.4896767341727846,   0.2551616329136077,
             0.2551616329136077,    0.2551616329136077,   0.13069767626803241,
             0.8301564644002754,    0.8301564644002754,   0.13069767626803241,
             0.03914585933169222,   0.8301564644002754,   0.8301564644002754,
             0.03914585933169222,   0.03914585933169222,  0.13069767626803241,
             0.13069767626803241,   0.03914585933169222,  0.31131762980954125,
             0.5593698057203009,    0.5593698057203009,   0.31131762980954125,
             0.12931256447015782,   0.5593698057203009,   0.5593698057203009,
             0.12931256447015782,   0.12931256447015782,  0.31131762980954125,
             0.31131762980954125,   0.12931256447015782,  0.0020689258966048074,
             0.6333132931287841,    0.6333132931287841,   0.0020689258966048074,
             0.36461778097461106,   0.6333132931287841,   0.6333132931287841,
             0.36461778097461106,   0.36461778097461106,  0.0020689258966048074,
             0.0020689258966048074, 0.36461778097461106,  0.07456029460162668,
             0.7040048199660421,    0.7040048199660421,   0.07456029460162668,
             0.22143488543233122,   0.7040048199660421,   0.7040048199660421,
             0.22143488543233122,   0.22143488543233122,  0.07456029460162668,
             0.07456029460162668,   0.22143488543233122,  0.005007288257354491,
             0.8525669543768892,    0.8525669543768892,   0.005007288257354491,
             0.1424257573657563,    0.8525669543768892,   0.8525669543768892,
             0.1424257573657563,    0.1424257573657563,   0.005007288257354491,
             0.005007288257354491,  0.1424257573657563,   0.04088801119601688,
             0.6050839790687079,    0.6050839790687079,   0.04088801119601688,
             0.35402800973527515,   0.6050839790687079,   0.6050839790687079,
             0.35402800973527515,   0.35402800973527515,  0.04088801119601688,
             0.04088801119601688,   0.35402800973527515,  0.24189457896057956,
             0.7431813689574364,    0.7431813689574364,   0.24189457896057956,
             0.014924052081984072,  0.7431813689574364,   0.7431813689574364,
             0.014924052081984072,  0.014924052081984072, 0.24189457896057956,
             0.24189457896057956,   0.014924052081984072, 0.0600862753223067,
             0.9301376988768052,    0.9301376988768052,   0.0600862753223067,
             0.009776025800888155,  0.9301376988768052,   0.9301376988768052,
             0.009776025800888155,  0.009776025800888155, 0.0600862753223067,
             0.0600862753223067,    0.009776025800888155};
      std::vector<T> w = {
          0.017234698852006167,  0.0035546282988990654, 0.0035546282988990654,
          0.0035546282988990654, 0.005160877571472141,  0.005160877571472141,
          0.005160877571472141,  0.00761717554650915,   0.00761717554650915,
          0.00761717554650915,   0.011491795013370804,  0.011491795013370804,
          0.011491795013370804,  0.015768767446577486,  0.015768767446577486,
          0.015768767446577486,  0.012325957424095427,  0.012325957424095427,
          0.012325957424095427,  0.0008826613882214238, 0.0008826613882214238,
          0.0008826613882214238, 0.015876509683001538,  0.015876509683001538,
          0.015876509683001538,  0.004847742243427523,  0.004847742243427523,
          0.004847742243427523,  0.004847742243427523,  0.004847742243427523,
          0.004847742243427523,  0.013173160988695367,  0.013173160988695367,
          0.013173160988695367,  0.013173160988695367,  0.013173160988695367,
          0.013173160988695367,  0.0016410382759179097, 0.0016410382759179097,
          0.0016410382759179097, 0.0016410382759179097, 0.0016410382759179097,
          0.0016410382759179097, 0.009053972465606226,  0.009053972465606226,
          0.009053972465606226,  0.009053972465606226,  0.009053972465606226,
          0.009053972465606226,  0.0014631575517351002, 0.0014631575517351002,
          0.0014631575517351002, 0.0014631575517351002, 0.0014631575517351002,
          0.0014631575517351002, 0.008051081382012054,  0.008051081382012054,
          0.008051081382012054,  0.008051081382012054,  0.008051081382012054,
          0.008051081382012054,  0.004227943749768248,  0.004227943749768248,
          0.004227943749768248,  0.004227943749768248,  0.004227943749768248,
          0.004227943749768248,  0.001663600681429694,  0.001663600681429694,
          0.001663600681429694,  0.001663600681429694,  0.001663600681429694,
          0.001663600681429694};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 20)
    {
      // Witherden-Vincent, 79 points, degree 20
      std::vector<T> x = {
          0.3333333333333333,    0.3333333333333333,    0.2545792676733391,
          0.4908414646533218,    0.4908414646533218,    0.2545792676733391,
          0.2545792676733391,    0.2545792676733391,    0.010976141028397763,
          0.9780477179432044,    0.9780477179432044,    0.010976141028397763,
          0.010976141028397763,  0.010976141028397763,  0.1093835967117146,
          0.7812328065765708,    0.7812328065765708,    0.1093835967117146,
          0.1093835967117146,    0.1093835967117146,    0.18629499774454095,
          0.6274100045109181,    0.6274100045109181,    0.18629499774454095,
          0.18629499774454095,   0.18629499774454095,   0.44555105695592484,
          0.10889788608815038,   0.10889788608815038,   0.44555105695592484,
          0.44555105695592484,   0.44555105695592484,   0.037310880598884696,
          0.9253782388022306,    0.9253782388022306,    0.037310880598884696,
          0.037310880598884696,  0.037310880598884696,  0.39342534781709987,
          0.2131493043658003,    0.2131493043658003,    0.39342534781709987,
          0.39342534781709987,   0.39342534781709987,   0.476245611540499,
          0.047508776919001974,  0.047508776919001974,  0.476245611540499,
          0.476245611540499,     0.476245611540499,     0.0075707805046965285,
          0.8332955118382362,    0.8332955118382362,    0.0075707805046965285,
          0.15913370765706722,   0.8332955118382362,    0.8332955118382362,
          0.15913370765706722,   0.15913370765706722,   0.0075707805046965285,
          0.0075707805046965285, 0.15913370765706722,   0.04656036490766432,
          0.7549215028635475,    0.7549215028635475,    0.04656036490766432,
          0.19851813222878817,   0.7549215028635475,    0.7549215028635475,
          0.19851813222878817,   0.19851813222878817,   0.04656036490766432,
          0.04656036490766432,   0.19851813222878817,   0.06409058560843406,
          0.9310544767839422,    0.9310544767839422,    0.06409058560843406,
          0.0048549376076237535, 0.9310544767839422,    0.9310544767839422,
          0.0048549376076237535, 0.0048549376076237535, 0.06409058560843406,
          0.06409058560843406,   0.0048549376076237535, 0.05498747914298681,
          0.6118777035474257,    0.6118777035474257,    0.05498747914298681,
          0.3331348173095875,    0.6118777035474257,    0.6118777035474257,
          0.3331348173095875,    0.3331348173095875,    0.05498747914298681,
          0.05498747914298681,   0.3331348173095875,    0.09995229628813866,
          0.8616840189364867,    0.8616840189364867,    0.09995229628813866,
          0.03836368477537459,   0.8616840189364867,    0.8616840189364867,
          0.03836368477537459,   0.03836368477537459,   0.09995229628813866,
          0.09995229628813866,   0.03836368477537459,   0.10622720472027004,
          0.6781657378896355,    0.6781657378896355,    0.10622720472027004,
          0.2156070573900944,    0.6781657378896355,    0.6781657378896355,
          0.2156070573900944,    0.2156070573900944,    0.10622720472027004,
          0.10622720472027004,   0.2156070573900944,    0.42002375881622406,
          0.5701446928909734,    0.5701446928909734,    0.42002375881622406,
          0.00983154829280256,   0.5701446928909734,    0.5701446928909734,
          0.00983154829280256,   0.00983154829280256,   0.42002375881622406,
          0.42002375881622406,   0.00983154829280256,   0.317860123835772,
          0.5423318041724281,    0.5423318041724281,    0.317860123835772,
          0.1398080719917999,    0.5423318041724281,    0.5423318041724281,
          0.1398080719917999,    0.1398080719917999,    0.317860123835772,
          0.317860123835772,     0.1398080719917999,    0.010737212856011088,
          0.7086813757203236,    0.7086813757203236,    0.010737212856011088,
          0.2805814114236652,    0.7086813757203236,    0.7086813757203236,
          0.2805814114236652,    0.2805814114236652,    0.010737212856011088,
          0.010737212856011088,  0.2805814114236652};
      std::vector<T> w = {
          0.013910110701453116,  0.014083201307520247,  0.014083201307520247,
          0.014083201307520247,  0.0007988407910666199, 0.0007988407910666199,
          0.0007988407910666199, 0.007830230776074533,  0.007830230776074533,
          0.007830230776074533,  0.009173462974252915,  0.009173462974252915,
          0.009173462974252915,  0.009452399933232448,  0.009452399933232448,
          0.009452399933232448,  0.0021612754106655773, 0.0021612754106655773,
          0.0021612754106655773, 0.013788050629070459,  0.013788050629070459,
          0.013788050629070459,  0.007101825303408441,  0.007101825303408441,
          0.007101825303408441,  0.0022028974185584974, 0.0022028974185584974,
          0.0022028974185584974, 0.0022028974185584974, 0.0022028974185584974,
          0.0022028974185584974, 0.00598639857895469,   0.00598639857895469,
          0.00598639857895469,   0.00598639857895469,   0.00598639857895469,
          0.00598639857895469,   0.0011298696021258656, 0.0011298696021258656,
          0.0011298696021258656, 0.0011298696021258656, 0.0011298696021258656,
          0.0011298696021258656, 0.008667225567219333,  0.008667225567219333,
          0.008667225567219333,  0.008667225567219333,  0.008667225567219333,
          0.008667225567219333,  0.004145711527613858,  0.004145711527613858,
          0.004145711527613858,  0.004145711527613858,  0.004145711527613858,
          0.004145711527613858,  0.00772260782209923,   0.00772260782209923,
          0.00772260782209923,   0.00772260782209923,   0.00772260782209923,
          0.00772260782209923,   0.003695681500255298,  0.003695681500255298,
          0.003695681500255298,  0.003695681500255298,  0.003695681500255298,
          0.003695681500255298,  0.011691745731827737,  0.011691745731827737,
          0.011691745731827737,  0.011691745731827737,  0.011691745731827737,
          0.011691745731827737,  0.003578200238457685,  0.003578200238457685,
          0.003578200238457685,  0.003578200238457685,  0.003578200238457685,
          0.003578200238457685};
      return {std::move(x), std::move(w)};
    }
    else
    {
      throw std::runtime_error(
          "Witherden-Vincent not implemented for this order.");
    }
  }
  case cell::type::quadrilateral:
  {
    if (m <= 1)
    {
      // Witherden-Vincent, 1 point, degree 1
      std::vector<T> x = {0.5, 0.5};
      std::vector<T> w = {1.0};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 3)
    {
      // Witherden-Vincent, 4 points, degree 3
      std::vector<T> x
          = {0.7886751345948129, 0.7886751345948129, 0.7886751345948129,
             0.2113248654051871, 0.2113248654051871, 0.7886751345948129,
             0.2113248654051871, 0.2113248654051871};
      std::vector<T> w = {0.25, 0.25, 0.25, 0.25};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 5)
    {
      // Witherden-Vincent, 8 points, degree 5
      std::vector<T> x = {0.8415650255319866,
                          0.5,
                          0.5,
                          0.8415650255319866,
                          0.15843497446801338,
                          0.5,
                          0.5,
                          0.15843497446801338,
                          0.9409585518440984,
                          0.9409585518440984,
                          0.9409585518440984,
                          0.059041448155901566,
                          0.059041448155901566,
                          0.9409585518440984,
                          0.059041448155901566,
                          0.059041448155901566};
      std::vector<T> w
          = {0.20408163265306123, 0.20408163265306123, 0.20408163265306123,
             0.20408163265306123, 0.04591836734693878, 0.04591836734693878,
             0.04591836734693878, 0.04591836734693878};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 7)
    {
      // Witherden-Vincent, 12 points, degree 7
      std::vector<T> x = {0.9629100498862757,
                          0.5,
                          0.5,
                          0.9629100498862757,
                          0.03708995011372427,
                          0.5,
                          0.5,
                          0.03708995011372427,
                          0.9029898914592994,
                          0.9029898914592994,
                          0.9029898914592994,
                          0.09701010854070063,
                          0.09701010854070063,
                          0.9029898914592994,
                          0.09701010854070063,
                          0.09701010854070063,
                          0.6902772166041579,
                          0.6902772166041579,
                          0.6902772166041579,
                          0.3097227833958422,
                          0.3097227833958422,
                          0.6902772166041579,
                          0.3097227833958422,
                          0.3097227833958422};
      std::vector<T> w
          = {0.06049382716049383, 0.06049382716049383, 0.06049382716049383,
             0.06049382716049383, 0.05935794367265756, 0.05935794367265756,
             0.05935794367265756, 0.05935794367265756, 0.13014822916684862,
             0.13014822916684862, 0.13014822916684862, 0.13014822916684862};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 9)
    {
      // Witherden-Vincent, 20 points, degree 9
      std::vector<T> x = {0.7444634284871845,
                          0.5,
                          0.5,
                          0.7444634284871845,
                          0.25553657151281545,
                          0.5,
                          0.5,
                          0.25553657151281545,
                          0.9698276290484189,
                          0.9698276290484189,
                          0.9698276290484189,
                          0.030172370951581148,
                          0.030172370951581148,
                          0.9698276290484189,
                          0.030172370951581148,
                          0.030172370951581148,
                          0.8454402752431719,
                          0.8454402752431719,
                          0.8454402752431719,
                          0.15455972475682805,
                          0.15455972475682805,
                          0.8454402752431719,
                          0.15455972475682805,
                          0.15455972475682805,
                          0.9593102205283611,
                          0.6724360126822018,
                          0.6724360126822018,
                          0.9593102205283611,
                          0.9593102205283611,
                          0.32756398731779823,
                          0.32756398731779823,
                          0.9593102205283611,
                          0.04068977947163887,
                          0.6724360126822018,
                          0.6724360126822018,
                          0.04068977947163887,
                          0.04068977947163887,
                          0.32756398731779823,
                          0.32756398731779823,
                          0.04068977947163887};
      std::vector<T> w
          = {0.11354099017168726,  0.11354099017168726,  0.11354099017168726,
             0.11354099017168726,  0.010682807966443941, 0.010682807966443941,
             0.010682807966443941, 0.010682807966443941, 0.053550090231715405,
             0.053550090231715405, 0.053550090231715405, 0.053550090231715405,
             0.0361130558150767,   0.0361130558150767,   0.0361130558150767,
             0.0361130558150767,   0.0361130558150767,   0.0361130558150767,
             0.0361130558150767,   0.0361130558150767};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 11)
    {
      // Witherden-Vincent, 28 points, degree 11
      std::vector<T> x = {0.857308914832303,
                          0.5,
                          0.5,
                          0.857308914832303,
                          0.14269108516769705,
                          0.5,
                          0.5,
                          0.14269108516769705,
                          0.6368286050857298,
                          0.6368286050857298,
                          0.6368286050857298,
                          0.3631713949142702,
                          0.3631713949142702,
                          0.6368286050857298,
                          0.3631713949142702,
                          0.3631713949142702,
                          0.8183019661061505,
                          0.8183019661061505,
                          0.8183019661061505,
                          0.18169803389384948,
                          0.18169803389384948,
                          0.8183019661061505,
                          0.18169803389384948,
                          0.18169803389384948,
                          0.9758151943920167,
                          0.9077827168448193,
                          0.9077827168448193,
                          0.9758151943920167,
                          0.9758151943920167,
                          0.0922172831551808,
                          0.0922172831551808,
                          0.9758151943920167,
                          0.02418480560798327,
                          0.9077827168448193,
                          0.9077827168448193,
                          0.02418480560798327,
                          0.02418480560798327,
                          0.0922172831551808,
                          0.0922172831551808,
                          0.02418480560798327,
                          0.6731036000238227,
                          0.9677839357437955,
                          0.9677839357437955,
                          0.6731036000238227,
                          0.6731036000238227,
                          0.032216064256204456,
                          0.032216064256204456,
                          0.6731036000238227,
                          0.32689639997617725,
                          0.9677839357437955,
                          0.9677839357437955,
                          0.32689639997617725,
                          0.32689639997617725,
                          0.032216064256204456,
                          0.032216064256204456,
                          0.32689639997617725};
      std::vector<T> w
          = {0.054350109967178,    0.054350109967178,    0.054350109967178,
             0.054350109967178,    0.06931852574596277,  0.06931852574596277,
             0.06931852574596277,  0.06931852574596277,  0.053483409469562024,
             0.053483409469562024, 0.053483409469562024, 0.053483409469562024,
             0.011018642278745773, 0.011018642278745773, 0.011018642278745773,
             0.011018642278745773, 0.011018642278745773, 0.011018642278745773,
             0.011018642278745773, 0.011018642278745773, 0.025405335129902826,
             0.025405335129902826, 0.025405335129902826, 0.025405335129902826,
             0.025405335129902826, 0.025405335129902826, 0.025405335129902826,
             0.025405335129902826};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 13)
    {
      // Witherden-Vincent, 37 points, degree 13
      std::vector<T> x = {0.5,
                          0.5,
                          0.9917306663066228,
                          0.5,
                          0.5,
                          0.9917306663066228,
                          0.008269333693377214,
                          0.5,
                          0.5,
                          0.008269333693377214,
                          0.8199307091835549,
                          0.5,
                          0.5,
                          0.8199307091835549,
                          0.18006929081644513,
                          0.5,
                          0.5,
                          0.18006929081644513,
                          0.9593892440398557,
                          0.9593892440398557,
                          0.9593892440398557,
                          0.040610755960144296,
                          0.040610755960144296,
                          0.9593892440398557,
                          0.040610755960144296,
                          0.040610755960144296,
                          0.6898142525933719,
                          0.6898142525933719,
                          0.6898142525933719,
                          0.3101857474066281,
                          0.3101857474066281,
                          0.6898142525933719,
                          0.3101857474066281,
                          0.3101857474066281,
                          0.8497771082566755,
                          0.8497771082566755,
                          0.8497771082566755,
                          0.15022289174332445,
                          0.15022289174332445,
                          0.8497771082566755,
                          0.15022289174332445,
                          0.15022289174332445,
                          0.8217986874983091,
                          0.9875344419529918,
                          0.9875344419529918,
                          0.8217986874983091,
                          0.8217986874983091,
                          0.012465558047008241,
                          0.012465558047008241,
                          0.8217986874983091,
                          0.17820131250169094,
                          0.9875344419529918,
                          0.9875344419529918,
                          0.17820131250169094,
                          0.17820131250169094,
                          0.012465558047008241,
                          0.012465558047008241,
                          0.17820131250169094,
                          0.6667699405823916,
                          0.932213804633531,
                          0.932213804633531,
                          0.6667699405823916,
                          0.6667699405823916,
                          0.06778619536646904,
                          0.06778619536646904,
                          0.6667699405823916,
                          0.33323005941760847,
                          0.932213804633531,
                          0.932213804633531,
                          0.33323005941760847,
                          0.33323005941760847,
                          0.06778619536646904,
                          0.06778619536646904,
                          0.33323005941760847};
      std::vector<T> w
          = {0.0749983360897274,   0.009532156337463184, 0.009532156337463184,
             0.009532156337463184, 0.009532156337463184, 0.046133867242451804,
             0.046133867242451804, 0.046133867242451804, 0.046133867242451804,
             0.009876785161863109, 0.009876785161863109, 0.009876785161863109,
             0.009876785161863109, 0.05784962985017135,  0.05784962985017135,
             0.05784962985017135,  0.05784962985017135,  0.034310524178250874,
             0.034310524178250874, 0.034310524178250874, 0.034310524178250874,
             0.008379945012595357, 0.008379945012595357, 0.008379945012595357,
             0.008379945012595357, 0.008379945012595357, 0.008379945012595357,
             0.008379945012595357, 0.008379945012595357, 0.02839378159108856,
             0.02839378159108856,  0.02839378159108856,  0.02839378159108856,
             0.02839378159108856,  0.02839378159108856,  0.02839378159108856,
             0.02839378159108856};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 15)
    {
      // Witherden-Vincent, 48 points, degree 15
      std::vector<T> x = {0.8990494939485035,
                          0.5,
                          0.5,
                          0.8990494939485035,
                          0.10095050605149646,
                          0.5,
                          0.5,
                          0.10095050605149646,
                          0.6519265131992299,
                          0.5,
                          0.5,
                          0.6519265131992299,
                          0.3480734868007701,
                          0.5,
                          0.5,
                          0.3480734868007701,
                          0.9412111350534427,
                          0.9412111350534427,
                          0.9412111350534427,
                          0.05878886494655732,
                          0.05878886494655732,
                          0.9412111350534427,
                          0.05878886494655732,
                          0.05878886494655732,
                          0.9888948997699514,
                          0.9888948997699514,
                          0.9888948997699514,
                          0.011105100230048652,
                          0.011105100230048652,
                          0.9888948997699514,
                          0.011105100230048652,
                          0.011105100230048652,
                          0.9043560679097518,
                          0.7836067866982953,
                          0.7836067866982953,
                          0.9043560679097518,
                          0.9043560679097518,
                          0.21639321330170466,
                          0.21639321330170466,
                          0.9043560679097518,
                          0.09564393209024821,
                          0.7836067866982953,
                          0.7836067866982953,
                          0.09564393209024821,
                          0.09564393209024821,
                          0.21639321330170466,
                          0.21639321330170466,
                          0.09564393209024821,
                          0.6523273995185263,
                          0.7893680970179032,
                          0.7893680970179032,
                          0.6523273995185263,
                          0.6523273995185263,
                          0.21063190298209675,
                          0.21063190298209675,
                          0.6523273995185263,
                          0.34767260048147375,
                          0.7893680970179032,
                          0.7893680970179032,
                          0.34767260048147375,
                          0.34767260048147375,
                          0.21063190298209675,
                          0.21063190298209675,
                          0.34767260048147375,
                          0.990252421862266,
                          0.8487318159954835,
                          0.8487318159954835,
                          0.990252421862266,
                          0.990252421862266,
                          0.1512681840045165,
                          0.1512681840045165,
                          0.990252421862266,
                          0.009747578137734018,
                          0.8487318159954835,
                          0.8487318159954835,
                          0.009747578137734018,
                          0.009747578137734018,
                          0.1512681840045165,
                          0.1512681840045165,
                          0.009747578137734018,
                          0.6324220779361581,
                          0.9778712599047559,
                          0.9778712599047559,
                          0.6324220779361581,
                          0.6324220779361581,
                          0.022128740095244147,
                          0.022128740095244147,
                          0.6324220779361581,
                          0.36757792206384193,
                          0.9778712599047559,
                          0.9778712599047559,
                          0.36757792206384193,
                          0.36757792206384193,
                          0.022128740095244147,
                          0.022128740095244147,
                          0.36757792206384193};
      std::vector<T> w = {
          0.028733231815881343,  0.028733231815881343,  0.028733231815881343,
          0.028733231815881343,  0.04542143974181796,   0.04542143974181796,
          0.04542143974181796,   0.04542143974181796,   0.010309620947191452,
          0.010309620947191452,  0.010309620947191452,  0.010309620947191452,
          0.0014834366214598054, 0.0014834366214598054, 0.0014834366214598054,
          0.0014834366214598054, 0.025287285869358106,  0.025287285869358106,
          0.025287285869358106,  0.025287285869358106,  0.025287285869358106,
          0.025287285869358106,  0.025287285869358106,  0.025287285869358106,
          0.036959180408220306,  0.036959180408220306,  0.036959180408220306,
          0.036959180408220306,  0.036959180408220306,  0.036959180408220306,
          0.036959180408220306,  0.036959180408220306,  0.005818298536168304,
          0.005818298536168304,  0.005818298536168304,  0.005818298536168304,
          0.005818298536168304,  0.005818298536168304,  0.005818298536168304,
          0.005818298536168304,  0.013961370623078005,  0.013961370623078005,
          0.013961370623078005,  0.013961370623078005,  0.013961370623078005,
          0.013961370623078005,  0.013961370623078005,  0.013961370623078005};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 17)
    {
      // Witherden-Vincent, 60 points, degree 17
      std::vector<T> x = {0.7725214936045742,
                          0.5,
                          0.5,
                          0.7725214936045742,
                          0.2274785063954258,
                          0.5,
                          0.5,
                          0.2274785063954258,
                          0.9587485941095766,
                          0.5,
                          0.5,
                          0.9587485941095766,
                          0.04125140589042339,
                          0.5,
                          0.5,
                          0.04125140589042339,
                          0.5962771213070235,
                          0.5962771213070235,
                          0.5962771213070235,
                          0.4037228786929766,
                          0.4037228786929766,
                          0.5962771213070235,
                          0.4037228786929766,
                          0.4037228786929766,
                          0.9356643423477217,
                          0.9356643423477217,
                          0.9356643423477217,
                          0.06433565765227836,
                          0.06433565765227836,
                          0.9356643423477217,
                          0.06433565765227836,
                          0.06433565765227836,
                          0.8471940450447754,
                          0.8471940450447754,
                          0.8471940450447754,
                          0.15280595495522462,
                          0.15280595495522462,
                          0.8471940450447754,
                          0.15280595495522462,
                          0.15280595495522462,
                          0.7288414964415778,
                          0.7288414964415778,
                          0.7288414964415778,
                          0.27115850355842225,
                          0.27115850355842225,
                          0.7288414964415778,
                          0.27115850355842225,
                          0.27115850355842225,
                          0.9846021280457544,
                          0.9846021280457544,
                          0.9846021280457544,
                          0.015397871954245633,
                          0.015397871954245633,
                          0.9846021280457544,
                          0.015397871954245633,
                          0.015397871954245633,
                          0.8810441380400992,
                          0.6397388833575961,
                          0.6397388833575961,
                          0.8810441380400992,
                          0.8810441380400992,
                          0.3602611166424039,
                          0.3602611166424039,
                          0.8810441380400992,
                          0.11895586195990086,
                          0.6397388833575961,
                          0.6397388833575961,
                          0.11895586195990086,
                          0.11895586195990086,
                          0.3602611166424039,
                          0.3602611166424039,
                          0.11895586195990086,
                          0.9536971760991119,
                          0.7734493352071551,
                          0.7734493352071551,
                          0.9536971760991119,
                          0.9536971760991119,
                          0.22655066479284494,
                          0.22655066479284494,
                          0.9536971760991119,
                          0.04630282390088816,
                          0.7734493352071551,
                          0.7734493352071551,
                          0.04630282390088816,
                          0.04630282390088816,
                          0.22655066479284494,
                          0.22655066479284494,
                          0.04630282390088816,
                          0.8806243332195414,
                          0.9918939857507748,
                          0.9918939857507748,
                          0.8806243332195414,
                          0.8806243332195414,
                          0.008106014249225224,
                          0.008106014249225224,
                          0.8806243332195414,
                          0.11937566678045862,
                          0.9918939857507748,
                          0.9918939857507748,
                          0.11937566678045862,
                          0.11937566678045862,
                          0.008106014249225224,
                          0.008106014249225224,
                          0.11937566678045862,
                          0.9935473535223839,
                          0.6514037408944306,
                          0.6514037408944306,
                          0.9935473535223839,
                          0.9935473535223839,
                          0.3485962591055693,
                          0.3485962591055693,
                          0.9935473535223839,
                          0.0064526464776160876,
                          0.6514037408944306,
                          0.6514037408944306,
                          0.0064526464776160876,
                          0.0064526464776160876,
                          0.3485962591055693,
                          0.3485962591055693,
                          0.0064526464776160876};
      std::vector<T> w = {
          0.03470109442180498,   0.03470109442180498,   0.03470109442180498,
          0.03470109442180498,   0.01633695595186623,   0.01633695595186623,
          0.01633695595186623,   0.01633695595186623,   0.035714104905549895,
          0.035714104905549895,  0.035714104905549895,  0.035714104905549895,
          0.009178977121312878,  0.009178977121312878,  0.009178977121312878,
          0.009178977121312878,  0.02156392490953595,   0.02156392490953595,
          0.02156392490953595,   0.02156392490953595,   0.03363906830297923,
          0.03363906830297923,   0.03363906830297923,   0.03363906830297923,
          0.0019139083537274841, 0.0019139083537274841, 0.0019139083537274841,
          0.0019139083537274841, 0.025794181196233838,  0.025794181196233838,
          0.025794181196233838,  0.025794181196233838,  0.025794181196233838,
          0.025794181196233838,  0.025794181196233838,  0.025794181196233838,
          0.01370660005932286,   0.01370660005932286,   0.01370660005932286,
          0.01370660005932286,   0.01370660005932286,   0.01370660005932286,
          0.01370660005932286,   0.01370660005932286,   0.0037799525970642154,
          0.0037799525970642154, 0.0037799525970642154, 0.0037799525970642154,
          0.0037799525970642154, 0.0037799525970642154, 0.0037799525970642154,
          0.0037799525970642154, 0.005195249163990761,  0.005195249163990761,
          0.005195249163990761,  0.005195249163990761,  0.005195249163990761,
          0.005195249163990761,  0.005195249163990761,  0.005195249163990761};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 19)
    {
      // Witherden-Vincent, 72 points, degree 19
      std::vector<T> x = {0.8571721297863971,
                          0.5,
                          0.5,
                          0.8571721297863971,
                          0.14282787021360288,
                          0.5,
                          0.5,
                          0.14282787021360288,
                          0.6328360260604818,
                          0.5,
                          0.5,
                          0.6328360260604818,
                          0.3671639739395181,
                          0.5,
                          0.5,
                          0.3671639739395181,
                          0.9822171346289836,
                          0.5,
                          0.5,
                          0.9822171346289836,
                          0.017782865371016376,
                          0.5,
                          0.5,
                          0.017782865371016376,
                          0.9016898647338425,
                          0.9016898647338425,
                          0.9016898647338425,
                          0.0983101352661575,
                          0.0983101352661575,
                          0.9016898647338425,
                          0.0983101352661575,
                          0.0983101352661575,
                          0.9960827029485674,
                          0.9960827029485674,
                          0.9960827029485674,
                          0.003917297051432639,
                          0.003917297051432639,
                          0.9960827029485674,
                          0.003917297051432639,
                          0.003917297051432639,
                          0.9647248013998047,
                          0.9647248013998047,
                          0.9647248013998047,
                          0.035275198600195304,
                          0.035275198600195304,
                          0.9647248013998047,
                          0.035275198600195304,
                          0.035275198600195304,
                          0.7551391286846717,
                          0.6333201572972811,
                          0.6333201572972811,
                          0.7551391286846717,
                          0.7551391286846717,
                          0.3666798427027189,
                          0.3666798427027189,
                          0.7551391286846717,
                          0.24486087131532833,
                          0.6333201572972811,
                          0.6333201572972811,
                          0.24486087131532833,
                          0.24486087131532833,
                          0.3666798427027189,
                          0.3666798427027189,
                          0.24486087131532833,
                          0.6953671028876248,
                          0.9939464665708676,
                          0.9939464665708676,
                          0.6953671028876248,
                          0.6953671028876248,
                          0.006053533429132345,
                          0.006053533429132345,
                          0.6953671028876248,
                          0.3046328971123751,
                          0.9939464665708676,
                          0.9939464665708676,
                          0.3046328971123751,
                          0.3046328971123751,
                          0.006053533429132345,
                          0.006053533429132345,
                          0.3046328971123751,
                          0.8585839606548726,
                          0.7562459386080488,
                          0.7562459386080488,
                          0.8585839606548726,
                          0.8585839606548726,
                          0.24375406139195116,
                          0.24375406139195116,
                          0.8585839606548726,
                          0.1414160393451274,
                          0.7562459386080488,
                          0.7562459386080488,
                          0.1414160393451274,
                          0.1414160393451274,
                          0.24375406139195116,
                          0.24375406139195116,
                          0.1414160393451274,
                          0.632720003905648,
                          0.9344512170772521,
                          0.9344512170772521,
                          0.632720003905648,
                          0.632720003905648,
                          0.06554878292274788,
                          0.06554878292274788,
                          0.632720003905648,
                          0.367279996094352,
                          0.9344512170772521,
                          0.9344512170772521,
                          0.367279996094352,
                          0.367279996094352,
                          0.06554878292274788,
                          0.06554878292274788,
                          0.367279996094352,
                          0.8100176993466282,
                          0.9631514779035646,
                          0.9631514779035646,
                          0.8100176993466282,
                          0.8100176993466282,
                          0.03684852209643538,
                          0.03684852209643538,
                          0.8100176993466282,
                          0.1899823006533718,
                          0.9631514779035646,
                          0.9631514779035646,
                          0.1899823006533718,
                          0.1899823006533718,
                          0.03684852209643538,
                          0.03684852209643538,
                          0.1899823006533718,
                          0.9008357923592984,
                          0.9942232653419869,
                          0.9942232653419869,
                          0.9008357923592984,
                          0.9008357923592984,
                          0.005776734658013124,
                          0.005776734658013124,
                          0.9008357923592984,
                          0.09916420764070158,
                          0.9942232653419869,
                          0.9942232653419869,
                          0.09916420764070158,
                          0.09916420764070158,
                          0.005776734658013124,
                          0.005776734658013124,
                          0.09916420764070158};
      std::vector<T> w = {
          0.024434341173207186,  0.024434341173207186,  0.024434341173207186,
          0.024434341173207186,  0.034826903230620564,  0.034826903230620564,
          0.034826903230620564,  0.034826903230620564,  0.008717395872971977,
          0.008717395872971977,  0.008717395872971977,  0.008717395872971977,
          0.011951136790698948,  0.011951136790698948,  0.011951136790698948,
          0.011951136790698948,  0.0004044287944779402, 0.0004044287944779402,
          0.0004044287944779402, 0.0004044287944779402, 0.0043602620085889405,
          0.0043602620085889405, 0.0043602620085889405, 0.0043602620085889405,
          0.02943145821001402,   0.02943145821001402,   0.02943145821001402,
          0.02943145821001402,   0.02943145821001402,   0.02943145821001402,
          0.02943145821001402,   0.02943145821001402,   0.0040448944029144625,
          0.0040448944029144625, 0.0040448944029144625, 0.0040448944029144625,
          0.0040448944029144625, 0.0040448944029144625, 0.0040448944029144625,
          0.0040448944029144625, 0.020711654745851224,  0.020711654745851224,
          0.020711654745851224,  0.020711654745851224,  0.020711654745851224,
          0.020711654745851224,  0.020711654745851224,  0.020711654745851224,
          0.016247270373149013,  0.016247270373149013,  0.016247270373149013,
          0.016247270373149013,  0.016247270373149013,  0.016247270373149013,
          0.016247270373149013,  0.016247270373149013,  0.009745138099102634,
          0.009745138099102634,  0.009745138099102634,  0.009745138099102634,
          0.009745138099102634,  0.009745138099102634,  0.009745138099102634,
          0.009745138099102634,  0.002472350233685871,  0.002472350233685871,
          0.002472350233685871,  0.002472350233685871,  0.002472350233685871,
          0.002472350233685871,  0.002472350233685871,  0.002472350233685871};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 21)
    {
      // Witherden-Vincent, 85 points, degree 21
      std::vector<T> x = {0.5,
                          0.5,
                          0.736675537179112,
                          0.5,
                          0.5,
                          0.736675537179112,
                          0.26332446282088795,
                          0.5,
                          0.5,
                          0.26332446282088795,
                          0.9176392148779342,
                          0.5,
                          0.5,
                          0.9176392148779342,
                          0.08236078512206585,
                          0.5,
                          0.5,
                          0.08236078512206585,
                          0.6286859503036145,
                          0.6286859503036145,
                          0.6286859503036145,
                          0.3713140496963855,
                          0.3713140496963855,
                          0.6286859503036145,
                          0.3713140496963855,
                          0.3713140496963855,
                          0.9816689160578117,
                          0.9816689160578117,
                          0.9816689160578117,
                          0.018331083942188322,
                          0.018331083942188322,
                          0.9816689160578117,
                          0.018331083942188322,
                          0.018331083942188322,
                          0.9312259626898258,
                          0.9312259626898258,
                          0.9312259626898258,
                          0.06877403731017423,
                          0.06877403731017423,
                          0.9312259626898258,
                          0.06877403731017423,
                          0.06877403731017423,
                          0.7484489812596729,
                          0.7484489812596729,
                          0.7484489812596729,
                          0.25155101874032715,
                          0.25155101874032715,
                          0.7484489812596729,
                          0.25155101874032715,
                          0.25155101874032715,
                          0.8521660875977003,
                          0.8521660875977003,
                          0.8521660875977003,
                          0.1478339124022997,
                          0.1478339124022997,
                          0.8521660875977003,
                          0.1478339124022997,
                          0.1478339124022997,
                          0.620939092738351,
                          0.8370731099776588,
                          0.8370731099776588,
                          0.620939092738351,
                          0.620939092738351,
                          0.16292689002234112,
                          0.16292689002234112,
                          0.620939092738351,
                          0.37906090726164904,
                          0.8370731099776588,
                          0.8370731099776588,
                          0.37906090726164904,
                          0.37906090726164904,
                          0.16292689002234112,
                          0.16292689002234112,
                          0.37906090726164904,
                          0.7400784831563976,
                          0.9123236876354603,
                          0.9123236876354603,
                          0.7400784831563976,
                          0.7400784831563976,
                          0.08767631236453963,
                          0.08767631236453963,
                          0.7400784831563976,
                          0.25992151684360243,
                          0.9123236876354603,
                          0.9123236876354603,
                          0.25992151684360243,
                          0.25992151684360243,
                          0.08767631236453963,
                          0.08767631236453963,
                          0.25992151684360243,
                          0.966120967960877,
                          0.6353495920508324,
                          0.6353495920508324,
                          0.966120967960877,
                          0.966120967960877,
                          0.36465040794916753,
                          0.36465040794916753,
                          0.966120967960877,
                          0.03387903203912302,
                          0.6353495920508324,
                          0.6353495920508324,
                          0.03387903203912302,
                          0.03387903203912302,
                          0.36465040794916753,
                          0.36465040794916753,
                          0.03387903203912302,
                          0.9674511629120053,
                          0.8375197837185376,
                          0.8375197837185376,
                          0.9674511629120053,
                          0.9674511629120053,
                          0.16248021628146236,
                          0.16248021628146236,
                          0.9674511629120053,
                          0.03254883708799473,
                          0.8375197837185376,
                          0.8375197837185376,
                          0.03254883708799473,
                          0.03254883708799473,
                          0.16248021628146236,
                          0.16248021628146236,
                          0.03254883708799473,
                          0.995227733764762,
                          0.7444581255885835,
                          0.7444581255885835,
                          0.995227733764762,
                          0.995227733764762,
                          0.25554187441141657,
                          0.25554187441141657,
                          0.995227733764762,
                          0.004772266235237924,
                          0.7444581255885835,
                          0.7444581255885835,
                          0.004772266235237924,
                          0.004772266235237924,
                          0.25554187441141657,
                          0.25554187441141657,
                          0.004772266235237924,
                          0.9155676229879507,
                          0.9944803056932862,
                          0.9944803056932862,
                          0.9155676229879507,
                          0.9155676229879507,
                          0.005519694306713804,
                          0.005519694306713804,
                          0.9155676229879507,
                          0.08443237701204931,
                          0.9944803056932862,
                          0.9944803056932862,
                          0.08443237701204931,
                          0.08443237701204931,
                          0.005519694306713804,
                          0.005519694306713804,
                          0.08443237701204931,
                          0.5457348004611456,
                          0.9920085742447995,
                          0.9920085742447995,
                          0.5457348004611456,
                          0.5457348004611456,
                          0.007991425755200536,
                          0.007991425755200536,
                          0.5457348004611456,
                          0.45426519953885436,
                          0.9920085742447995,
                          0.9920085742447995,
                          0.45426519953885436,
                          0.45426519953885436,
                          0.007991425755200536,
                          0.007991425755200536,
                          0.45426519953885436};
      std::vector<T> w = {
          0.033798210089035705,  0.026698539696485096,  0.026698539696485096,
          0.026698539696485096,  0.026698539696485096,  0.016564302001236098,
          0.016564302001236098,  0.016564302001236098,  0.016564302001236098,
          0.029971120911585117,  0.029971120911585117,  0.029971120911585117,
          0.029971120911585117,  0.0020573653223050024, 0.0020573653223050024,
          0.0020573653223050024, 0.0020573653223050024, 0.007650910232193913,
          0.007650910232193913,  0.007650910232193913,  0.007650910232193913,
          0.024197947973398803,  0.024197947973398803,  0.024197947973398803,
          0.024197947973398803,  0.01537908537032914,   0.01537908537032914,
          0.01537908537032914,   0.01537908537032914,   0.021682124878302057,
          0.021682124878302057,  0.021682124878302057,  0.021682124878302057,
          0.021682124878302057,  0.021682124878302057,  0.021682124878302057,
          0.021682124878302057,  0.013969779351031837,  0.013969779351031837,
          0.013969779351031837,  0.013969779351031837,  0.013969779351031837,
          0.013969779351031837,  0.013969779351031837,  0.013969779351031837,
          0.009282206860228454,  0.009282206860228454,  0.009282206860228454,
          0.009282206860228454,  0.009282206860228454,  0.009282206860228454,
          0.009282206860228454,  0.009282206860228454,  0.0071939002127551326,
          0.0071939002127551326, 0.0071939002127551326, 0.0071939002127551326,
          0.0071939002127551326, 0.0071939002127551326, 0.0071939002127551326,
          0.0071939002127551326, 0.002877380110623292,  0.002877380110623292,
          0.002877380110623292,  0.002877380110623292,  0.002877380110623292,
          0.002877380110623292,  0.002877380110623292,  0.002877380110623292,
          0.0018821205326004114, 0.0018821205326004114, 0.0018821205326004114,
          0.0018821205326004114, 0.0018821205326004114, 0.0018821205326004114,
          0.0018821205326004114, 0.0018821205326004114, 0.0026280760395627693,
          0.0026280760395627693, 0.0026280760395627693, 0.0026280760395627693,
          0.0026280760395627693, 0.0026280760395627693, 0.0026280760395627693,
          0.0026280760395627693};
      return {std::move(x), std::move(w)};
    }
    else
    {
      throw std::runtime_error(
          "Witherden-Vincent not implemented for this order.");
    }
  }
  case cell::type::tetrahedron:
  {
    if (m <= 1)
    {
      // Witherden-Vincent, 1 point, degree 1
      std::vector<T> x = {0.25, 0.25, 0.25};
      std::vector<T> w = {0.16666666666666666};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 2)
    {
      // Witherden-Vincent, 4 points, degree 2
      std::vector<T> x
          = {0.1381966011250105, 0.1381966011250105, 0.5854101966249684,
             0.1381966011250105, 0.5854101966249684, 0.1381966011250105,
             0.5854101966249684, 0.1381966011250105, 0.1381966011250105,
             0.1381966011250105, 0.1381966011250105, 0.1381966011250105};
      std::vector<T> w = {0.041666666666666664, 0.041666666666666664,
                          0.041666666666666664, 0.041666666666666664};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 3)
    {
      // Witherden-Vincent, 8 points, degree 3
      std::vector<T> x
          = {0.3281633025163817,  0.3281633025163817,  0.01551009245085494,
             0.3281633025163817,  0.01551009245085494, 0.3281633025163817,
             0.01551009245085494, 0.3281633025163817,  0.3281633025163817,
             0.3281633025163817,  0.3281633025163817,  0.3281633025163817,
             0.10804724989842861, 0.10804724989842861, 0.6758582503047141,
             0.10804724989842861, 0.6758582503047141,  0.10804724989842861,
             0.6758582503047141,  0.10804724989842861, 0.10804724989842861,
             0.10804724989842861, 0.10804724989842861, 0.10804724989842861};
      std::vector<T> w
          = {0.022702973756181227, 0.022702973756181227, 0.022702973756181227,
             0.022702973756181227, 0.01896369291048544,  0.01896369291048544,
             0.01896369291048544,  0.01896369291048544};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 5)
    {
      // Witherden-Vincent, 14 points, degree 5
      std::vector<T> x
          = {0.3108859192633006,  0.3108859192633006,  0.06734224221009817,
             0.3108859192633006,  0.06734224221009817, 0.3108859192633006,
             0.06734224221009817, 0.3108859192633006,  0.3108859192633006,
             0.3108859192633006,  0.3108859192633006,  0.3108859192633006,
             0.09273525031089122, 0.09273525031089122, 0.7217942490673264,
             0.09273525031089122, 0.7217942490673264,  0.09273525031089122,
             0.7217942490673264,  0.09273525031089122, 0.09273525031089122,
             0.09273525031089122, 0.09273525031089122, 0.09273525031089122,
             0.04550370412564965, 0.45449629587435036, 0.45449629587435036,
             0.45449629587435036, 0.04550370412564965, 0.45449629587435036,
             0.04550370412564965, 0.04550370412564965, 0.45449629587435036,
             0.04550370412564965, 0.45449629587435036, 0.04550370412564965,
             0.45449629587435036, 0.04550370412564965, 0.04550370412564965,
             0.45449629587435036, 0.45449629587435036, 0.04550370412564965};
      std::vector<T> w
          = {0.018781320953002643, 0.018781320953002643, 0.018781320953002643,
             0.018781320953002643, 0.012248840519393659, 0.012248840519393659,
             0.012248840519393659, 0.012248840519393659, 0.007091003462846911,
             0.007091003462846911, 0.007091003462846911, 0.007091003462846911,
             0.007091003462846911, 0.007091003462846911};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 6)
    {
      // Witherden-Vincent, 24 points, degree 6
      std::vector<T> x
          = {0.04067395853461135, 0.04067395853461135, 0.877978124396166,
             0.04067395853461135, 0.877978124396166,   0.04067395853461135,
             0.877978124396166,   0.04067395853461135, 0.04067395853461135,
             0.04067395853461135, 0.04067395853461135, 0.04067395853461135,
             0.3223378901422755,  0.3223378901422755,  0.03298632957317347,
             0.3223378901422755,  0.03298632957317347, 0.3223378901422755,
             0.03298632957317347, 0.3223378901422755,  0.3223378901422755,
             0.3223378901422755,  0.3223378901422755,  0.3223378901422755,
             0.21460287125915203, 0.21460287125915203, 0.3561913862225439,
             0.21460287125915203, 0.3561913862225439,  0.21460287125915203,
             0.3561913862225439,  0.21460287125915203, 0.21460287125915203,
             0.21460287125915203, 0.21460287125915203, 0.21460287125915203,
             0.6030056647916492,  0.06366100187501753, 0.2696723314583158,
             0.6030056647916492,  0.06366100187501753, 0.06366100187501753,
             0.06366100187501753, 0.06366100187501753, 0.6030056647916492,
             0.2696723314583158,  0.6030056647916492,  0.06366100187501753,
             0.06366100187501753, 0.2696723314583158,  0.6030056647916492,
             0.06366100187501753, 0.6030056647916492,  0.06366100187501753,
             0.2696723314583158,  0.06366100187501753, 0.6030056647916492,
             0.06366100187501753, 0.2696723314583158,  0.06366100187501753,
             0.06366100187501753, 0.06366100187501753, 0.2696723314583158,
             0.06366100187501753, 0.6030056647916492,  0.2696723314583158,
             0.2696723314583158,  0.06366100187501753, 0.06366100187501753,
             0.6030056647916492,  0.2696723314583158,  0.06366100187501753};
      std::vector<T> w
          = {0.001679535175886774, 0.001679535175886774, 0.001679535175886774,
             0.001679535175886774, 0.009226196923942455, 0.009226196923942455,
             0.009226196923942455, 0.009226196923942455, 0.006653791709694582,
             0.006653791709694582, 0.006653791709694582, 0.006653791709694582,
             0.008035714285714285, 0.008035714285714285, 0.008035714285714285,
             0.008035714285714285, 0.008035714285714285, 0.008035714285714285,
             0.008035714285714285, 0.008035714285714285, 0.008035714285714285,
             0.008035714285714285, 0.008035714285714285, 0.008035714285714285};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 7)
    {
      // Witherden-Vincent, 35 points, degree 7
      std::vector<T> x = {0.25,
                          0.25,
                          0.25,
                          0.3157011497782028,
                          0.3157011497782028,
                          0.0528965506653916,
                          0.3157011497782028,
                          0.0528965506653916,
                          0.3157011497782028,
                          0.0528965506653916,
                          0.3157011497782028,
                          0.3157011497782028,
                          0.3157011497782028,
                          0.3157011497782028,
                          0.3157011497782028,
                          0.05048982259839637,
                          0.44951017740160365,
                          0.44951017740160365,
                          0.44951017740160365,
                          0.05048982259839637,
                          0.44951017740160365,
                          0.05048982259839637,
                          0.05048982259839637,
                          0.44951017740160365,
                          0.05048982259839637,
                          0.44951017740160365,
                          0.05048982259839637,
                          0.44951017740160365,
                          0.05048982259839637,
                          0.05048982259839637,
                          0.44951017740160365,
                          0.44951017740160365,
                          0.05048982259839637,
                          0.5751716375870001,
                          0.18883383102600104,
                          0.047160700360997884,
                          0.5751716375870001,
                          0.18883383102600104,
                          0.18883383102600104,
                          0.18883383102600104,
                          0.18883383102600104,
                          0.5751716375870001,
                          0.047160700360997884,
                          0.5751716375870001,
                          0.18883383102600104,
                          0.18883383102600104,
                          0.047160700360997884,
                          0.5751716375870001,
                          0.18883383102600104,
                          0.5751716375870001,
                          0.18883383102600104,
                          0.047160700360997884,
                          0.18883383102600104,
                          0.5751716375870001,
                          0.18883383102600104,
                          0.047160700360997884,
                          0.18883383102600104,
                          0.18883383102600104,
                          0.18883383102600104,
                          0.047160700360997884,
                          0.18883383102600104,
                          0.5751716375870001,
                          0.047160700360997884,
                          0.047160700360997884,
                          0.18883383102600104,
                          0.18883383102600104,
                          0.5751716375870001,
                          0.047160700360997884,
                          0.18883383102600104,
                          0.8108302410985485,
                          0.021265472541483248,
                          0.14663881381848495,
                          0.8108302410985485,
                          0.021265472541483248,
                          0.021265472541483248,
                          0.021265472541483248,
                          0.021265472541483248,
                          0.8108302410985485,
                          0.14663881381848495,
                          0.8108302410985485,
                          0.021265472541483248,
                          0.021265472541483248,
                          0.14663881381848495,
                          0.8108302410985485,
                          0.021265472541483248,
                          0.8108302410985485,
                          0.021265472541483248,
                          0.14663881381848495,
                          0.021265472541483248,
                          0.8108302410985485,
                          0.021265472541483248,
                          0.14663881381848495,
                          0.021265472541483248,
                          0.021265472541483248,
                          0.021265472541483248,
                          0.14663881381848495,
                          0.021265472541483248,
                          0.8108302410985485,
                          0.14663881381848495,
                          0.14663881381848495,
                          0.021265472541483248,
                          0.021265472541483248,
                          0.8108302410985485,
                          0.14663881381848495,
                          0.021265472541483248};
      std::vector<T> w = {
          0.015914214910688475,  0.007054930201661171,  0.007054930201661171,
          0.007054930201661171,  0.007054930201661171,  0.005316154638809596,
          0.005316154638809596,  0.005316154638809596,  0.005316154638809596,
          0.005316154638809596,  0.005316154638809596,  0.006201188454722437,
          0.006201188454722437,  0.006201188454722437,  0.006201188454722437,
          0.006201188454722437,  0.006201188454722437,  0.006201188454722437,
          0.006201188454722437,  0.006201188454722437,  0.006201188454722437,
          0.006201188454722437,  0.006201188454722437,  0.0013517951383172236,
          0.0013517951383172236, 0.0013517951383172236, 0.0013517951383172236,
          0.0013517951383172236, 0.0013517951383172236, 0.0013517951383172236,
          0.0013517951383172236, 0.0013517951383172236, 0.0013517951383172236,
          0.0013517951383172236, 0.0013517951383172236};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 8)
    {
      // Witherden-Vincent, 46 points, degree 8
      std::vector<T> x
          = {0.10795272496221085,  0.10795272496221085,  0.6761418251133674,
             0.10795272496221085,  0.6761418251133674,   0.10795272496221085,
             0.6761418251133674,   0.10795272496221085,  0.10795272496221085,
             0.10795272496221085,  0.10795272496221085,  0.10795272496221085,
             0.18510948778258657,  0.18510948778258657,  0.4446715366522403,
             0.18510948778258657,  0.4446715366522403,   0.18510948778258657,
             0.4446715366522403,   0.18510948778258657,  0.18510948778258657,
             0.18510948778258657,  0.18510948778258657,  0.18510948778258657,
             0.04231654368476728,  0.04231654368476728,  0.8730503689456982,
             0.04231654368476728,  0.8730503689456982,   0.04231654368476728,
             0.8730503689456982,   0.04231654368476728,  0.04231654368476728,
             0.04231654368476728,  0.04231654368476728,  0.04231654368476728,
             0.314181709124039,    0.314181709124039,    0.057454872627883014,
             0.314181709124039,    0.057454872627883014, 0.314181709124039,
             0.057454872627883014, 0.314181709124039,    0.314181709124039,
             0.314181709124039,    0.314181709124039,    0.314181709124039,
             0.4355913285838302,   0.0644086714161698,   0.0644086714161698,
             0.0644086714161698,   0.4355913285838302,   0.0644086714161698,
             0.4355913285838302,   0.4355913285838302,   0.0644086714161698,
             0.4355913285838302,   0.0644086714161698,   0.4355913285838302,
             0.0644086714161698,   0.4355913285838302,   0.4355913285838302,
             0.0644086714161698,   0.0644086714161698,   0.4355913285838302,
             0.7174640634263083,   0.021433930127130574, 0.23966807631943052,
             0.7174640634263083,   0.021433930127130574, 0.021433930127130574,
             0.021433930127130574, 0.021433930127130574, 0.7174640634263083,
             0.23966807631943052,  0.7174640634263083,   0.021433930127130574,
             0.021433930127130574, 0.23966807631943052,  0.7174640634263083,
             0.021433930127130574, 0.7174640634263083,   0.021433930127130574,
             0.23966807631943052,  0.021433930127130574, 0.7174640634263083,
             0.021433930127130574, 0.23966807631943052,  0.021433930127130574,
             0.021433930127130574, 0.021433930127130574, 0.23966807631943052,
             0.021433930127130574, 0.7174640634263083,   0.23966807631943052,
             0.23966807631943052,  0.021433930127130574, 0.021433930127130574,
             0.7174640634263083,   0.23966807631943052,  0.021433930127130574,
             0.5837973783021444,   0.20413933387602912,  0.007923953945797353,
             0.5837973783021444,   0.20413933387602912,  0.20413933387602912,
             0.20413933387602912,  0.20413933387602912,  0.5837973783021444,
             0.007923953945797353, 0.5837973783021444,   0.20413933387602912,
             0.20413933387602912,  0.007923953945797353, 0.5837973783021444,
             0.20413933387602912,  0.5837973783021444,   0.20413933387602912,
             0.007923953945797353, 0.20413933387602912,  0.5837973783021444,
             0.20413933387602912,  0.007923953945797353, 0.20413933387602912,
             0.20413933387602912,  0.20413933387602912,  0.007923953945797353,
             0.20413933387602912,  0.5837973783021444,   0.007923953945797353,
             0.007923953945797353, 0.20413933387602912,  0.20413933387602912,
             0.5837973783021444,   0.007923953945797353, 0.20413933387602912};
      std::vector<T> w = {
          0.004404441818068139,  0.004404441818068139,  0.004404441818068139,
          0.004404441818068139,  0.008671957927289755,  0.008671957927289755,
          0.008671957927289755,  0.008671957927289755,  0.0012542093589233666,
          0.0012542093589233666, 0.0012542093589233666, 0.0012542093589233666,
          0.006960630476155816,  0.006960630476155816,  0.006960630476155816,
          0.006960630476155816,  0.006046821710218137,  0.006046821710218137,
          0.006046821710218137,  0.006046821710218137,  0.006046821710218137,
          0.006046821710218137,  0.0011928171484740721, 0.0011928171484740721,
          0.0011928171484740721, 0.0011928171484740721, 0.0011928171484740721,
          0.0011928171484740721, 0.0011928171484740721, 0.0011928171484740721,
          0.0011928171484740721, 0.0011928171484740721, 0.0011928171484740721,
          0.0011928171484740721, 0.002575581025160056,  0.002575581025160056,
          0.002575581025160056,  0.002575581025160056,  0.002575581025160056,
          0.002575581025160056,  0.002575581025160056,  0.002575581025160056,
          0.002575581025160056,  0.002575581025160056,  0.002575581025160056,
          0.002575581025160056};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 9)
    {
      // Witherden-Vincent, 59 points, degree 9
      std::vector<T> x = {0.25,
                          0.25,
                          0.25,
                          6.198169944546509e-10,
                          6.198169944546509e-10,
                          0.999999998140549,
                          6.198169944546509e-10,
                          0.999999998140549,
                          6.198169944546509e-10,
                          0.999999998140549,
                          6.198169944546509e-10,
                          6.198169944546509e-10,
                          6.198169944546509e-10,
                          6.198169944546509e-10,
                          6.198169944546509e-10,
                          0.1607745353952616,
                          0.1607745353952616,
                          0.5176763938142153,
                          0.1607745353952616,
                          0.5176763938142153,
                          0.1607745353952616,
                          0.5176763938142153,
                          0.1607745353952616,
                          0.1607745353952616,
                          0.1607745353952616,
                          0.1607745353952616,
                          0.1607745353952616,
                          0.32227652182142097,
                          0.32227652182142097,
                          0.03317043453573708,
                          0.32227652182142097,
                          0.03317043453573708,
                          0.32227652182142097,
                          0.03317043453573708,
                          0.32227652182142097,
                          0.32227652182142097,
                          0.32227652182142097,
                          0.32227652182142097,
                          0.32227652182142097,
                          0.045108918345413584,
                          0.045108918345413584,
                          0.8646732449637593,
                          0.045108918345413584,
                          0.8646732449637593,
                          0.045108918345413584,
                          0.8646732449637593,
                          0.045108918345413584,
                          0.045108918345413584,
                          0.045108918345413584,
                          0.045108918345413584,
                          0.045108918345413584,
                          0.11229654600437605,
                          0.38770345399562395,
                          0.38770345399562395,
                          0.38770345399562395,
                          0.11229654600437605,
                          0.38770345399562395,
                          0.11229654600437605,
                          0.11229654600437605,
                          0.38770345399562395,
                          0.11229654600437605,
                          0.38770345399562395,
                          0.11229654600437605,
                          0.38770345399562395,
                          0.11229654600437605,
                          0.11229654600437605,
                          0.38770345399562395,
                          0.38770345399562395,
                          0.11229654600437605,
                          0.002554579233041331,
                          0.4588714487524593,
                          0.07970252326204012,
                          0.002554579233041331,
                          0.4588714487524593,
                          0.4588714487524593,
                          0.4588714487524593,
                          0.4588714487524593,
                          0.002554579233041331,
                          0.07970252326204012,
                          0.002554579233041331,
                          0.4588714487524593,
                          0.4588714487524593,
                          0.07970252326204012,
                          0.002554579233041331,
                          0.4588714487524593,
                          0.002554579233041331,
                          0.4588714487524593,
                          0.07970252326204012,
                          0.4588714487524593,
                          0.002554579233041331,
                          0.4588714487524593,
                          0.07970252326204012,
                          0.4588714487524593,
                          0.4588714487524593,
                          0.4588714487524593,
                          0.07970252326204012,
                          0.4588714487524593,
                          0.002554579233041331,
                          0.07970252326204012,
                          0.07970252326204012,
                          0.4588714487524593,
                          0.4588714487524593,
                          0.002554579233041331,
                          0.07970252326204012,
                          0.4588714487524593,
                          0.7183503264420745,
                          0.03377587068533858,
                          0.21409793218724835,
                          0.7183503264420745,
                          0.03377587068533858,
                          0.03377587068533858,
                          0.03377587068533858,
                          0.03377587068533858,
                          0.7183503264420745,
                          0.21409793218724835,
                          0.7183503264420745,
                          0.03377587068533858,
                          0.03377587068533858,
                          0.21409793218724835,
                          0.7183503264420745,
                          0.03377587068533858,
                          0.7183503264420745,
                          0.03377587068533858,
                          0.21409793218724835,
                          0.03377587068533858,
                          0.7183503264420745,
                          0.03377587068533858,
                          0.21409793218724835,
                          0.03377587068533858,
                          0.03377587068533858,
                          0.03377587068533858,
                          0.21409793218724835,
                          0.03377587068533858,
                          0.7183503264420745,
                          0.21409793218724835,
                          0.21409793218724835,
                          0.03377587068533858,
                          0.03377587068533858,
                          0.7183503264420745,
                          0.21409793218724835,
                          0.03377587068533858,
                          0.03441591057817527,
                          0.1836413698099279,
                          0.5983013498019689,
                          0.03441591057817527,
                          0.1836413698099279,
                          0.1836413698099279,
                          0.1836413698099279,
                          0.1836413698099279,
                          0.03441591057817527,
                          0.5983013498019689,
                          0.03441591057817527,
                          0.1836413698099279,
                          0.1836413698099279,
                          0.5983013498019689,
                          0.03441591057817527,
                          0.1836413698099279,
                          0.03441591057817527,
                          0.1836413698099279,
                          0.5983013498019689,
                          0.1836413698099279,
                          0.03441591057817527,
                          0.1836413698099279,
                          0.5983013498019689,
                          0.1836413698099279,
                          0.1836413698099279,
                          0.1836413698099279,
                          0.5983013498019689,
                          0.1836413698099279,
                          0.03441591057817527,
                          0.5983013498019689,
                          0.5983013498019689,
                          0.1836413698099279,
                          0.1836413698099279,
                          0.03441591057817527,
                          0.5983013498019689,
                          0.1836413698099279};
      std::vector<T> w = {0.00966842481874671,    1.0719880293209398e-05,
                          1.0719880293209398e-05, 1.0719880293209398e-05,
                          1.0719880293209398e-05, 0.0038622230770709097,
                          0.0038622230770709097,  0.0038622230770709097,
                          0.0038622230770709097,  0.004927152055904881,
                          0.004927152055904881,   0.004927152055904881,
                          0.004927152055904881,   0.0013439966632693638,
                          0.0013439966632693638,  0.0013439966632693638,
                          0.0013439966632693638,  0.006355680017283745,
                          0.006355680017283745,   0.006355680017283745,
                          0.006355680017283745,   0.006355680017283745,
                          0.006355680017283745,   0.0013974036997164254,
                          0.0013974036997164254,  0.0013974036997164254,
                          0.0013974036997164254,  0.0013974036997164254,
                          0.0013974036997164254,  0.0013974036997164254,
                          0.0013974036997164254,  0.0013974036997164254,
                          0.0013974036997164254,  0.0013974036997164254,
                          0.0013974036997164254,  0.0017057598921242213,
                          0.0017057598921242213,  0.0017057598921242213,
                          0.0017057598921242213,  0.0017057598921242213,
                          0.0017057598921242213,  0.0017057598921242213,
                          0.0017057598921242213,  0.0017057598921242213,
                          0.0017057598921242213,  0.0017057598921242213,
                          0.0017057598921242213,  0.003420819327998023,
                          0.003420819327998023,   0.003420819327998023,
                          0.003420819327998023,   0.003420819327998023,
                          0.003420819327998023,   0.003420819327998023,
                          0.003420819327998023,   0.003420819327998023,
                          0.003420819327998023,   0.003420819327998023,
                          0.003420819327998023};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 10)
    {
      // Witherden-Vincent, 81 points, degree 10
      std::vector<T> x = {0.25,
                          0.25,
                          0.25,
                          0.3122500686951886,
                          0.3122500686951886,
                          0.06324979391443405,
                          0.3122500686951886,
                          0.06324979391443405,
                          0.3122500686951886,
                          0.06324979391443405,
                          0.3122500686951886,
                          0.3122500686951886,
                          0.3122500686951886,
                          0.3122500686951886,
                          0.3122500686951886,
                          0.11430965385734615,
                          0.11430965385734615,
                          0.6570710384279616,
                          0.11430965385734615,
                          0.6570710384279616,
                          0.11430965385734615,
                          0.6570710384279616,
                          0.11430965385734615,
                          0.11430965385734615,
                          0.11430965385734615,
                          0.11430965385734615,
                          0.11430965385734615,
                          0.16548602561961104,
                          0.4104307392189655,
                          0.01365249594245796,
                          0.16548602561961104,
                          0.4104307392189655,
                          0.4104307392189655,
                          0.4104307392189655,
                          0.4104307392189655,
                          0.16548602561961104,
                          0.01365249594245796,
                          0.16548602561961104,
                          0.4104307392189655,
                          0.4104307392189655,
                          0.01365249594245796,
                          0.16548602561961104,
                          0.4104307392189655,
                          0.16548602561961104,
                          0.4104307392189655,
                          0.01365249594245796,
                          0.4104307392189655,
                          0.16548602561961104,
                          0.4104307392189655,
                          0.01365249594245796,
                          0.4104307392189655,
                          0.4104307392189655,
                          0.4104307392189655,
                          0.01365249594245796,
                          0.4104307392189655,
                          0.16548602561961104,
                          0.01365249594245796,
                          0.01365249594245796,
                          0.4104307392189655,
                          0.4104307392189655,
                          0.16548602561961104,
                          0.01365249594245796,
                          0.4104307392189655,
                          0.9429887673452049,
                          0.006138008824790748,
                          0.04473521500521364,
                          0.9429887673452049,
                          0.006138008824790748,
                          0.006138008824790748,
                          0.006138008824790748,
                          0.006138008824790748,
                          0.9429887673452049,
                          0.04473521500521364,
                          0.9429887673452049,
                          0.006138008824790748,
                          0.006138008824790748,
                          0.04473521500521364,
                          0.9429887673452049,
                          0.006138008824790748,
                          0.9429887673452049,
                          0.006138008824790748,
                          0.04473521500521364,
                          0.006138008824790748,
                          0.9429887673452049,
                          0.006138008824790748,
                          0.04473521500521364,
                          0.006138008824790748,
                          0.006138008824790748,
                          0.006138008824790748,
                          0.04473521500521364,
                          0.006138008824790748,
                          0.9429887673452049,
                          0.04473521500521364,
                          0.04473521500521364,
                          0.006138008824790748,
                          0.006138008824790748,
                          0.9429887673452049,
                          0.04473521500521364,
                          0.006138008824790748,
                          0.47719037990428037,
                          0.12105018114558942,
                          0.2807092578045408,
                          0.47719037990428037,
                          0.12105018114558942,
                          0.12105018114558942,
                          0.12105018114558942,
                          0.12105018114558942,
                          0.47719037990428037,
                          0.2807092578045408,
                          0.47719037990428037,
                          0.12105018114558942,
                          0.12105018114558942,
                          0.2807092578045408,
                          0.47719037990428037,
                          0.12105018114558942,
                          0.47719037990428037,
                          0.12105018114558942,
                          0.2807092578045408,
                          0.12105018114558942,
                          0.47719037990428037,
                          0.12105018114558942,
                          0.2807092578045408,
                          0.12105018114558942,
                          0.12105018114558942,
                          0.12105018114558942,
                          0.2807092578045408,
                          0.12105018114558942,
                          0.47719037990428037,
                          0.2807092578045408,
                          0.2807092578045408,
                          0.12105018114558942,
                          0.12105018114558942,
                          0.47719037990428037,
                          0.2807092578045408,
                          0.12105018114558942,
                          0.594256269480007,
                          0.03277946821644267,
                          0.3401847940871076,
                          0.594256269480007,
                          0.03277946821644267,
                          0.03277946821644267,
                          0.03277946821644267,
                          0.03277946821644267,
                          0.594256269480007,
                          0.3401847940871076,
                          0.594256269480007,
                          0.03277946821644267,
                          0.03277946821644267,
                          0.3401847940871076,
                          0.594256269480007,
                          0.03277946821644267,
                          0.594256269480007,
                          0.03277946821644267,
                          0.3401847940871076,
                          0.03277946821644267,
                          0.594256269480007,
                          0.03277946821644267,
                          0.3401847940871076,
                          0.03277946821644267,
                          0.03277946821644267,
                          0.03277946821644267,
                          0.3401847940871076,
                          0.03277946821644267,
                          0.594256269480007,
                          0.3401847940871076,
                          0.3401847940871076,
                          0.03277946821644267,
                          0.03277946821644267,
                          0.594256269480007,
                          0.3401847940871076,
                          0.03277946821644267,
                          0.8011772846583444,
                          0.03248528156482305,
                          0.13385215221200952,
                          0.8011772846583444,
                          0.03248528156482305,
                          0.03248528156482305,
                          0.03248528156482305,
                          0.03248528156482305,
                          0.8011772846583444,
                          0.13385215221200952,
                          0.8011772846583444,
                          0.03248528156482305,
                          0.03248528156482305,
                          0.13385215221200952,
                          0.8011772846583444,
                          0.03248528156482305,
                          0.8011772846583444,
                          0.03248528156482305,
                          0.13385215221200952,
                          0.03248528156482305,
                          0.8011772846583444,
                          0.03248528156482305,
                          0.13385215221200952,
                          0.03248528156482305,
                          0.03248528156482305,
                          0.03248528156482305,
                          0.13385215221200952,
                          0.03248528156482305,
                          0.8011772846583444,
                          0.13385215221200952,
                          0.13385215221200952,
                          0.03248528156482305,
                          0.03248528156482305,
                          0.8011772846583444,
                          0.13385215221200952,
                          0.03248528156482305,
                          0.628071845475366,
                          0.174979342183939,
                          0.02196947015675594,
                          0.628071845475366,
                          0.174979342183939,
                          0.174979342183939,
                          0.174979342183939,
                          0.174979342183939,
                          0.628071845475366,
                          0.02196947015675594,
                          0.628071845475366,
                          0.174979342183939,
                          0.174979342183939,
                          0.02196947015675594,
                          0.628071845475366,
                          0.174979342183939,
                          0.628071845475366,
                          0.174979342183939,
                          0.02196947015675594,
                          0.174979342183939,
                          0.628071845475366,
                          0.174979342183939,
                          0.02196947015675594,
                          0.174979342183939,
                          0.174979342183939,
                          0.174979342183939,
                          0.02196947015675594,
                          0.174979342183939,
                          0.628071845475366,
                          0.02196947015675594,
                          0.02196947015675594,
                          0.174979342183939,
                          0.174979342183939,
                          0.628071845475366,
                          0.02196947015675594,
                          0.174979342183939};
      std::vector<T> w = {
          0.00789996225933679,   0.00448950999871145,   0.00448950999871145,
          0.00448950999871145,   0.00448950999871145,   0.001644859952798897,
          0.001644859952798897,  0.001644859952798897,  0.001644859952798897,
          0.0018989802033658719, 0.0018989802033658719, 0.0018989802033658719,
          0.0018989802033658719, 0.0018989802033658719, 0.0018989802033658719,
          0.0018989802033658719, 0.0018989802033658719, 0.0018989802033658719,
          0.0018989802033658719, 0.0018989802033658719, 0.0018989802033658719,
          6.03240573898756e-05,  6.03240573898756e-05,  6.03240573898756e-05,
          6.03240573898756e-05,  6.03240573898756e-05,  6.03240573898756e-05,
          6.03240573898756e-05,  6.03240573898756e-05,  6.03240573898756e-05,
          6.03240573898756e-05,  6.03240573898756e-05,  6.03240573898756e-05,
          0.0042899553300760115, 0.0042899553300760115, 0.0042899553300760115,
          0.0042899553300760115, 0.0042899553300760115, 0.0042899553300760115,
          0.0042899553300760115, 0.0042899553300760115, 0.0042899553300760115,
          0.0042899553300760115, 0.0042899553300760115, 0.0042899553300760115,
          0.0016893119466259655, 0.0016893119466259655, 0.0016893119466259655,
          0.0016893119466259655, 0.0016893119466259655, 0.0016893119466259655,
          0.0016893119466259655, 0.0016893119466259655, 0.0016893119466259655,
          0.0016893119466259655, 0.0016893119466259655, 0.0016893119466259655,
          0.0010960245461726506, 0.0010960245461726506, 0.0010960245461726506,
          0.0010960245461726506, 0.0010960245461726506, 0.0010960245461726506,
          0.0010960245461726506, 0.0010960245461726506, 0.0010960245461726506,
          0.0010960245461726506, 0.0010960245461726506, 0.0010960245461726506,
          0.002151172633143665,  0.002151172633143665,  0.002151172633143665,
          0.002151172633143665,  0.002151172633143665,  0.002151172633143665,
          0.002151172633143665,  0.002151172633143665,  0.002151172633143665,
          0.002151172633143665,  0.002151172633143665,  0.002151172633143665};
      return {std::move(x), std::move(w)};
    }
    else
    {
      throw std::runtime_error(
          "Witherden-Vincent not implemented for this order.");
    }
  }
  case cell::type::hexahedron:
  {
    if (m <= 1)
    {
      // Witherden-Vincent, 1 point, degree 1
      std::vector<T> x = {0.5, 0.5, 0.5};
      std::vector<T> w = {1.0};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 3)
    {
      // Witherden-Vincent, 6 points, degree 3
      std::vector<T> x = {0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 1.0, 0.5,
                          0.5, 0.5, 0.0, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5};
      std::vector<T> w
          = {0.16666666666666666, 0.16666666666666666, 0.16666666666666666,
             0.16666666666666666, 0.16666666666666666, 0.16666666666666666};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 5)
    {
      // Witherden-Vincent, 14 points, degree 5
      std::vector<T> x = {0.10208878712288927,
                          0.5,
                          0.5,
                          0.5,
                          0.5,
                          0.8979112128771107,
                          0.5,
                          0.8979112128771107,
                          0.5,
                          0.5,
                          0.5,
                          0.10208878712288927,
                          0.8979112128771107,
                          0.5,
                          0.5,
                          0.5,
                          0.10208878712288927,
                          0.5,
                          0.8793934553196641,
                          0.12060654468033592,
                          0.12060654468033592,
                          0.12060654468033592,
                          0.8793934553196641,
                          0.8793934553196641,
                          0.12060654468033592,
                          0.8793934553196641,
                          0.12060654468033592,
                          0.12060654468033592,
                          0.12060654468033592,
                          0.12060654468033592,
                          0.12060654468033592,
                          0.12060654468033592,
                          0.8793934553196641,
                          0.8793934553196641,
                          0.8793934553196641,
                          0.12060654468033592,
                          0.8793934553196641,
                          0.8793934553196641,
                          0.8793934553196641,
                          0.8793934553196641,
                          0.12060654468033592,
                          0.8793934553196641};
      std::vector<T> w
          = {0.11080332409972299, 0.11080332409972299, 0.11080332409972299,
             0.11080332409972299, 0.11080332409972299, 0.11080332409972299,
             0.04189750692520776, 0.04189750692520776, 0.04189750692520776,
             0.04189750692520776, 0.04189750692520776, 0.04189750692520776,
             0.04189750692520776, 0.04189750692520776};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 7)
    {
      // Witherden-Vincent, 34 points, degree 7
      std::vector<T> x = {0.005956919405661725,
                          0.5,
                          0.5,
                          0.5,
                          0.5,
                          0.9940430805943383,
                          0.5,
                          0.9940430805943383,
                          0.5,
                          0.5,
                          0.5,
                          0.005956919405661725,
                          0.9940430805943383,
                          0.5,
                          0.5,
                          0.5,
                          0.005956919405661725,
                          0.5,
                          0.7039775836791596,
                          0.2960224163208403,
                          0.2960224163208403,
                          0.2960224163208403,
                          0.7039775836791596,
                          0.7039775836791596,
                          0.2960224163208403,
                          0.7039775836791596,
                          0.2960224163208403,
                          0.2960224163208403,
                          0.2960224163208403,
                          0.2960224163208403,
                          0.2960224163208403,
                          0.2960224163208403,
                          0.7039775836791596,
                          0.7039775836791596,
                          0.7039775836791596,
                          0.2960224163208403,
                          0.7039775836791596,
                          0.7039775836791596,
                          0.7039775836791596,
                          0.7039775836791596,
                          0.2960224163208403,
                          0.7039775836791596,
                          0.8905514105020592,
                          0.10944858949794074,
                          0.10944858949794074,
                          0.10944858949794074,
                          0.8905514105020592,
                          0.8905514105020592,
                          0.10944858949794074,
                          0.8905514105020592,
                          0.10944858949794074,
                          0.10944858949794074,
                          0.10944858949794074,
                          0.10944858949794074,
                          0.10944858949794074,
                          0.10944858949794074,
                          0.8905514105020592,
                          0.8905514105020592,
                          0.8905514105020592,
                          0.10944858949794074,
                          0.8905514105020592,
                          0.8905514105020592,
                          0.8905514105020592,
                          0.8905514105020592,
                          0.10944858949794074,
                          0.8905514105020592,
                          0.07597386215798063,
                          0.07597386215798063,
                          0.5,
                          0.9240261378420194,
                          0.5,
                          0.07597386215798063,
                          0.5,
                          0.9240261378420194,
                          0.07597386215798063,
                          0.9240261378420194,
                          0.9240261378420194,
                          0.5,
                          0.9240261378420194,
                          0.5,
                          0.9240261378420194,
                          0.5,
                          0.07597386215798063,
                          0.9240261378420194,
                          0.5,
                          0.07597386215798063,
                          0.07597386215798063,
                          0.07597386215798063,
                          0.5,
                          0.9240261378420194,
                          0.07597386215798063,
                          0.9240261378420194,
                          0.5,
                          0.9240261378420194,
                          0.07597386215798063,
                          0.5,
                          0.5,
                          0.9240261378420194,
                          0.9240261378420194,
                          0.07597386215798063,
                          0.5,
                          0.07597386215798063};
      std::vector<T> w
          = {0.025016236372945876, 0.025016236372945876, 0.025016236372945876,
             0.025016236372945876, 0.025016236372945876, 0.025016236372945876,
             0.057144232008731595, 0.057144232008731595, 0.057144232008731595,
             0.057144232008731595, 0.057144232008731595, 0.057144232008731595,
             0.057144232008731595, 0.057144232008731595, 0.019224517508244835,
             0.019224517508244835, 0.019224517508244835, 0.019224517508244835,
             0.019224517508244835, 0.019224517508244835, 0.019224517508244835,
             0.019224517508244835, 0.01991271546887611,  0.01991271546887611,
             0.01991271546887611,  0.01991271546887611,  0.01991271546887611,
             0.01991271546887611,  0.01991271546887611,  0.01991271546887611,
             0.01991271546887611,  0.01991271546887611,  0.01991271546887611,
             0.01991271546887611};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 9)
    {
      // Witherden-Vincent, 58 points, degree 9
      std::vector<T> x = {0.1931592652041455,
                          0.5,
                          0.5,
                          0.5,
                          0.5,
                          0.8068407347958545,
                          0.5,
                          0.8068407347958545,
                          0.5,
                          0.5,
                          0.5,
                          0.1931592652041455,
                          0.8068407347958545,
                          0.5,
                          0.5,
                          0.5,
                          0.1931592652041455,
                          0.5,
                          0.935049892330988,
                          0.06495010766901205,
                          0.06495010766901205,
                          0.06495010766901205,
                          0.935049892330988,
                          0.935049892330988,
                          0.06495010766901205,
                          0.935049892330988,
                          0.06495010766901205,
                          0.06495010766901205,
                          0.06495010766901205,
                          0.06495010766901205,
                          0.06495010766901205,
                          0.06495010766901205,
                          0.935049892330988,
                          0.935049892330988,
                          0.935049892330988,
                          0.06495010766901205,
                          0.935049892330988,
                          0.935049892330988,
                          0.935049892330988,
                          0.935049892330988,
                          0.06495010766901205,
                          0.935049892330988,
                          0.782055403510015,
                          0.21794459648998496,
                          0.21794459648998496,
                          0.21794459648998496,
                          0.782055403510015,
                          0.782055403510015,
                          0.21794459648998496,
                          0.782055403510015,
                          0.21794459648998496,
                          0.21794459648998496,
                          0.21794459648998496,
                          0.21794459648998496,
                          0.21794459648998496,
                          0.21794459648998496,
                          0.782055403510015,
                          0.782055403510015,
                          0.782055403510015,
                          0.21794459648998496,
                          0.782055403510015,
                          0.782055403510015,
                          0.782055403510015,
                          0.782055403510015,
                          0.21794459648998496,
                          0.782055403510015,
                          0.06115643837116086,
                          0.06115643837116086,
                          0.5,
                          0.9388435616288392,
                          0.5,
                          0.06115643837116086,
                          0.5,
                          0.9388435616288392,
                          0.06115643837116086,
                          0.9388435616288392,
                          0.9388435616288392,
                          0.5,
                          0.9388435616288392,
                          0.5,
                          0.9388435616288392,
                          0.5,
                          0.06115643837116086,
                          0.9388435616288392,
                          0.5,
                          0.06115643837116086,
                          0.06115643837116086,
                          0.06115643837116086,
                          0.5,
                          0.9388435616288392,
                          0.06115643837116086,
                          0.9388435616288392,
                          0.5,
                          0.9388435616288392,
                          0.06115643837116086,
                          0.5,
                          0.5,
                          0.9388435616288392,
                          0.9388435616288392,
                          0.06115643837116086,
                          0.5,
                          0.06115643837116086,
                          0.7161339513154311,
                          0.9692652109323359,
                          0.28386604868456894,
                          0.28386604868456894,
                          0.030734789067664126,
                          0.7161339513154311,
                          0.28386604868456894,
                          0.9692652109323359,
                          0.28386604868456894,
                          0.030734789067664126,
                          0.28386604868456894,
                          0.7161339513154311,
                          0.28386604868456894,
                          0.28386604868456894,
                          0.9692652109323359,
                          0.9692652109323359,
                          0.7161339513154311,
                          0.28386604868456894,
                          0.7161339513154311,
                          0.030734789067664126,
                          0.28386604868456894,
                          0.7161339513154311,
                          0.030734789067664126,
                          0.7161339513154311,
                          0.28386604868456894,
                          0.030734789067664126,
                          0.28386604868456894,
                          0.7161339513154311,
                          0.7161339513154311,
                          0.030734789067664126,
                          0.030734789067664126,
                          0.7161339513154311,
                          0.28386604868456894,
                          0.28386604868456894,
                          0.7161339513154311,
                          0.030734789067664126,
                          0.28386604868456894,
                          0.7161339513154311,
                          0.9692652109323359,
                          0.28386604868456894,
                          0.28386604868456894,
                          0.030734789067664126,
                          0.7161339513154311,
                          0.28386604868456894,
                          0.9692652109323359,
                          0.9692652109323359,
                          0.28386604868456894,
                          0.28386604868456894,
                          0.7161339513154311,
                          0.9692652109323359,
                          0.7161339513154311,
                          0.9692652109323359,
                          0.7161339513154311,
                          0.7161339513154311,
                          0.9692652109323359,
                          0.28386604868456894,
                          0.7161339513154311,
                          0.030734789067664126,
                          0.7161339513154311,
                          0.7161339513154311,
                          0.7161339513154311,
                          0.7161339513154311,
                          0.9692652109323359,
                          0.28386604868456894,
                          0.9692652109323359,
                          0.7161339513154311,
                          0.7161339513154311,
                          0.28386604868456894,
                          0.030734789067664126,
                          0.030734789067664126,
                          0.28386604868456894,
                          0.28386604868456894};
      std::vector<T> w = {
          0.05415937446870682,   0.05415937446870682,   0.05415937446870682,
          0.05415937446870682,   0.05415937446870682,   0.05415937446870682,
          0.0062685994124186285, 0.0062685994124186285, 0.0062685994124186285,
          0.0062685994124186285, 0.0062685994124186285, 0.0062685994124186285,
          0.0062685994124186285, 0.0062685994124186285, 0.024857479768002937,
          0.024857479768002937,  0.024857479768002937,  0.024857479768002937,
          0.024857479768002937,  0.024857479768002937,  0.024857479768002937,
          0.024857479768002937,  0.011473725767022205,  0.011473725767022205,
          0.011473725767022205,  0.011473725767022205,  0.011473725767022205,
          0.011473725767022205,  0.011473725767022205,  0.011473725767022205,
          0.011473725767022205,  0.011473725767022205,  0.011473725767022205,
          0.011473725767022205,  0.012014600439171672,  0.012014600439171672,
          0.012014600439171672,  0.012014600439171672,  0.012014600439171672,
          0.012014600439171672,  0.012014600439171672,  0.012014600439171672,
          0.012014600439171672,  0.012014600439171672,  0.012014600439171672,
          0.012014600439171672,  0.012014600439171672,  0.012014600439171672,
          0.012014600439171672,  0.012014600439171672,  0.012014600439171672,
          0.012014600439171672,  0.012014600439171672,  0.012014600439171672,
          0.012014600439171672,  0.012014600439171672,  0.012014600439171672,
          0.012014600439171672};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 11)
    {
      // Witherden-Vincent, 90 points, degree 11
      std::vector<T> x = {0.09369283295018675,
                          0.5,
                          0.5,
                          0.5,
                          0.5,
                          0.9063071670498133,
                          0.5,
                          0.9063071670498133,
                          0.5,
                          0.5,
                          0.5,
                          0.09369283295018675,
                          0.9063071670498133,
                          0.5,
                          0.5,
                          0.5,
                          0.09369283295018675,
                          0.5,
                          0.8008376320991314,
                          0.19916236790086864,
                          0.19916236790086864,
                          0.19916236790086864,
                          0.8008376320991314,
                          0.8008376320991314,
                          0.19916236790086864,
                          0.8008376320991314,
                          0.19916236790086864,
                          0.19916236790086864,
                          0.19916236790086864,
                          0.19916236790086864,
                          0.19916236790086864,
                          0.19916236790086864,
                          0.8008376320991314,
                          0.8008376320991314,
                          0.8008376320991314,
                          0.19916236790086864,
                          0.8008376320991314,
                          0.8008376320991314,
                          0.8008376320991314,
                          0.8008376320991314,
                          0.19916236790086864,
                          0.8008376320991314,
                          0.92772788050888,
                          0.07227211949112,
                          0.07227211949112,
                          0.07227211949112,
                          0.92772788050888,
                          0.92772788050888,
                          0.07227211949112,
                          0.92772788050888,
                          0.07227211949112,
                          0.07227211949112,
                          0.07227211949112,
                          0.07227211949112,
                          0.07227211949112,
                          0.07227211949112,
                          0.92772788050888,
                          0.92772788050888,
                          0.92772788050888,
                          0.07227211949112,
                          0.92772788050888,
                          0.92772788050888,
                          0.92772788050888,
                          0.92772788050888,
                          0.07227211949112,
                          0.92772788050888,
                          0.6566967022580273,
                          0.34330329774197266,
                          0.34330329774197266,
                          0.34330329774197266,
                          0.6566967022580273,
                          0.6566967022580273,
                          0.34330329774197266,
                          0.6566967022580273,
                          0.34330329774197266,
                          0.34330329774197266,
                          0.34330329774197266,
                          0.34330329774197266,
                          0.34330329774197266,
                          0.34330329774197266,
                          0.6566967022580273,
                          0.6566967022580273,
                          0.6566967022580273,
                          0.34330329774197266,
                          0.6566967022580273,
                          0.6566967022580273,
                          0.6566967022580273,
                          0.6566967022580273,
                          0.34330329774197266,
                          0.6566967022580273,
                          0.13266585650149598,
                          0.13266585650149598,
                          0.5,
                          0.867334143498504,
                          0.5,
                          0.13266585650149598,
                          0.5,
                          0.867334143498504,
                          0.13266585650149598,
                          0.867334143498504,
                          0.867334143498504,
                          0.5,
                          0.867334143498504,
                          0.5,
                          0.867334143498504,
                          0.5,
                          0.13266585650149598,
                          0.867334143498504,
                          0.5,
                          0.13266585650149598,
                          0.13266585650149598,
                          0.13266585650149598,
                          0.5,
                          0.867334143498504,
                          0.13266585650149598,
                          0.867334143498504,
                          0.5,
                          0.867334143498504,
                          0.13266585650149598,
                          0.5,
                          0.5,
                          0.867334143498504,
                          0.867334143498504,
                          0.13266585650149598,
                          0.5,
                          0.13266585650149598,
                          0.7253999675572547,
                          0.9825498327563551,
                          0.2746000324427453,
                          0.2746000324427453,
                          0.017450167243644867,
                          0.7253999675572547,
                          0.2746000324427453,
                          0.9825498327563551,
                          0.2746000324427453,
                          0.017450167243644867,
                          0.2746000324427453,
                          0.7253999675572547,
                          0.2746000324427453,
                          0.2746000324427453,
                          0.9825498327563551,
                          0.9825498327563551,
                          0.7253999675572547,
                          0.2746000324427453,
                          0.7253999675572547,
                          0.017450167243644867,
                          0.2746000324427453,
                          0.7253999675572547,
                          0.017450167243644867,
                          0.7253999675572547,
                          0.2746000324427453,
                          0.017450167243644867,
                          0.2746000324427453,
                          0.7253999675572547,
                          0.7253999675572547,
                          0.017450167243644867,
                          0.017450167243644867,
                          0.7253999675572547,
                          0.2746000324427453,
                          0.2746000324427453,
                          0.7253999675572547,
                          0.017450167243644867,
                          0.2746000324427453,
                          0.7253999675572547,
                          0.9825498327563551,
                          0.2746000324427453,
                          0.2746000324427453,
                          0.017450167243644867,
                          0.7253999675572547,
                          0.2746000324427453,
                          0.9825498327563551,
                          0.9825498327563551,
                          0.2746000324427453,
                          0.2746000324427453,
                          0.7253999675572547,
                          0.9825498327563551,
                          0.7253999675572547,
                          0.9825498327563551,
                          0.7253999675572547,
                          0.7253999675572547,
                          0.9825498327563551,
                          0.2746000324427453,
                          0.7253999675572547,
                          0.017450167243644867,
                          0.7253999675572547,
                          0.7253999675572547,
                          0.7253999675572547,
                          0.7253999675572547,
                          0.9825498327563551,
                          0.2746000324427453,
                          0.9825498327563551,
                          0.7253999675572547,
                          0.7253999675572547,
                          0.2746000324427453,
                          0.017450167243644867,
                          0.017450167243644867,
                          0.2746000324427453,
                          0.2746000324427453,
                          0.9706224286053017,
                          0.676951407298315,
                          0.02937757139469837,
                          0.02937757139469837,
                          0.32304859270168496,
                          0.9706224286053017,
                          0.02937757139469837,
                          0.676951407298315,
                          0.02937757139469837,
                          0.32304859270168496,
                          0.02937757139469837,
                          0.9706224286053017,
                          0.02937757139469837,
                          0.02937757139469837,
                          0.676951407298315,
                          0.676951407298315,
                          0.9706224286053017,
                          0.02937757139469837,
                          0.9706224286053017,
                          0.32304859270168496,
                          0.02937757139469837,
                          0.9706224286053017,
                          0.32304859270168496,
                          0.9706224286053017,
                          0.02937757139469837,
                          0.32304859270168496,
                          0.02937757139469837,
                          0.9706224286053017,
                          0.9706224286053017,
                          0.32304859270168496,
                          0.32304859270168496,
                          0.9706224286053017,
                          0.02937757139469837,
                          0.02937757139469837,
                          0.9706224286053017,
                          0.32304859270168496,
                          0.02937757139469837,
                          0.9706224286053017,
                          0.676951407298315,
                          0.02937757139469837,
                          0.02937757139469837,
                          0.32304859270168496,
                          0.9706224286053017,
                          0.02937757139469837,
                          0.676951407298315,
                          0.676951407298315,
                          0.02937757139469837,
                          0.02937757139469837,
                          0.9706224286053017,
                          0.676951407298315,
                          0.9706224286053017,
                          0.676951407298315,
                          0.9706224286053017,
                          0.9706224286053017,
                          0.676951407298315,
                          0.02937757139469837,
                          0.9706224286053017,
                          0.32304859270168496,
                          0.9706224286053017,
                          0.9706224286053017,
                          0.9706224286053017,
                          0.9706224286053017,
                          0.676951407298315,
                          0.02937757139469837,
                          0.676951407298315,
                          0.9706224286053017,
                          0.9706224286053017,
                          0.02937757139469837,
                          0.32304859270168496,
                          0.32304859270168496,
                          0.02937757139469837,
                          0.02937757139469837};
      std::vector<T> w = {
          0.025309634201600022,  0.025309634201600022,  0.025309634201600022,
          0.025309634201600022,  0.025309634201600022,  0.025309634201600022,
          0.014692293494557035,  0.014692293494557035,  0.014692293494557035,
          0.014692293494557035,  0.014692293494557035,  0.014692293494557035,
          0.014692293494557035,  0.014692293494557035,  0.005580489009853655,
          0.005580489009853655,  0.005580489009853655,  0.005580489009853655,
          0.005580489009853655,  0.005580489009853655,  0.005580489009853655,
          0.005580489009853655,  0.02699900565687114,   0.02699900565687114,
          0.02699900565687114,   0.02699900565687114,   0.02699900565687114,
          0.02699900565687114,   0.02699900565687114,   0.02699900565687114,
          0.018149918232514464,  0.018149918232514464,  0.018149918232514464,
          0.018149918232514464,  0.018149918232514464,  0.018149918232514464,
          0.018149918232514464,  0.018149918232514464,  0.018149918232514464,
          0.018149918232514464,  0.018149918232514464,  0.018149918232514464,
          0.007680249262229417,  0.007680249262229417,  0.007680249262229417,
          0.007680249262229417,  0.007680249262229417,  0.007680249262229417,
          0.007680249262229417,  0.007680249262229417,  0.007680249262229417,
          0.007680249262229417,  0.007680249262229417,  0.007680249262229417,
          0.007680249262229417,  0.007680249262229417,  0.007680249262229417,
          0.007680249262229417,  0.007680249262229417,  0.007680249262229417,
          0.007680249262229417,  0.007680249262229417,  0.007680249262229417,
          0.007680249262229417,  0.007680249262229417,  0.007680249262229417,
          0.0028267870173527354, 0.0028267870173527354, 0.0028267870173527354,
          0.0028267870173527354, 0.0028267870173527354, 0.0028267870173527354,
          0.0028267870173527354, 0.0028267870173527354, 0.0028267870173527354,
          0.0028267870173527354, 0.0028267870173527354, 0.0028267870173527354,
          0.0028267870173527354, 0.0028267870173527354, 0.0028267870173527354,
          0.0028267870173527354, 0.0028267870173527354, 0.0028267870173527354,
          0.0028267870173527354, 0.0028267870173527354, 0.0028267870173527354,
          0.0028267870173527354, 0.0028267870173527354, 0.0028267870173527354};
      return {std::move(x), std::move(w)};
    }
    else
    {
      throw std::runtime_error(
          "Witherden-Vincent not implemented for this order.");
    }
  }
  case cell::type::prism:
  {
    if (m <= 1)
    {
      // Witherden-Vincent, 1 point, degree 1
      std::vector<T> x = {0.3333333333333333, 0.3333333333333333, 0.5};
      std::vector<T> w = {0.5};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 2)
    {
      // Witherden-Vincent, 5 points, degree 2
      std::vector<T> x
          = {0.3333333333333333, 0.3333333333333333, 7.440348094461739e-17,
             0.3333333333333333, 0.3333333333333333, 0.9999999999999999,
             0.1292091881014018, 0.7415816237971964, 0.5,
             0.7415816237971964, 0.1292091881014018, 0.5,
             0.1292091881014018, 0.1292091881014018, 0.5};
      std::vector<T> w
          = {0.08333333333333336, 0.08333333333333336, 0.11111111111111109,
             0.11111111111111109, 0.11111111111111109};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 3)
    {
      // Witherden-Vincent, 8 points, degree 3
      std::vector<T> x
          = {0.3333333333333333,    0.3333333333333333,    0.06966539776761103,
             0.3333333333333333,    0.3333333333333333,    0.930334602232389,
             0.23888343595657835,   0.7611141478748917,    0.5,
             0.7611141478748917,    0.23888343595657835,   0.5,
             2.416168529911161e-06, 0.7611141478748917,    0.5,
             0.7611141478748917,    2.416168529911161e-06, 0.5,
             2.416168529911161e-06, 0.23888343595657835,   0.5,
             0.23888343595657835,   2.416168529911161e-06, 0.5};
      std::vector<T> w
          = {0.11249836907245024, 0.11249836907245024, 0.04583387697584992,
             0.04583387697584992, 0.04583387697584992, 0.04583387697584992,
             0.04583387697584992, 0.04583387697584992};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 4)
    {
      // Witherden-Vincent, 11 points, degree 4
      std::vector<T> x
          = {0.3333333333333333,  0.3333333333333333,  0.0665690129954826,
             0.3333333333333333,  0.3333333333333333,  0.9334309870045174,
             0.46865580986199523, 0.06268838027600958, 0.5,
             0.06268838027600958, 0.46865580986199523, 0.5,
             0.46865580986199523, 0.46865580986199523, 0.5,
             0.10074040579891064, 0.7985191884021787,  0.16218008815887014,
             0.7985191884021787,  0.10074040579891064, 0.16218008815887014,
             0.10074040579891064, 0.10074040579891064, 0.16218008815887014,
             0.10074040579891064, 0.7985191884021787,  0.8378199118411298,
             0.7985191884021787,  0.10074040579891064, 0.8378199118411298,
             0.10074040579891064, 0.10074040579891064, 0.8378199118411298};
      std::vector<T> w
          = {0.053955987407767754, 0.053955987407767754, 0.06820730630273882,
             0.06820730630273882,  0.06820730630273882,  0.031244351046041337,
             0.031244351046041337, 0.031244351046041337, 0.031244351046041337,
             0.031244351046041337, 0.031244351046041337};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 5)
    {
      // Witherden-Vincent, 16 points, degree 5
      std::vector<T> x
          = {0.3333333333333333,  0.3333333333333333,  0.5,
             0.48729996455024566, 0.02540007089950867, 0.5,
             0.02540007089950867, 0.48729996455024566, 0.5,
             0.48729996455024566, 0.48729996455024566, 0.5,
             0.44559810467037203, 0.108803790659256,   0.06449853256727797,
             0.108803790659256,   0.44559810467037203, 0.06449853256727797,
             0.44559810467037203, 0.44559810467037203, 0.06449853256727797,
             0.44559810467037203, 0.108803790659256,   0.935501467432722,
             0.108803790659256,   0.44559810467037203, 0.935501467432722,
             0.44559810467037203, 0.44559810467037203, 0.935501467432722,
             0.10085894598270853, 0.798282108034583,   0.21478650964742035,
             0.798282108034583,   0.10085894598270853, 0.21478650964742035,
             0.10085894598270853, 0.10085894598270853, 0.21478650964742035,
             0.10085894598270853, 0.798282108034583,   0.7852134903525796,
             0.798282108034583,   0.10085894598270853, 0.7852134903525796,
             0.10085894598270853, 0.10085894598270853, 0.7852134903525796};
      std::vector<T> w
          = {0.08893194449143586,  0.028088758403533358, 0.028088758403533358,
             0.028088758403533358, 0.02320767766451977,  0.02320767766451977,
             0.02320767766451977,  0.02320767766451977,  0.02320767766451977,
             0.02320767766451977,  0.03125928571847424,  0.03125928571847424,
             0.03125928571847424,  0.03125928571847424,  0.03125928571847424,
             0.03125928571847424};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 6)
    {
      // Witherden-Vincent, 28 points, degree 6
      std::vector<T> x
          = {0.3333333333333333,   0.3333333333333333,   0.23668864759751262,
             0.3333333333333333,   0.3333333333333333,   0.7633113524024874,
             0.3333333333333333,   0.3333333333333333,   0.004581849590377626,
             0.3333333333333333,   0.3333333333333333,   0.9954181504096223,
             0.16520326502809246,  0.6695934699438151,   0.5,
             0.6695934699438151,   0.16520326502809246,  0.5,
             0.16520326502809246,  0.16520326502809246,  0.5,
             0.020473269612256918, 0.9590534607754861,   0.5,
             0.9590534607754861,   0.020473269612256918, 0.5,
             0.020473269612256918, 0.020473269612256918, 0.5,
             0.4661116655901595,   0.06777666881968099,  0.25916293255780626,
             0.06777666881968099,  0.4661116655901595,   0.25916293255780626,
             0.4661116655901595,   0.4661116655901595,   0.25916293255780626,
             0.4661116655901595,   0.06777666881968099,  0.7408370674421938,
             0.06777666881968099,  0.4661116655901595,   0.7408370674421938,
             0.4661116655901595,   0.4661116655901595,   0.7408370674421938,
             0.20257389672523077,  0.7627467408337223,   0.09535710337208363,
             0.7627467408337223,   0.20257389672523077,  0.09535710337208363,
             0.03467936244104695,  0.7627467408337223,   0.09535710337208363,
             0.7627467408337223,   0.03467936244104695,  0.09535710337208363,
             0.03467936244104695,  0.20257389672523077,  0.09535710337208363,
             0.20257389672523077,  0.03467936244104695,  0.09535710337208363,
             0.20257389672523077,  0.7627467408337223,   0.9046428966279164,
             0.7627467408337223,   0.20257389672523077,  0.9046428966279164,
             0.03467936244104695,  0.7627467408337223,   0.9046428966279164,
             0.7627467408337223,   0.03467936244104695,  0.9046428966279164,
             0.03467936244104695,  0.20257389672523077,  0.9046428966279164,
             0.20257389672523077,  0.03467936244104695,  0.9046428966279164};
      std::vector<T> w
          = {0.027836641761528128, 0.027836641761528128, 0.017299513671765562,
             0.017299513671765562, 0.038978501851666916, 0.038978501851666916,
             0.038978501851666916, 0.006268316904837069, 0.006268316904837069,
             0.006268316904837069, 0.02450632002576027,  0.02450632002576027,
             0.02450632002576027,  0.02450632002576027,  0.02450632002576027,
             0.02450632002576027,  0.01057910939244492,  0.01057910939244492,
             0.01057910939244492,  0.01057910939244492,  0.01057910939244492,
             0.01057910939244492,  0.01057910939244492,  0.01057910939244492,
             0.01057910939244492,  0.01057910939244492,  0.01057910939244492,
             0.01057910939244492};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 7)
    {
      // Witherden-Vincent, 35 points, degree 7
      std::vector<T> x
          = {0.3333333333333333,   0.3333333333333333,   0.0098859652045542,
             0.3333333333333333,   0.3333333333333333,   0.9901140347954458,
             0.008337595466021473, 0.983324809067957,    0.5,
             0.983324809067957,    0.008337595466021473, 0.5,
             0.008337595466021473, 0.008337595466021473, 0.5,
             0.4815219753291366,   0.0369560493417268,   0.079306322896737,
             0.0369560493417268,   0.4815219753291366,   0.079306322896737,
             0.4815219753291366,   0.4815219753291366,   0.079306322896737,
             0.4815219753291366,   0.0369560493417268,   0.920693677103263,
             0.0369560493417268,   0.4815219753291366,   0.920693677103263,
             0.4815219753291366,   0.4815219753291366,   0.920693677103263,
             0.09548324837148936,  0.8090335032570213,   0.10207545470650847,
             0.8090335032570213,   0.09548324837148936,  0.10207545470650847,
             0.09548324837148936,  0.09548324837148936,  0.10207545470650847,
             0.09548324837148936,  0.8090335032570213,   0.8979245452934915,
             0.8090335032570213,   0.09548324837148936,  0.8979245452934915,
             0.09548324837148936,  0.09548324837148936,  0.8979245452934915,
             0.01214913159837829,  0.7429966820728956,   0.5,
             0.7429966820728956,   0.01214913159837829,  0.5,
             0.24485418632872608,  0.7429966820728956,   0.5,
             0.7429966820728956,   0.24485418632872608,  0.5,
             0.24485418632872608,  0.01214913159837829,  0.5,
             0.01214913159837829,  0.24485418632872608,  0.5,
             0.30515621643222607,  0.1529845984247976,   0.298032719733945,
             0.1529845984247976,   0.30515621643222607,  0.298032719733945,
             0.5418591851429763,   0.1529845984247976,   0.298032719733945,
             0.1529845984247976,   0.5418591851429763,   0.298032719733945,
             0.5418591851429763,   0.30515621643222607,  0.298032719733945,
             0.30515621643222607,  0.5418591851429763,   0.298032719733945,
             0.30515621643222607,  0.1529845984247976,   0.7019672802660549,
             0.1529845984247976,   0.30515621643222607,  0.7019672802660549,
             0.5418591851429763,   0.1529845984247976,   0.7019672802660549,
             0.1529845984247976,   0.5418591851429763,   0.7019672802660549,
             0.5418591851429763,   0.30515621643222607,  0.7019672802660549,
             0.30515621643222607,  0.5418591851429763,   0.7019672802660549};
      std::vector<T> w = {
          0.014222840556344644,  0.014222840556344644,  0.0030591210865475197,
          0.0030591210865475197, 0.0030591210865475197, 0.010775432043113231,
          0.010775432043113231,  0.010775432043113231,  0.010775432043113231,
          0.010775432043113231,  0.010775432043113231,  0.01458926245104923,
          0.01458926245104923,   0.01458926245104923,   0.01458926245104923,
          0.01458926245104923,   0.01458926245104923,   0.012757428167574626,
          0.012757428167574626,  0.012757428167574626,  0.012757428167574626,
          0.012757428167574626,  0.012757428167574626,  0.019470351638103803,
          0.019470351638103803,  0.019470351638103803,  0.019470351638103803,
          0.019470351638103803,  0.019470351638103803,  0.019470351638103803,
          0.019470351638103803,  0.019470351638103803,  0.019470351638103803,
          0.019470351638103803,  0.019470351638103803};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 8)
    {
      // Witherden-Vincent, 46 points, degree 8
      std::vector<T> x
          = {0.3333333333333333,   0.3333333333333333,   0.3877830669470047,
             0.3333333333333333,   0.3333333333333333,   0.6122169330529953,
             0.3333333333333333,   0.3333333333333333,   0.15908727921456708,
             0.3333333333333333,   0.3333333333333333,   0.8409127207854329,
             0.05341235093690714,  0.8931752981261857,   0.5,
             0.8931752981261857,   0.05341235093690714,  0.5,
             0.05341235093690714,  0.05341235093690714,  0.5,
             0.46008896281371064,  0.07982207437257877,  0.5,
             0.07982207437257877,  0.46008896281371064,  0.5,
             0.46008896281371064,  0.46008896281371064,  0.5,
             0.45856906879095133,  0.08286186241809733,  0.061902044999872875,
             0.08286186241809733,  0.45856906879095133,  0.061902044999872875,
             0.45856906879095133,  0.45856906879095133,  0.061902044999872875,
             0.45856906879095133,  0.08286186241809733,  0.9380979550001272,
             0.08286186241809733,  0.45856906879095133,  0.9380979550001272,
             0.45856906879095133,  0.45856906879095133,  0.9380979550001272,
             0.1740616079243704,   0.6518767841512593,   0.2969578853227049,
             0.6518767841512593,   0.1740616079243704,   0.2969578853227049,
             0.1740616079243704,   0.1740616079243704,   0.2969578853227049,
             0.1740616079243704,   0.6518767841512593,   0.7030421146772952,
             0.6518767841512593,   0.1740616079243704,   0.7030421146772952,
             0.1740616079243704,   0.1740616079243704,   0.7030421146772952,
             0.04723878583976938,  0.9055224283204613,   0.0855922784706599,
             0.9055224283204613,   0.04723878583976938,  0.0855922784706599,
             0.04723878583976938,  0.04723878583976938,  0.0855922784706599,
             0.04723878583976938,  0.9055224283204613,   0.9144077215293401,
             0.9055224283204613,   0.04723878583976938,  0.9144077215293401,
             0.04723878583976938,  0.04723878583976938,  0.9144077215293401,
             0.15974926394258895,  0.680501472114822,    0.017067416729672824,
             0.680501472114822,    0.15974926394258895,  0.017067416729672824,
             0.15974926394258895,  0.15974926394258895,  0.017067416729672824,
             0.15974926394258895,  0.680501472114822,    0.9829325832703272,
             0.680501472114822,    0.15974926394258895,  0.9829325832703272,
             0.15974926394258895,  0.15974926394258895,  0.9829325832703272,
             0.262813800691241,    0.728598071801,       0.2113248654051871,
             0.728598071801,       0.262813800691241,    0.2113248654051871,
             0.008588127507759013, 0.728598071801,       0.2113248654051871,
             0.728598071801,       0.008588127507759013, 0.2113248654051871,
             0.008588127507759013, 0.262813800691241,    0.2113248654051871,
             0.262813800691241,    0.008588127507759013, 0.2113248654051871,
             0.262813800691241,    0.728598071801,       0.7886751345948129,
             0.728598071801,       0.262813800691241,    0.7886751345948129,
             0.008588127507759013, 0.728598071801,       0.7886751345948129,
             0.728598071801,       0.008588127507759013, 0.7886751345948129,
             0.008588127507759013, 0.262813800691241,    0.7886751345948129,
             0.262813800691241,    0.008588127507759013, 0.7886751345948129};
      std::vector<T> w
          = {0.01036746832142773,  0.01036746832142773,  0.025544713556090774,
             0.025544713556090774, 0.009076124392074858, 0.009076124392074858,
             0.009076124392074858, 0.026349872453253584, 0.026349872453253584,
             0.026349872453253584, 0.010543254646793246, 0.010543254646793246,
             0.010543254646793246, 0.010543254646793246, 0.010543254646793246,
             0.010543254646793246, 0.020314631329494664, 0.020314631329494664,
             0.020314631329494664, 0.020314631329494664, 0.020314631329494664,
             0.020314631329494664, 0.003572678876769308, 0.003572678876769308,
             0.003572678876769308, 0.003572678876769308, 0.003572678876769308,
             0.003572678876769308, 0.005571513674232654, 0.005571513674232654,
             0.005571513674232654, 0.005571513674232654, 0.005571513674232654,
             0.005571513674232654, 0.006823764545436537, 0.006823764545436537,
             0.006823764545436537, 0.006823764545436537, 0.006823764545436537,
             0.006823764545436537, 0.006823764545436537, 0.006823764545436537,
             0.006823764545436537, 0.006823764545436537, 0.006823764545436537,
             0.006823764545436537};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 9)
    {
      // Witherden-Vincent, 60 points, degree 9
      std::vector<T> x
          = {0.3333333333333333,   0.3333333333333333,   0.5,
             0.3333333333333333,   0.3333333333333333,   0.06208042060006119,
             0.3333333333333333,   0.3333333333333333,   0.9379195793999389,
             0.4994650351706859,   0.001069929658628155, 0.5,
             0.001069929658628155, 0.4994650351706859,   0.5,
             0.4994650351706859,   0.4994650351706859,   0.5,
             0.44506750004250367,  0.10986499991499268,  0.23518322239696163,
             0.10986499991499268,  0.44506750004250367,  0.23518322239696163,
             0.44506750004250367,  0.44506750004250367,  0.23518322239696163,
             0.44506750004250367,  0.10986499991499268,  0.7648167776030383,
             0.10986499991499268,  0.44506750004250367,  0.7648167776030383,
             0.44506750004250367,  0.44506750004250367,  0.7648167776030383,
             0.21115443371051051,  0.577691132578979,    0.33801155891561496,
             0.577691132578979,    0.21115443371051051,  0.33801155891561496,
             0.21115443371051051,  0.21115443371051051,  0.33801155891561496,
             0.21115443371051051,  0.577691132578979,    0.661988441084385,
             0.577691132578979,    0.21115443371051051,  0.661988441084385,
             0.21115443371051051,  0.21115443371051051,  0.661988441084385,
             0.043842481441492824, 0.9123150371170143,   0.29917416571460564,
             0.9123150371170143,   0.043842481441492824, 0.29917416571460564,
             0.043842481441492824, 0.043842481441492824, 0.29917416571460564,
             0.043842481441492824, 0.9123150371170143,   0.7008258342853944,
             0.9123150371170143,   0.043842481441492824, 0.7008258342853944,
             0.043842481441492824, 0.043842481441492824, 0.7008258342853944,
             0.4827499316708669,   0.03450013665826621,  0.016103937267894747,
             0.03450013665826621,  0.4827499316708669,   0.016103937267894747,
             0.4827499316708669,   0.4827499316708669,   0.016103937267894747,
             0.4827499316708669,   0.03450013665826621,  0.9838960627321053,
             0.03450013665826621,  0.4827499316708669,   0.9838960627321053,
             0.4827499316708669,   0.4827499316708669,   0.9838960627321053,
             0.17717824411167726,  0.6456435117766455,   0.054609129771514664,
             0.6456435117766455,   0.17717824411167726,  0.054609129771514664,
             0.17717824411167726,  0.17717824411167726,  0.054609129771514664,
             0.17717824411167726,  0.6456435117766455,   0.9453908702284853,
             0.6456435117766455,   0.17717824411167726,  0.9453908702284853,
             0.17717824411167726,  0.17717824411167726,  0.9453908702284853,
             0.046231180410943824, 0.9075376391781124,   0.02114689805353652,
             0.9075376391781124,   0.046231180410943824, 0.02114689805353652,
             0.046231180410943824, 0.046231180410943824, 0.02114689805353652,
             0.046231180410943824, 0.9075376391781124,   0.9788531019464635,
             0.9075376391781124,   0.046231180410943824, 0.9788531019464635,
             0.046231180410943824, 0.046231180410943824, 0.9788531019464635,
             0.05936700974336562,  0.7246952695529995,   0.5,
             0.7246952695529995,   0.05936700974336562,  0.5,
             0.2159377207036349,   0.7246952695529995,   0.5,
             0.7246952695529995,   0.2159377207036349,   0.5,
             0.2159377207036349,   0.05936700974336562,  0.5,
             0.05936700974336562,  0.2159377207036349,   0.5,
             0.22412754575335606,  0.7484054595660387,   0.1523897230641826,
             0.7484054595660387,   0.22412754575335606,  0.1523897230641826,
             0.027466994680605237, 0.7484054595660387,   0.1523897230641826,
             0.7484054595660387,   0.027466994680605237, 0.1523897230641826,
             0.027466994680605237, 0.22412754575335606,  0.1523897230641826,
             0.22412754575335606,  0.027466994680605237, 0.1523897230641826,
             0.22412754575335606,  0.7484054595660387,   0.8476102769358174,
             0.7484054595660387,   0.22412754575335606,  0.8476102769358174,
             0.027466994680605237, 0.7484054595660387,   0.8476102769358174,
             0.7484054595660387,   0.027466994680605237, 0.8476102769358174,
             0.027466994680605237, 0.22412754575335606,  0.8476102769358174,
             0.22412754575335606,  0.027466994680605237, 0.8476102769358174};
      std::vector<T> w = {
          0.023893498555775137,  0.01326211494866592,   0.01326211494866592,
          0.006182800680236903,  0.006182800680236903,  0.006182800680236903,
          0.018850991576594825,  0.018850991576594825,  0.018850991576594825,
          0.018850991576594825,  0.018850991576594825,  0.018850991576594825,
          0.0116728087181923,    0.0116728087181923,    0.0116728087181923,
          0.0116728087181923,    0.0116728087181923,    0.0116728087181923,
          0.004798535798150804,  0.004798535798150804,  0.004798535798150804,
          0.004798535798150804,  0.004798535798150804,  0.004798535798150804,
          0.003316053900290054,  0.003316053900290054,  0.003316053900290054,
          0.003316053900290054,  0.003316053900290054,  0.003316053900290054,
          0.00833999282168105,   0.00833999282168105,   0.00833999282168105,
          0.00833999282168105,   0.00833999282168105,   0.00833999282168105,
          0.0015163967295508936, 0.0015163967295508936, 0.0015163967295508936,
          0.0015163967295508936, 0.0015163967295508936, 0.0015163967295508936,
          0.010961395164104885,  0.010961395164104885,  0.010961395164104885,
          0.010961395164104885,  0.010961395164104885,  0.010961395164104885,
          0.006191401771232786,  0.006191401771232786,  0.006191401771232786,
          0.006191401771232786,  0.006191401771232786,  0.006191401771232786,
          0.006191401771232786,  0.006191401771232786,  0.006191401771232786,
          0.006191401771232786,  0.006191401771232786,  0.006191401771232786};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 10)
    {
      // Witherden-Vincent, 85 points, degree 10
      std::vector<T> x = {
          0.3333333333333333,     0.3333333333333333,     0.0,
          0.3333333333333333,     0.3333333333333333,     1.0,
          0.3333333333333333,     0.3333333333333333,     0.3580817993007142,
          0.3333333333333333,     0.3333333333333333,     0.6419182006992857,
          0.20745037498921712,    0.5850992500215658,     0.5,
          0.5850992500215658,     0.20745037498921712,    0.5,
          0.20745037498921712,    0.20745037498921712,    0.5,
          0.45612736286808175,    0.08774527426383646,    0.5,
          0.08774527426383646,    0.45612736286808175,    0.5,
          0.45612736286808175,    0.45612736286808175,    0.5,
          0.08331925215005274,    0.8333614956998945,     0.5,
          0.8333614956998945,     0.08331925215005274,    0.5,
          0.08331925215005274,    0.08331925215005274,    0.5,
          0.2071663521398311,     0.5856672957203378,     0.2608548695498159,
          0.5856672957203378,     0.2071663521398311,     0.2608548695498159,
          0.2071663521398311,     0.2071663521398311,     0.2608548695498159,
          0.2071663521398311,     0.5856672957203378,     0.739145130450184,
          0.5856672957203378,     0.2071663521398311,     0.739145130450184,
          0.2071663521398311,     0.2071663521398311,     0.739145130450184,
          0.14299119654284081,    0.7140176069143184,     0.07396666800043018,
          0.7140176069143184,     0.14299119654284081,    0.07396666800043018,
          0.14299119654284081,    0.14299119654284081,    0.07396666800043018,
          0.14299119654284081,    0.7140176069143184,     0.9260333319995698,
          0.7140176069143184,     0.14299119654284081,    0.9260333319995698,
          0.14299119654284081,    0.14299119654284081,    0.9260333319995698,
          0.4970408872397459,     0.005918225520508265,   0.18625494417602328,
          0.005918225520508265,   0.4970408872397459,     0.18625494417602328,
          0.4970408872397459,     0.4970408872397459,     0.18625494417602328,
          0.4970408872397459,     0.005918225520508265,   0.8137450558239767,
          0.005918225520508265,   0.4970408872397459,     0.8137450558239767,
          0.4970408872397459,     0.4970408872397459,     0.8137450558239767,
          0.4277824598534921,     0.1444350802930158,     0.11551405872352948,
          0.1444350802930158,     0.4277824598534921,     0.11551405872352948,
          0.4277824598534921,     0.4277824598534921,     0.11551405872352948,
          0.4277824598534921,     0.1444350802930158,     0.8844859412764705,
          0.1444350802930158,     0.4277824598534921,     0.8844859412764705,
          0.4277824598534921,     0.4277824598534921,     0.8844859412764705,
          0.0367246894695538,     0.9265506210608924,     0.027319072039121498,
          0.9265506210608924,     0.0367246894695538,     0.027319072039121498,
          0.0367246894695538,     0.0367246894695538,     0.027319072039121498,
          0.0367246894695538,     0.9265506210608924,     0.9726809279608785,
          0.9265506210608924,     0.0367246894695538,     0.9726809279608785,
          0.0367246894695538,     0.0367246894695538,     0.9726809279608785,
          0.0018926195990493983,  0.9979100409523263,     0.5,
          0.9979100409523263,     0.0018926195990493983,  0.5,
          0.00019733944862427498, 0.9979100409523263,     0.5,
          0.9979100409523263,     0.00019733944862427498, 0.5,
          0.00019733944862427498, 0.0018926195990493983,  0.5,
          0.0018926195990493983,  0.00019733944862427498, 0.5,
          0.04328968377949191,    0.6627532432395955,     0.028818266274581252,
          0.6627532432395955,     0.04328968377949191,    0.028818266274581252,
          0.2939570729809126,     0.6627532432395955,     0.028818266274581252,
          0.6627532432395955,     0.2939570729809126,     0.028818266274581252,
          0.2939570729809126,     0.04328968377949191,    0.028818266274581252,
          0.04328968377949191,    0.2939570729809126,     0.028818266274581252,
          0.04328968377949191,    0.6627532432395955,     0.9711817337254187,
          0.6627532432395955,     0.04328968377949191,    0.9711817337254187,
          0.2939570729809126,     0.6627532432395955,     0.9711817337254187,
          0.6627532432395955,     0.2939570729809126,     0.9711817337254187,
          0.2939570729809126,     0.04328968377949191,    0.9711817337254187,
          0.04328968377949191,    0.2939570729809126,     0.9711817337254187,
          0.03829151917220358,    0.6875657341783573,     0.3323319456663028,
          0.6875657341783573,     0.03829151917220358,    0.3323319456663028,
          0.27414274664943916,    0.6875657341783573,     0.3323319456663028,
          0.6875657341783573,     0.27414274664943916,    0.3323319456663028,
          0.27414274664943916,    0.03829151917220358,    0.3323319456663028,
          0.03829151917220358,    0.27414274664943916,    0.3323319456663028,
          0.03829151917220358,    0.6875657341783573,     0.6676680543336971,
          0.6875657341783573,     0.03829151917220358,    0.6676680543336971,
          0.27414274664943916,    0.6875657341783573,     0.6676680543336971,
          0.6875657341783573,     0.27414274664943916,    0.6676680543336971,
          0.27414274664943916,    0.03829151917220358,    0.6676680543336971,
          0.03829151917220358,    0.27414274664943916,    0.6676680543336971,
          0.016039393627233445,   0.870263608639369,      0.1928382490614613,
          0.870263608639369,      0.016039393627233445,   0.1928382490614613,
          0.11369699773339756,    0.870263608639369,      0.1928382490614613,
          0.870263608639369,      0.11369699773339756,    0.1928382490614613,
          0.11369699773339756,    0.016039393627233445,   0.1928382490614613,
          0.016039393627233445,   0.11369699773339756,    0.1928382490614613,
          0.016039393627233445,   0.870263608639369,      0.8071617509385387,
          0.870263608639369,      0.016039393627233445,   0.8071617509385387,
          0.11369699773339756,    0.870263608639369,      0.8071617509385387,
          0.870263608639369,      0.11369699773339756,    0.8071617509385387,
          0.11369699773339756,    0.016039393627233445,   0.8071617509385387,
          0.016039393627233445,   0.11369699773339756,    0.8071617509385387};
      std::vector<T> w = {
          0.004755388510530096,  0.004755388510530096,  0.016006150804069735,
          0.016006150804069735,  0.005677894810017122,  0.005677894810017122,
          0.005677894810017122,  0.015098327262543287,  0.015098327262543287,
          0.015098327262543287,  0.008815454750201412,  0.008815454750201412,
          0.008815454750201412,  0.013044147663042542,  0.013044147663042542,
          0.013044147663042542,  0.013044147663042542,  0.013044147663042542,
          0.013044147663042542,  0.00566913122128927,   0.00566913122128927,
          0.00566913122128927,   0.00566913122128927,   0.00566913122128927,
          0.00566913122128927,   0.003515821706869243,  0.003515821706869243,
          0.003515821706869243,  0.003515821706869243,  0.003515821706869243,
          0.003515821706869243,  0.012162508439342815,  0.012162508439342815,
          0.012162508439342815,  0.012162508439342815,  0.012162508439342815,
          0.012162508439342815,  0.0009462178550419522, 0.0009462178550419522,
          0.0009462178550419522, 0.0009462178550419522, 0.0009462178550419522,
          0.0009462178550419522, 0.0003901079550228376, 0.0003901079550228376,
          0.0003901079550228376, 0.0003901079550228376, 0.0003901079550228376,
          0.0003901079550228376, 0.003022942581225831,  0.003022942581225831,
          0.003022942581225831,  0.003022942581225831,  0.003022942581225831,
          0.003022942581225831,  0.003022942581225831,  0.003022942581225831,
          0.003022942581225831,  0.003022942581225831,  0.003022942581225831,
          0.003022942581225831,  0.0068253885920864345, 0.0068253885920864345,
          0.0068253885920864345, 0.0068253885920864345, 0.0068253885920864345,
          0.0068253885920864345, 0.0068253885920864345, 0.0068253885920864345,
          0.0068253885920864345, 0.0068253885920864345, 0.0068253885920864345,
          0.0068253885920864345, 0.003096192314926311,  0.003096192314926311,
          0.003096192314926311,  0.003096192314926311,  0.003096192314926311,
          0.003096192314926311,  0.003096192314926311,  0.003096192314926311,
          0.003096192314926311,  0.003096192314926311,  0.003096192314926311,
          0.003096192314926311};
      return {std::move(x), std::move(w)};
    }
    else
    {
      throw std::runtime_error(
          "Witherden-Vincent not implemented for this order.");
    }
  }
  case cell::type::pyramid:
  {
    if (m <= 1)
    {
      // Witherden-Vincent, 1 point, degree 1
      std::vector<T> x = {0.375, 0.375, 0.25};
      std::vector<T> w = {0.3333333333333333};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 2)
    {
      // Witherden-Vincent, 5 points, degree 2
      std::vector<T> x
          = {0.19585448072011055, 0.19585448072011055, 0.6082910385597788,
             0.7867922861194703,  0.42733175821357217, 0.14533648357285572,
             0.42733175821357217, 0.7867922861194703,  0.14533648357285572,
             0.06787123030767406, 0.42733175821357217, 0.14533648357285572,
             0.42733175821357217, 0.06787123030767406, 0.14533648357285572};
      std::vector<T> w
          = {0.07535910044136741, 0.06449355822299148, 0.06449355822299148,
             0.06449355822299148, 0.06449355822299148};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 3)
    {
      // Witherden-Vincent, 6 points, degree 3
      std::vector<T> x
          = {0.21428571480696582, 0.21428571480696582, 0.5714285703860683,
             0.499999997162075,   0.499999997162075,   5.675850031606506e-09,
             0.6972084721960364,  0.6972084721960364,  0.16666666666666666,
             0.6972084721960364,  0.13612486113729685, 0.16666666666666666,
             0.13612486113729685, 0.6972084721960364,  0.16666666666666666,
             0.13612486113729685, 0.13612486113729685, 0.16666666666666666};
      std::vector<T> w
          = {0.08406862797425352, 0.03750000202202165, 0.05294117583426454,
             0.05294117583426454, 0.05294117583426454, 0.05294117583426454};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 4)
    {
      // Witherden-Vincent, 10 points, degree 4
      std::vector<T> x
          = {0.16138360555693132,  0.16138360555693132,  0.6772327888861374,
             0.4374315234456268,   0.4374315234456268,   0.12513695310874645,
             0.6640987034100094,   0.33880792521089315,  0.32238414957821365,
             0.33880792521089315,  0.6640987034100094,   0.32238414957821365,
             0.013517147011776915, 0.33880792521089315,  0.32238414957821365,
             0.33880792521089315,  0.013517147011776915, 0.32238414957821365,
             0.8093593566114373,   0.8093593566114373,   0.03924828389881535,
             0.8093593566114373,   0.15139235948974727,  0.03924828389881535,
             0.15139235948974727,  0.8093593566114373,   0.03924828389881535,
             0.15139235948974727,  0.15139235948974727,  0.03924828389881535};
      std::vector<T> w
          = {0.03791396105688065,  0.06896113419651742,  0.035441529296441836,
             0.035441529296441836, 0.035441529296441836, 0.035441529296441836,
             0.021173030223541982, 0.021173030223541982, 0.021173030223541982,
             0.021173030223541982};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 5)
    {
      // Witherden-Vincent, 15 points, degree 5
      std::vector<T> x = {0.13507105960874666,
                          0.13507105960874666,
                          0.7298578807825067,
                          0.3497994895931155,
                          0.3497994895931155,
                          0.300401020813769,
                          0.49999999675411394,
                          0.49999999675411394,
                          6.4917721537927e-09,
                          0.7907630157731623,
                          0.4375,
                          0.125,
                          0.4375,
                          0.7907630157731623,
                          0.125,
                          0.08423698422683772,
                          0.4375,
                          0.125,
                          0.4375,
                          0.08423698422683772,
                          0.125,
                          0.8220026078584023,
                          0.8220026078584023,
                          0.06111190706202296,
                          0.8220026078584023,
                          0.11688548507957472,
                          0.06111190706202296,
                          0.11688548507957472,
                          0.8220026078584023,
                          0.06111190706202296,
                          0.11688548507957472,
                          0.11688548507957472,
                          0.06111190706202296,
                          0.5046407634919081,
                          0.5046407634919081,
                          0.4236013371197248,
                          0.5046407634919081,
                          0.0717578993883671,
                          0.4236013371197248,
                          0.0717578993883671,
                          0.5046407634919081,
                          0.4236013371197248,
                          0.0717578993883671,
                          0.0717578993883671,
                          0.4236013371197248};
      std::vector<T> w
          = {0.022811789969713364, 0.05646570483090801,  0.019568178609524846,
             0.025480431049373406, 0.025480431049373406, 0.025480431049373406,
             0.025480431049373406, 0.013223633859881822, 0.013223633859881822,
             0.013223633859881822, 0.013223633859881822, 0.01991785007154155,
             0.01991785007154155,  0.01991785007154155,  0.01991785007154155};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 6)
    {
      // Witherden-Vincent, 24 points, degree 6
      std::vector<T> x = {0.09617710115302029,
                          0.09617710115302029,
                          0.8076457976939594,
                          0.49911809557359016,
                          0.49911809557359016,
                          0.0017638088528196326,
                          0.4308685967681347,
                          0.4308685967681347,
                          0.13826280646373057,
                          0.2892880440321815,
                          0.2892880440321815,
                          0.42142391193563705,
                          0.8685740050446232,
                          0.45127632948726903,
                          0.09744734102546197,
                          0.45127632948726903,
                          0.8685740050446232,
                          0.09744734102546197,
                          0.03397865392991484,
                          0.45127632948726903,
                          0.09744734102546197,
                          0.45127632948726903,
                          0.03397865392991484,
                          0.09744734102546197,
                          0.4339254093766991,
                          0.21696270468834955,
                          0.5660745906233009,
                          0.21696270468834955,
                          0.4339254093766991,
                          0.5660745906233009,
                          2.5e-39,
                          0.21696270468834955,
                          0.5660745906233009,
                          0.21696270468834955,
                          2.5e-39,
                          0.5660745906233009,
                          0.7681015617899773,
                          0.7681015617899773,
                          0.02947773084572071,
                          0.7681015617899773,
                          0.2024207073643019,
                          0.02947773084572071,
                          0.2024207073643019,
                          0.7681015617899773,
                          0.02947773084572071,
                          0.2024207073643019,
                          0.2024207073643019,
                          0.02947773084572071,
                          0.6165816142842883,
                          0.6165816142842883,
                          0.2649158632121295,
                          0.6165816142842883,
                          0.11850252250358227,
                          0.2649158632121295,
                          0.11850252250358227,
                          0.6165816142842883,
                          0.2649158632121295,
                          0.11850252250358227,
                          0.11850252250358227,
                          0.2649158632121295,
                          0.9513252082912732,
                          0.9513252082912732,
                          0.04824907063193599,
                          0.9513252082912732,
                          0.00042572107679078933,
                          0.04824907063193599,
                          0.00042572107679078933,
                          0.9513252082912732,
                          0.04824907063193599,
                          0.00042572107679078933,
                          0.00042572107679078933,
                          0.04824907063193599};
      std::vector<T> w
          = {0.008487631220880677, 0.005351171058397101, 0.039859851484174605,
             0.034353556733050614, 0.012316862645443162, 0.012316862645443162,
             0.012316862645443162, 0.012316862645443162, 0.01052919316272442,
             0.01052919316272442,  0.01052919316272442,  0.01052919316272442,
             0.012400043129816093, 0.012400043129816093, 0.012400043129816093,
             0.012400043129816093, 0.02462746155897562,  0.02462746155897562,
             0.02462746155897562,  0.02462746155897562,  0.001446720212248291,
             0.001446720212248291, 0.001446720212248291, 0.001446720212248291};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 7)
    {
      // Witherden-Vincent, 31 points, degree 7
      std::vector<T> x
          = {0.49997725989589864,  0.49997725989589864,  4.548020820275838e-05,
             0.3032085616211729,   0.3032085616211729,   0.3935828767576542,
             0.08065860859157423,  0.08065860859157423,  0.8386827828168515,
             0.8987160465605241,   0.4666666666666667,   0.06666666666666667,
             0.4666666666666667,   0.8987160465605241,   0.06666666666666667,
             0.03461728677280932,  0.4666666666666667,   0.06666666666666667,
             0.4666666666666667,   0.03461728677280932,  0.06666666666666667,
             0.6419400332575171,   0.3333333333333333,   0.3333333333333333,
             0.3333333333333333,   0.6419400332575171,   0.3333333333333333,
             0.024726633409149514, 0.3333333333333333,   0.3333333333333333,
             0.3333333333333333,   0.024726633409149514, 0.3333333333333333,
             0.6124329856642833,   0.6124329856642833,   0.1292864095395495,
             0.6124329856642833,   0.25828060479616716,  0.1292864095395495,
             0.25828060479616716,  0.6124329856642833,   0.1292864095395495,
             0.25828060479616716,  0.25828060479616716,  0.1292864095395495,
             0.8612936599601548,   0.8612936599601548,   0.08013853017977805,
             0.8612936599601548,   0.05856780986006707,  0.08013853017977805,
             0.05856780986006707,  0.8612936599601548,   0.08013853017977805,
             0.05856780986006707,  0.05856780986006707,  0.08013853017977805,
             0.32430770837562867,  0.32430770837562867,  0.6055199301105999,
             0.32430770837562867,  0.07017236151377143,  0.6055199301105999,
             0.07017236151377143,  0.32430770837562867,  0.6055199301105999,
             0.07017236151377143,  0.07017236151377143,  0.6055199301105999,
             0.8071525538155135,   0.8071525538155135,   8.975830099045535e-11,
             0.8071525538155135,   0.1928474460947282,   8.975830099045535e-11,
             0.1928474460947282,   0.8071525538155135,   8.975830099045535e-11,
             0.1928474460947282,   0.1928474460947282,   8.975830099045535e-11,
             0.617144811379889,    0.617144811379889,    0.2905430754945975,
             0.617144811379889,    0.09231211312551346,  0.2905430754945975,
             0.09231211312551346,  0.617144811379889,    0.2905430754945975,
             0.09231211312551346,  0.09231211312551346,  0.2905430754945975};
      std::vector<T> w
          = {0.008333525678065659, 0.03351006089996436,  0.005235724809005776,
             0.008897253097667639, 0.008897253097667639, 0.008897253097667639,
             0.008897253097667639, 0.0095703125,         0.0095703125,
             0.0095703125,         0.0095703125,         0.02198672904999243,
             0.02198672904999243,  0.02198672904999243,  0.02198672904999243,
             0.004928712652078553, 0.004928712652078553, 0.004928712652078553,
             0.004928712652078553, 0.009839703231407959, 0.009839703231407959,
             0.009839703231407959, 0.009839703231407959, 0.004434614989874702,
             0.004434614989874702, 0.004434614989874702, 0.004434614989874702,
             0.011906179965553104, 0.011906179965553104, 0.011906179965553104,
             0.011906179965553104};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 8)
    {
      // Witherden-Vincent, 47 points, degree 8
      std::vector<T> x
          = {0.207597282945937,    0.207597282945937,    0.584805434108126,
             0.36427480229040526,  0.36427480229040526,  0.2714503954191895,
             0.464279784892273,    0.464279784892273,    0.07144043021545401,
             0.8980480407253096,   0.49999999633818054,  7.3236389580519935e-09,
             0.49999999633818054,  0.8980480407253096,   7.3236389580519935e-09,
             0.10195195195105143,  0.49999999633818054,  7.3236389580519935e-09,
             0.49999999633818054,  0.10195195195105143,  7.3236389580519935e-09,
             0.5713526290703906,   0.2889081496994707,   0.4221837006010586,
             0.2889081496994707,   0.5713526290703906,   0.4221837006010586,
             0.006463670328550765, 0.2889081496994707,   0.4221837006010586,
             0.2889081496994707,   0.006463670328550765, 0.4221837006010586,
             0.8026293481261989,   0.4261180013517252,   0.14776399729654968,
             0.4261180013517252,   0.8026293481261989,   0.14776399729654968,
             0.04960665457725138,  0.4261180013517252,   0.14776399729654968,
             0.4261180013517252,   0.04960665457725138,  0.14776399729654968,
             0.28656254726840114,  0.28656254726840114,  0.6734528479254847,
             0.28656254726840114,  0.03998460480611414,  0.6734528479254847,
             0.03998460480611414,  0.28656254726840114,  0.6734528479254847,
             0.03998460480611414,  0.03998460480611414,  0.6734528479254847,
             0.7441593637552427,   0.7441593637552427,   0.24557080503625056,
             0.7441593637552427,   0.010269831208506702, 0.24557080503625056,
             0.010269831208506702, 0.7441593637552427,   0.24557080503625056,
             0.010269831208506702, 0.010269831208506702, 0.24557080503625056,
             0.07091616001046556,  0.07091616001046556,  0.8582503626505816,
             0.07091616001046556,  0.0708334773389528,   0.8582503626505816,
             0.0708334773389528,   0.07091616001046556,  0.8582503626505816,
             0.0708334773389528,   0.0708334773389528,   0.8582503626505816,
             0.6757966627378701,   0.6757966627378701,   0.047403231619487535,
             0.6757966627378701,   0.2768001056426424,   0.047403231619487535,
             0.2768001056426424,   0.6757966627378701,   0.047403231619487535,
             0.2768001056426424,   0.2768001056426424,   0.047403231619487535,
             0.48156335412954193,  0.48156335412954193,  0.4021857062205061,
             0.48156335412954193,  0.11625093964995196,  0.4021857062205061,
             0.11625093964995196,  0.48156335412954193,  0.4021857062205061,
             0.11625093964995196,  0.11625093964995196,  0.4021857062205061,
             0.6465552114772145,   0.6465552114772145,   0.1903148210864091,
             0.6465552114772145,   0.16312996743637637,  0.1903148210864091,
             0.16312996743637637,  0.6465552114772145,   0.1903148210864091,
             0.16312996743637637,  0.16312996743637637,  0.1903148210864091,
             0.7929644966884049,   0.9246901961338897,   0.0454758042237511,
             0.9246901961338897,   0.7929644966884049,   0.0454758042237511,
             0.7929644966884049,   0.02983399964235917,  0.0454758042237511,
             0.02983399964235917,  0.7929644966884049,   0.0454758042237511,
             0.161559699087844,    0.9246901961338897,   0.0454758042237511,
             0.9246901961338897,   0.161559699087844,    0.0454758042237511,
             0.161559699087844,    0.02983399964235917,  0.0454758042237511,
             0.02983399964235917,  0.161559699087844,    0.0454758042237511};
      std::vector<T> w = {
          0.01648033171859313,   0.029118229813715344,  0.012820192498952318,
          0.0031253531333758286, 0.0031253531333758286, 0.0031253531333758286,
          0.0031253531333758286, 0.0054370064054788565, 0.0054370064054788565,
          0.0054370064054788565, 0.0054370064054788565, 0.01074657402369029,
          0.01074657402369029,   0.01074657402369029,   0.01074657402369029,
          0.004211699873961774,  0.004211699873961774,  0.004211699873961774,
          0.004211699873961774,  0.0020718483420383515, 0.0020718483420383515,
          0.0020718483420383515, 0.0020718483420383515, 0.0008710842778830959,
          0.0008710842778830959, 0.0008710842778830959, 0.0008710842778830959,
          0.009774989398347,     0.009774989398347,     0.009774989398347,
          0.009774989398347,     0.012119794754724895,  0.012119794754724895,
          0.012119794754724895,  0.012119794754724895,  0.013660686981973735,
          0.013660686981973735,  0.013660686981973735,  0.013660686981973735,
          0.0033548038170221547, 0.0033548038170221547, 0.0033548038170221547,
          0.0033548038170221547, 0.0033548038170221547, 0.0033548038170221547,
          0.0033548038170221547, 0.0033548038170221547};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 9)
    {
      // Witherden-Vincent, 62 points, degree 9
      std::vector<T> x
          = {0.48142703453424723,  0.48142703453424723,  0.037145930931505554,
             0.15289986841483696,  0.15289986841483696,  0.6942002631703261,
             0.962910045817857,    0.4999999978874357,   4.225128545167222e-09,
             0.4999999978874357,   0.962910045817857,    4.225128545167222e-09,
             0.037089949957014465, 0.4999999978874357,   4.225128545167222e-09,
             0.4999999978874357,   0.037089949957014465, 4.225128545167222e-09,
             0.7031826037981533,   0.36513410773997146,  0.2697317845200571,
             0.36513410773997146,  0.7031826037981533,   0.2697317845200571,
             0.027085611681789527, 0.36513410773997146,  0.2697317845200571,
             0.36513410773997146,  0.027085611681789527, 0.2697317845200571,
             0.766499422258796,    0.4583333333333333,   0.08333333333333333,
             0.4583333333333333,   0.766499422258796,    0.08333333333333333,
             0.15016724440787071,  0.4583333333333333,   0.08333333333333333,
             0.4583333333333333,   0.15016724440787071,  0.08333333333333333,
             0.4492766794196984,   0.23329109477710827,  0.5334178104457834,
             0.23329109477710827,  0.4492766794196984,   0.5334178104457834,
             0.017305510134518132, 0.23329109477710827,  0.5334178104457834,
             0.23329109477710827,  0.017305510134518132, 0.5334178104457834,
             0.2899069024577096,   0.2899069024577096,   0.6619880087153507,
             0.2899069024577096,   0.04810508882693973,  0.6619880087153507,
             0.04810508882693973,  0.2899069024577096,   0.6619880087153507,
             0.04810508882693973,  0.04810508882693973,  0.6619880087153507,
             0.845937226548863,    0.845937226548863,    0.15220470980546366,
             0.845937226548863,    0.001858063645673257, 0.15220470980546366,
             0.001858063645673257, 0.845937226548863,    0.15220470980546366,
             0.001858063645673257, 0.001858063645673257, 0.15220470980546366,
             0.4028507010856899,   0.4028507010856899,   0.4222655365894455,
             0.4028507010856899,   0.17488376232486463,  0.4222655365894455,
             0.17488376232486463,  0.4028507010856899,   0.4222655365894455,
             0.17488376232486463,  0.17488376232486463,  0.4222655365894455,
             0.5530103036180599,   0.5530103036180599,   0.39676436989889025,
             0.5530103036180599,   0.050225326483049816, 0.39676436989889025,
             0.050225326483049816, 0.5530103036180599,   0.39676436989889025,
             0.050225326483049816, 0.050225326483049816, 0.39676436989889025,
             0.5298401765231302,   0.5298401765231302,   0.20082072193717074,
             0.5298401765231302,   0.2693391015396991,   0.20082072193717074,
             0.2693391015396991,   0.5298401765231302,   0.20082072193717074,
             0.2693391015396991,   0.2693391015396991,   0.20082072193717074,
             0.11327525115354325,  0.11327525115354325,  0.8661453707796378,
             0.11327525115354325,  0.02057937806681896,  0.8661453707796378,
             0.02057937806681896,  0.11327525115354325,  0.8661453707796378,
             0.02057937806681896,  0.02057937806681896,  0.8661453707796378,
             0.7318560777891165,   0.7318560777891165,   0.019504012492411515,
             0.7318560777891165,   0.248639909718472,    0.019504012492411515,
             0.248639909718472,    0.7318560777891165,   0.019504012492411515,
             0.248639909718472,    0.248639909718472,    0.019504012492411515,
             0.6770084519508361,   0.6770084519508361,   0.2131686017065601,
             0.6770084519508361,   0.10982294634260374,  0.2131686017065601,
             0.10982294634260374,  0.6770084519508361,   0.2131686017065601,
             0.10982294634260374,  0.10982294634260374,  0.2131686017065601,
             0.9155235971329332,   0.9155235971329332,   0.015347797248009717,
             0.9155235971329332,   0.06912860561905709,  0.015347797248009717,
             0.06912860561905709,  0.9155235971329332,   0.015347797248009717,
             0.06912860561905709,  0.06912860561905709,  0.015347797248009717,
             0.881922267392206,    0.6904286898164065,   0.08333333333333333,
             0.6904286898164065,   0.881922267392206,    0.08333333333333333,
             0.881922267392206,    0.2262379768502602,   0.08333333333333333,
             0.2262379768502602,   0.881922267392206,    0.08333333333333333,
             0.034744399274460665, 0.6904286898164065,   0.08333333333333333,
             0.6904286898164065,   0.034744399274460665, 0.08333333333333333,
             0.034744399274460665, 0.2262379768502602,   0.08333333333333333,
             0.2262379768502602,   0.034744399274460665, 0.08333333333333333};
      std::vector<T> w = {
          0.01068262304511417,   0.007931073648359366,  0.0013911241009981403,
          0.0013911241009981403, 0.0013911241009981403, 0.0013911241009981403,
          0.007431263786539323,  0.007431263786539323,  0.007431263786539323,
          0.007431263786539323,  0.009709340925984385,  0.009709340925984385,
          0.009709340925984385,  0.009709340925984385,  0.0038472688494336754,
          0.0038472688494336754, 0.0038472688494336754, 0.0038472688494336754,
          0.0033668582757865784, 0.0033668582757865784, 0.0033668582757865784,
          0.0033668582757865784, 0.0006863260671401915, 0.0006863260671401915,
          0.0006863260671401915, 0.0006863260671401915, 0.01141522958857261,
          0.01141522958857261,   0.01141522958857261,   0.01141522958857261,
          0.004303137880632389,  0.004303137880632389,  0.004303137880632389,
          0.004303137880632389,  0.012370760382611149,  0.012370760382611149,
          0.012370760382611149,  0.012370760382611149,  0.0007395660595837028,
          0.0007395660595837028, 0.0007395660595837028, 0.0007395660595837028,
          0.005318736378731699,  0.005318736378731699,  0.005318736378731699,
          0.005318736378731699,  0.007689106042442535,  0.007689106042442535,
          0.007689106042442535,  0.007689106042442535,  0.0015331335242327131,
          0.0015331335242327131, 0.0015331335242327131, 0.0015331335242327131,
          0.004439028648637928,  0.004439028648637928,  0.004439028648637928,
          0.004439028648637928,  0.004439028648637928,  0.004439028648637928,
          0.004439028648637928,  0.004439028648637928};
      return {std::move(x), std::move(w)};
    }
    else if (m <= 10)
    {
      // Witherden-Vincent, 83 points, degree 10
      std::vector<T> x
          = {0.04160210911043636,  0.04160210911043636,  0.9167957817791272,
             0.3244315968298237,   0.3244315968298237,   0.3511368063403526,
             0.12904321601835003,  0.12904321601835003,  0.7419135679633,
             0.9289842793673629,   0.4684192013600428,   0.06316159727991448,
             0.4684192013600428,   0.9289842793673629,   0.06316159727991448,
             0.007854123352722583, 0.4684192013600428,   0.06316159727991448,
             0.4684192013600428,   0.007854123352722583, 0.06316159727991448,
             0.6389822366317607,   0.4116347944191338,   0.1767304111617324,
             0.4116347944191338,   0.6389822366317607,   0.1767304111617324,
             0.18428735220650694,  0.4116347944191338,   0.1767304111617324,
             0.4116347944191338,   0.18428735220650694,  0.1767304111617324,
             0.37372658689439636,  0.19895309464604302,  0.602093810707914,
             0.19895309464604302,  0.37372658689439636,  0.602093810707914,
             0.02417960239768968,  0.19895309464604302,  0.602093810707914,
             0.19895309464604302,  0.02417960239768968,  0.602093810707914,
             0.438320733115388,    0.438320733115388,    0.5270494697681531,
             0.438320733115388,    0.034629797116458916, 0.5270494697681531,
             0.034629797116458916, 0.438320733115388,    0.5270494697681531,
             0.034629797116458916, 0.034629797116458916, 0.5270494697681531,
             0.5111752245039625,   0.5111752245039625,   0.3491104418985491,
             0.5111752245039625,   0.13971433359748836,  0.3491104418985491,
             0.13971433359748836,  0.5111752245039625,   0.3491104418985491,
             0.13971433359748836,  0.13971433359748836,  0.3491104418985491,
             0.6463320118839744,   0.6463320118839744,   0.0069339728754634015,
             0.6463320118839744,   0.3467340152405622,   0.0069339728754634015,
             0.3467340152405622,   0.6463320118839744,   0.0069339728754634015,
             0.3467340152405622,   0.3467340152405622,   0.0069339728754634015,
             0.191480968310491,    0.191480968310491,    0.7752643179331966,
             0.191480968310491,    0.03325471375631245,  0.7752643179331966,
             0.03325471375631245,  0.191480968310491,    0.7752643179331966,
             0.03325471375631245,  0.03325471375631245,  0.7752643179331966,
             0.3174664584028288,   0.3174664584028288,   0.5470650181463391,
             0.3174664584028288,   0.1354685234508321,   0.5470650181463391,
             0.1354685234508321,   0.3174664584028288,   0.5470650181463391,
             0.1354685234508321,   0.1354685234508321,   0.5470650181463391,
             0.8863428691627904,   0.8863428691627904,   0.0728984114756605,
             0.8863428691627904,   0.0407587193615491,   0.0728984114756605,
             0.0407587193615491,   0.8863428691627904,   0.0728984114756605,
             0.0407587193615491,   0.0407587193615491,   0.0728984114756605,
             0.6816235330697941,   0.6816235330697941,   0.2836223397977548,
             0.6816235330697941,   0.03475412713245114,  0.2836223397977548,
             0.03475412713245114,  0.6816235330697941,   0.2836223397977548,
             0.03475412713245114,  0.03475412713245114,  0.2836223397977548,
             0.7638870038079971,   0.7638870038079971,   0.059813021792726545,
             0.7638870038079971,   0.17629997439927633,  0.059813021792726545,
             0.17629997439927633,  0.7638870038079971,   0.059813021792726545,
             0.17629997439927633,  0.17629997439927633,  0.059813021792726545,
             0.5789655900015964,   0.5789655900015964,   0.06583087998062333,
             0.5789655900015964,   0.35520353001778027,  0.06583087998062333,
             0.35520353001778027,  0.5789655900015964,   0.06583087998062333,
             0.35520353001778027,  0.35520353001778027,  0.06583087998062333,
             0.8591577265975235,   0.5987335883051161,   0.027250255274673948,
             0.5987335883051161,   0.8591577265975235,   0.027250255274673948,
             0.8591577265975235,   0.37401615642020997,  0.027250255274673948,
             0.37401615642020997,  0.8591577265975235,   0.027250255274673948,
             0.1135920181278025,   0.5987335883051161,   0.027250255274673948,
             0.5987335883051161,   0.1135920181278025,   0.027250255274673948,
             0.1135920181278025,   0.37401615642020997,  0.027250255274673948,
             0.37401615642020997,  0.1135920181278025,   0.027250255274673948,
             0.6219309699024131,   0.793463374937414,    0.1618000129270508,
             0.793463374937414,    0.6219309699024131,   0.1618000129270508,
             0.6219309699024131,   0.04473661213553521,  0.1618000129270508,
             0.04473661213553521,  0.6219309699024131,   0.1618000129270508,
             0.21626901717053607,  0.793463374937414,    0.1618000129270508,
             0.793463374937414,    0.21626901717053607,  0.1618000129270508,
             0.21626901717053607,  0.04473661213553521,  0.1618000129270508,
             0.04473661213553521,  0.21626901717053607,  0.1618000129270508,
             0.35613600234607945,  0.6209931737689114,   0.3584705635890508,
             0.6209931737689114,   0.35613600234607945,  0.3584705635890508,
             0.35613600234607945,  0.020536262642037832, 0.3584705635890508,
             0.020536262642037832, 0.35613600234607945,  0.3584705635890508,
             0.28539343406486983,  0.6209931737689114,   0.3584705635890508,
             0.6209931737689114,   0.28539343406486983,  0.3584705635890508,
             0.28539343406486983,  0.020536262642037832, 0.3584705635890508,
             0.020536262642037832, 0.28539343406486983,  0.3584705635890508,
             0.9673892530013917,   0.815061231500827,    0.00744064107688475,
             0.815061231500827,    0.9673892530013917,   0.00744064107688475,
             0.9673892530013917,   0.17749812742228827,  0.00744064107688475,
             0.17749812742228827,  0.9673892530013917,   0.00744064107688475,
             0.02517010592172358,  0.815061231500827,    0.00744064107688475,
             0.815061231500827,    0.02517010592172358,  0.00744064107688475,
             0.02517010592172358,  0.17749812742228827,  0.00744064107688475,
             0.17749812742228827,  0.02517010592172358,  0.00744064107688475};
      std::vector<T> w = {
          0.0008004089208404736, 0.01563864263658789,   0.003282220249509718,
          0.0018924641059770999, 0.0018924641059770999, 0.0018924641059770999,
          0.0018924641059770999, 0.013740419913938115,  0.013740419913938115,
          0.013740419913938115,  0.013740419913938115,  0.0030695844226772445,
          0.0030695844226772445, 0.0030695844226772445, 0.0030695844226772445,
          0.0023561320550230212, 0.0023561320550230212, 0.0023561320550230212,
          0.0023561320550230212, 0.009943561942124998,  0.009943561942124998,
          0.009943561942124998,  0.009943561942124998,  0.0019566642672722606,
          0.0019566642672722606, 0.0019566642672722606, 0.0019566642672722606,
          0.0016514551449229628, 0.0016514551449229628, 0.0016514551449229628,
          0.0016514551449229628, 0.005310312423889462,  0.005310312423889462,
          0.005310312423889462,  0.005310312423889462,  0.0016971758283943043,
          0.0016971758283943043, 0.0016971758283943043, 0.0016971758283943043,
          0.0022819311151029273, 0.0022819311151029273, 0.0022819311151029273,
          0.0022819311151029273, 0.005012673460122121,  0.005012673460122121,
          0.005012673460122121,  0.005012673460122121,  0.005205054374598072,
          0.005205054374598072,  0.005205054374598072,  0.005205054374598072,
          0.0029697866889659918, 0.0029697866889659918, 0.0029697866889659918,
          0.0029697866889659918, 0.0029697866889659918, 0.0029697866889659918,
          0.0029697866889659918, 0.0029697866889659918, 0.005597240486617381,
          0.005597240486617381,  0.005597240486617381,  0.005597240486617381,
          0.005597240486617381,  0.005597240486617381,  0.005597240486617381,
          0.005597240486617381,  0.002735249701767929,  0.002735249701767929,
          0.002735249701767929,  0.002735249701767929,  0.002735249701767929,
          0.002735249701767929,  0.002735249701767929,  0.002735249701767929,
          0.0008405162864268107, 0.0008405162864268107, 0.0008405162864268107,
          0.0008405162864268107, 0.0008405162864268107, 0.0008405162864268107,
          0.0008405162864268107, 0.0008405162864268107};
      return {std::move(x), std::move(w)};
    }
    else
    {
      throw std::runtime_error(
          "Witherden-Vincent not implemented for this order.");
    }
  }
  default:
    throw std::runtime_error(
        "Witherden-Vincent not implemented for this cell type.");
  }
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
    else if (m == 3)
      return type::gauss_jacobi;
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
    case quadrature::type::witherden_vincent:
      return make_witherden_vincent_quadrature<T>(celltype, m);
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
                         [](auto x) { return static_cast<T>(0.5 + 0.5 * x); });
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
