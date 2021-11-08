// Copyright (c) 2020 Chris Richardson and Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "quadrature.h"
#include "math.h"
#include <array>
#include <cmath>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xspan.hpp>

using namespace xt::placeholders; // required for `_` to work

using namespace basix;

namespace
{

//----------------------------------------------------------------------------
std::array<std::vector<double>, 2> rec_jacobi(int N, double a, double b)
{
  // Generate the recursion coefficients alpha_k, beta_k

  // P_{k+1}(x) = (x-alpha_k)*P_{k}(x) - beta_k P_{k-1}(x)

  // for the Jacobi polynomials which are orthogonal on [-1,1]
  // with respect to the weight w(x)=[(1-x)^a]*[(1+x)^b]

  // Inputs:
  // N - polynomial order
  // a - weight parameter
  // b - weight parameter

  // Outputs:
  // alpha - recursion coefficients
  // beta - recursion coefficients

  // Adapted from the MATLAB code by Dirk Laurie and Walter Gautschi
  // http://www.cs.purdue.edu/archives/2002/wxg/codes/r_jacobi.m

  double nu = (b - a) / (a + b + 2.0);
  double mu = std::pow(2.0, (a + b + 1)) * std::tgamma(a + 1.0)
              * std::tgamma(b + 1.0) / std::tgamma(a + b + 2.0);

  std::vector<double> alpha(N), beta(N);
  alpha[0] = nu;
  beta[0] = mu;

  auto n = xt::linspace<double>(1.0, N - 1, N - 1);
  auto nab = 2.0 * n + a + b;

  auto _alpha = xt::adapt(alpha);
  auto _beta = xt::adapt(beta);
  xt::view(_alpha, xt::range(1, _)) = (b * b - a * a) / (nab * (nab + 2.0));
  xt::view(_beta, xt::range(1, _)) = 4 * (n + a) * (n + b) * n * (n + a + b)
                                     / (nab * nab * (nab + 1.0) * (nab - 1.0));

  return {std::move(alpha), std::move(beta)};
}
//----------------------------------------------------------------------------
std::array<std::vector<double>, 2> gauss(const std::vector<double>& alpha,
                                         const std::vector<double>& beta)
{
  // Compute the Gauss nodes and weights from the recursion
  // coefficients associated with a set of orthogonal polynomials
  //
  //  Inputs:
  //  alpha - recursion coefficients
  //  beta - recursion coefficients
  //
  // Outputs:
  // x - quadrature nodes
  // w - quadrature weights
  //
  // Adapted from the MATLAB code by Walter Gautschi
  // http://www.cs.purdue.edu/archives/2002/wxg/codes/gauss.m

  auto _alpha = xt::adapt(alpha);
  auto _beta = xt::adapt(beta);

  auto tmp = xt::sqrt(xt::view(_beta, xt::range(1, _)));

  xt::xtensor<double, 2> A
      = xt::diag(_alpha) + xt::diag(tmp, 1) + xt::diag(tmp, -1);

  auto [evals, evecs] = math::eigh(A);

  std::vector<double> x(evals.shape(0)), w(evals.shape(0));
  xt::adapt(x) = evals;
  xt::adapt(w) = beta[0] * xt::square(xt::row(evecs, 0));
  return {std::move(x), std::move(w)};
}
//----------------------------------------------------------------------------
std::array<std::vector<double>, 2> lobatto(const std::vector<double>& alpha,
                                           const std::vector<double>& beta,
                                           double xl1, double xl2)
{
  // Compute the Lobatto nodes and weights with the preassigned
  // nodes xl1,xl2
  //
  // Inputs:
  //   alpha - recursion coefficients
  //   beta - recursion coefficients
  //   xl1 - assigned node location
  //   xl2 - assigned node location

  // Outputs:
  // x - quadrature nodes
  // w - quadrature weights

  // Based on the section 7 of the paper
  // "Some modified matrix eigenvalue problems"
  // by Gene Golub, SIAM Review Vol 15, No. 2, April 1973, pp.318--334

  assert(alpha.size() == beta.size());

  // Solve tridiagonal system using Thomas algorithm
  double g1(0.0), g2(0.0);
  const std::size_t n = alpha.size();
  for (std::size_t i = 1; i < n - 1; ++i)
  {
    g1 = std::sqrt(beta[i]) / (alpha[i] - xl1 - std::sqrt(beta[i - 1]) * g1);
    g2 = std::sqrt(beta[i]) / (alpha[i] - xl2 - std::sqrt(beta[i - 1]) * g2);
  }
  g1 = 1.0 / (alpha[n - 1] - xl1 - std::sqrt(beta[n - 2]) * g1);
  g2 = 1.0 / (alpha[n - 1] - xl2 - std::sqrt(beta[n - 2]) * g2);

  std::vector<double> alpha_l = alpha;
  alpha_l[n - 1] = (g1 * xl2 - g2 * xl1) / (g1 - g2);
  std::vector<double> beta_l = beta;
  beta_l[n - 1] = (xl2 - xl1) / (g1 - g2);

  return gauss(alpha_l, beta_l);
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> compute_jacobi_deriv(double a, std::size_t n,
                                            std::size_t nderiv,
                                            const xtl::span<const double>& x)
{
  // Evaluate the nth Jacobi polynomial and derivatives with weight
  // parameters (a, 0) at points x
  // @param[in] a Jacobi weight a
  // @param[in] n Order of polynomial
  // @param[in] nderiv Number of derivatives (if zero, just compute
  // polynomial itself)
  // @param[in] x Points at which to evaluate
  // @return Array of polynomial derivative values (rows) at points
  // (columns)

  std::vector<std::size_t> shape = {x.size()};
  const auto _x = xt::adapt(x.data(), x.size(), xt::no_ownership(), shape);
  xt::xtensor<double, 3> J({nderiv + 1, n + 1, x.size()});
  xt::xtensor<double, 2> Jd({n + 1, x.size()});

  for (std::size_t i = 0; i < nderiv + 1; ++i)
  {
    if (i == 0)
      xt::row(Jd, 0) = 1.0;
    else
      xt::row(Jd, 0) = 0.0;

    if (n > 0)
    {
      if (i == 0)
        xt::row(Jd, 1) = (_x * (a + 2.0) + a) * 0.5;
      else if (i == 1)
        xt::row(Jd, 1) = a * 0.5 + 1;
      else
        xt::row(Jd, 1) = 0.0;
    }

    for (std::size_t k = 2; k < n + 1; ++k)
    {
      const double a1 = 2 * k * (k + a) * (2 * k + a - 2);
      const double a2 = (2 * k + a - 1) * (a * a) / a1;
      const double a3 = (2 * k + a - 1) * (2 * k + a) / (2 * k * (k + a));
      const double a4 = 2 * (k + a - 1) * (k - 1) * (2 * k + a) / a1;
      xt::row(Jd, k)
          = xt::row(Jd, k - 1) * (_x * a3 + a2) - xt::row(Jd, k - 2) * a4;
      if (i > 0)
        xt::row(Jd, k) += i * a3 * xt::view(J, i - 1, k - 1, xt::all());
    }
    // Note: using assign, instead of copy assignment,  to get around an xtensor
    // bug with Intel Compilers
    // https://github.com/xtensor-stack/xtensor/issues/2351
    auto J_view = xt::view(J, i, xt::all(), xt::all());
    J_view.assign(Jd);
  }

  xt::xtensor<double, 2> result({nderiv + 1, x.size()});
  for (std::size_t i = 0; i < nderiv + 1; ++i)
    xt::row(result, i) = xt::view(J, i, n, xt::all());

  return result;
}
//-----------------------------------------------------------------------------
std::vector<double> compute_gauss_jacobi_points(double a, int m)
{
  /// Computes the m roots of \f$P_{m}^{a,0}\f$ on [-1,1] by Newton's method.
  ///    The initial guesses are the Chebyshev points.  Algorithm
  ///    implemented from the pseudocode given by Karniadakis and
  ///    Sherwin

  const double eps = 1.e-8;
  const int max_iter = 100;
  std::vector<double> x(m);
  for (int k = 0; k < m; ++k)
  {
    // Initial guess
    x[k] = -cos((2.0 * k + 1.0) * M_PI / (2.0 * m));
    if (k > 0)
      x[k] = 0.5 * (x[k] + x[k - 1]);

    int j = 0;
    while (j < max_iter)
    {
      double s = 0;
      for (int i = 0; i < k; ++i)
        s += 1.0 / (x[k] - x[i]);
      xtl::span<const double> _x(&x[k], 1);
      const xt::xtensor<double, 2> f = compute_jacobi_deriv(a, m, 1, _x);
      const double delta = f(0, 0) / (f(1, 0) - f(0, 0) * s);
      x[k] -= delta;

      if (std::abs(delta) < eps)
        break;
      ++j;
    }
  }

  return x;
}
//-----------------------------------------------------------------------------
std::pair<xt::xarray<double>, std::vector<double>>
compute_gauss_jacobi_rule(double a, int m)
{
  /// @note Computes on [-1, 1]
  std::vector<double> _pts = compute_gauss_jacobi_points(a, m);
  auto pts = xt::adapt(_pts);

  const xt::xtensor<double, 1> Jd
      = xt::row(compute_jacobi_deriv(a, m, 1, pts), 1);

  const double a1 = std::pow(2.0, a + 1.0);

  std::vector<double> wts(m);
  for (int i = 0; i < m; ++i)
  {
    const double x = pts[i];
    const double f = Jd[i];
    wts[i] = a1 / (1.0 - x * x) / (f * f);
  }

  return {pts, wts};
}
//-----------------------------------------------------------------------------
std::pair<xt::xarray<double>, std::vector<double>> make_quadrature_line(int m)
{
  auto [ptx, wx] = compute_gauss_jacobi_rule(0.0, m);
  std::transform(wx.begin(), wx.end(), wx.begin(),
                 [](auto x) { return 0.5 * x; });
  return {0.5 * (ptx + 1.0), wx};
}
//-----------------------------------------------------------------------------
std::pair<xt::xarray<double>, std::vector<double>>
make_quadrature_triangle_collapsed(std::size_t m)
{
  auto [ptx, wx] = compute_gauss_jacobi_rule(0.0, m);
  auto [pty, wy] = compute_gauss_jacobi_rule(1.0, m);
  xt::xtensor<double, 2> pts({m * m, 2});
  std::vector<double> wts(m * m);
  int c = 0;
  for (std::size_t i = 0; i < m; ++i)
  {
    for (std::size_t j = 0; j < m; ++j)
    {
      pts(c, 0) = 0.25 * (1.0 + ptx[i]) * (1.0 - pty[j]);
      pts(c, 1) = 0.5 * (1.0 + pty[j]);
      wts[c] = wx[i] * wy[j] * 0.125;
      ++c;
    }
  }

  return {pts, wts};
}
//-----------------------------------------------------------------------------
std::pair<xt::xarray<double>, std::vector<double>>
make_quadrature_tetrahedron_collapsed(std::size_t m)
{
  auto [ptx, wx] = compute_gauss_jacobi_rule(0.0, m);
  auto [pty, wy] = compute_gauss_jacobi_rule(1.0, m);
  auto [ptz, wz] = compute_gauss_jacobi_rule(2.0, m);

  xt::xtensor<double, 2> pts({m * m * m, 3});
  std::vector<double> wts(m * m * m);
  int c = 0;
  for (std::size_t i = 0; i < m; ++i)
  {
    for (std::size_t j = 0; j < m; ++j)
    {
      for (std::size_t k = 0; k < m; ++k)
      {
        pts(c, 0) = 0.125 * (1.0 + ptx[i]) * (1.0 - pty[j]) * (1.0 - ptz[k]);
        pts(c, 1) = 0.25 * (1. + pty[j]) * (1. - ptz[k]);
        pts(c, 2) = 0.5 * (1.0 + ptz[k]);
        wts[c] = wx[i] * wy[j] * wz[k] * 0.125 * 0.125;
        ++c;
      }
    }
  }

  return {pts, wts};
}
//-----------------------------------------------------------------------------
std::pair<xt::xarray<double>, std::vector<double>>
make_gauss_jacobi_quadrature(cell::type celltype, std::size_t m)
{
  const std::size_t np = (m + 2) / 2;
  switch (celltype)
  {
  case cell::type::interval:
    return make_quadrature_line(np);
  case cell::type::quadrilateral:
  {
    auto [QptsL, QwtsL] = make_quadrature_line(np);
    xt::xtensor<double, 2> Qpts({np * np, 2});
    std::vector<double> Qwts(np * np);
    int c = 0;
    for (std::size_t j = 0; j < np; ++j)
    {
      for (std::size_t i = 0; i < np; ++i)
      {
        Qpts(c, 0) = QptsL[i];
        Qpts(c, 1) = QptsL[j];
        Qwts[c] = QwtsL[i] * QwtsL[j];
        ++c;
      }
    }
    return {Qpts, Qwts};
  }
  case cell::type::hexahedron:
  {
    auto [QptsL, QwtsL] = make_quadrature_line(np);
    xt::xtensor<double, 2> Qpts({np * np * np, 3});
    std::vector<double> Qwts(np * np * np);
    int c = 0;
    for (std::size_t k = 0; k < np; ++k)
    {
      for (std::size_t j = 0; j < np; ++j)
      {
        for (std::size_t i = 0; i < np; ++i)
        {
          Qpts(c, 0) = QptsL[i];
          Qpts(c, 1) = QptsL[j];
          Qpts(c, 2) = QptsL[k];
          Qwts[c] = QwtsL[i] * QwtsL[j] * QwtsL[k];
          ++c;
        }
      }
    }
    return {Qpts, Qwts};
  }
  case cell::type::prism:
  {
    auto [QptsL, QwtsL] = make_quadrature_line(np);
    auto [QptsT, QwtsT] = make_quadrature_triangle_collapsed(np);
    xt::xtensor<double, 2> Qpts({np * QptsT.shape(0), 3});
    std::vector<double> Qwts(np * QptsT.shape(0));
    int c = 0;
    for (std::size_t k = 0; k < np; ++k)
    {
      for (std::size_t i = 0; i < QptsT.shape(0); ++i)
      {
        Qpts(c, 0) = QptsT(i, 0);
        Qpts(c, 1) = QptsT(i, 1);
        Qpts(c, 2) = QptsL[k];
        Qwts[c] = QwtsT[i] * QwtsL[k];
        ++c;
      }
    }
    return {Qpts, Qwts};
  }
  case cell::type::pyramid:
    throw std::runtime_error("Pyramid not yet supported");
  case cell::type::triangle:
    return make_quadrature_triangle_collapsed(np);
  case cell::type::tetrahedron:
    return make_quadrature_tetrahedron_collapsed(np);
  default:
    throw std::runtime_error("Unsupported celltype for make_quadrature");
  }
}
//-----------------------------------------------------------------------------
std::pair<xt::xarray<double>, std::vector<double>> compute_gll_rule(int m)
{
  // Implement the Gauss-Lobatto-Legendre quadrature rules on the interval
  // using Greg von Winckel's implementation. This facilitates implementing
  // spectral elements
  // The quadrature rule uses m points for a degree of precision of 2m-3.

  if (m < 2)
  {
    throw std::runtime_error(
        "Gauss-Lobatto-Legendre quadrature invalid for fewer than 2 points");
  }

  // Calculate the recursion coefficients
  auto [alpha, beta] = rec_jacobi(m, 0.0, 0.0);

  // Compute Lobatto nodes and weights
  auto [xs_ref, ws_ref] = lobatto(alpha, beta, -1.0, 1.0);

  return {xt::adapt(xs_ref), ws_ref};
}
//-----------------------------------------------------------------------------
std::pair<xt::xarray<double>, std::vector<double>> make_gll_line(int m)
{
  auto [ptx, wx] = compute_gll_rule(m);
  std::transform(wx.begin(), wx.end(), wx.begin(),
                 [](auto x) { return 0.5 * x; });
  return {0.5 * (ptx + 1.0), wx};
}
//-----------------------------------------------------------------------------
std::pair<xt::xarray<double>, std::vector<double>>
make_gll_quadrature(cell::type celltype, std::size_t m)
{
  const std::size_t np = (m + 4) / 2;
  switch (celltype)
  {
  case cell::type::interval:
    return make_gll_line(np);
  case cell::type::quadrilateral:
  {
    auto [QptsL, QwtsL] = make_gll_line(np);
    xt::xtensor<double, 2> Qpts({np * np, 2});
    std::vector<double> Qwts(np * np);
    int c = 0;
    for (std::size_t j = 0; j < np; ++j)
    {
      for (std::size_t i = 0; i < np; ++i)
      {
        Qpts(c, 0) = QptsL[i];
        Qpts(c, 1) = QptsL[j];
        Qwts[c] = QwtsL[i] * QwtsL[j];
        ++c;
      }
    }
    return {Qpts, Qwts};
  }
  case cell::type::hexahedron:
  {
    auto [QptsL, QwtsL] = make_gll_line(np);
    xt::xtensor<double, 2> Qpts({np * np * np, 3});
    std::vector<double> Qwts(np * np * np);
    int c = 0;
    for (std::size_t k = 0; k < np; ++k)
    {
      for (std::size_t j = 0; j < np; ++j)
      {
        for (std::size_t i = 0; i < np; ++i)
        {
          Qpts(c, 0) = QptsL[i];
          Qpts(c, 1) = QptsL[j];
          Qpts(c, 2) = QptsL[k];
          Qwts[c] = QwtsL[i] * QwtsL[j] * QwtsL[k];
          ++c;
        }
      }
    }
    return {Qpts, Qwts};
  }
  case cell::type::prism:
  {
    throw std::runtime_error("Prism not yet supported");
  }
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
std::pair<xt::xarray<double>, std::vector<double>>
make_strang_fix_quadrature(cell::type celltype, std::size_t m)
{
  if (celltype == cell::type::triangle)
  {
    if (m == 2)
    {
      // Scheme from Strang and Fix, 3 points, degree of precision 2
      xt::xtensor<double, 2> x = {{1.0 / 6.0, 1.0 / 6.0},
                                  {1.0 / 6.0, 2.0 / 3.0},
                                  {2.0 / 3.0, 1.0 / 6.0}};
      return {x, {1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0}};
    }
    else if (m == 3)
    {
      // Scheme from Strang and Fix, 6 points, degree of precision 3
      xt::xtensor<double, 2> x = {{0.659027622374092, 0.231933368553031},
                                  {0.659027622374092, 0.109039009072877},
                                  {0.231933368553031, 0.659027622374092},
                                  {0.231933368553031, 0.109039009072877},
                                  {0.109039009072877, 0.659027622374092},
                                  {0.109039009072877, 0.231933368553031}};
      std::vector<double> w(6, 1.0 / 12.0);
      return {x, w};
    }
    else if (m == 4)
    {
      // Scheme from Strang and Fix, 6 points, degree of precision 4
      xt::xtensor<double, 2> x = {{0.816847572980459, 0.091576213509771},
                                  {0.091576213509771, 0.816847572980459},
                                  {0.091576213509771, 0.091576213509771},
                                  {0.108103018168070, 0.445948490915965},
                                  {0.445948490915965, 0.108103018168070},
                                  {0.445948490915965, 0.445948490915965}};
      std::vector<double> w
          = {0.109951743655322, 0.109951743655322, 0.109951743655322,
             0.223381589678011, 0.223381589678011, 0.223381589678011};
      std::transform(w.cbegin(), w.cend(), w.begin(),
                     [](auto x) { return 0.5 * x; });
      return {x, w};
    }
    else if (m == 5)
    {
      // Scheme from Strang and Fix, 7 points, degree of precision 5
      xt::xtensor<double, 2> x = {{0.33333333333333333, 0.33333333333333333},
                                  {0.79742698535308720, 0.10128650732345633},
                                  {0.10128650732345633, 0.79742698535308720},
                                  {0.10128650732345633, 0.10128650732345633},
                                  {0.05971587178976981, 0.47014206410511505},
                                  {0.47014206410511505, 0.05971587178976981},
                                  {0.47014206410511505, 0.47014206410511505}};
      std::vector<double> w
          = {0.22500000000000000, 0.12593918054482717, 0.12593918054482717,
             0.12593918054482717, 0.13239415278850616, 0.13239415278850616,
             0.13239415278850616};
      std::transform(w.cbegin(), w.cend(), w.begin(),
                     [](auto x) { return 0.5 * x; });
      return {x, w};
    }
    else if (m == 6)
    {
      // Scheme from Strang and Fix, 12 points, degree of precision 6
      xt::xtensor<double, 2> x = {{0.873821971016996, 0.063089014491502},
                                  {0.063089014491502, 0.873821971016996},
                                  {0.063089014491502, 0.063089014491502},
                                  {0.501426509658179, 0.249286745170910},
                                  {0.249286745170910, 0.501426509658179},
                                  {0.249286745170910, 0.249286745170910},
                                  {0.636502499121399, 0.310352451033785},
                                  {0.636502499121399, 0.053145049844816},
                                  {0.310352451033785, 0.636502499121399},
                                  {0.310352451033785, 0.053145049844816},
                                  {0.053145049844816, 0.636502499121399},
                                  {0.053145049844816, 0.310352451033785}};
      std::vector<double> w
          = {0.050844906370207, 0.050844906370207, 0.050844906370207,
             0.116786275726379, 0.116786275726379, 0.116786275726379,
             0.082851075618374, 0.082851075618374, 0.082851075618374,
             0.082851075618374, 0.082851075618374, 0.082851075618374};
      std::transform(w.cbegin(), w.cend(), w.begin(),
                     [](auto x) { return 0.5 * x; });
      return {x, w};
    }
    else
      throw std::runtime_error("Strang-Fix not implemented for this order.");
  }
  throw std::runtime_error("Strang-Fix not implemented for this cell type.");
}
//-------------------------------------------------------------------------------------------------------
std::pair<xt::xarray<double>, std::vector<double>>
make_zienkiewicz_taylor_quadrature(cell::type celltype, std::size_t m)
{
  if (celltype == cell::type::triangle)
  {
    if (m == 0 or m == 1)
    {
      // Scheme from Zienkiewicz and Taylor, 1 point, degree of precision 1
      return {{{1.0 / 3.0, 1.0 / 3.0}}, {0.5}};
    }
    else
      throw std::runtime_error(
          "Zienkiewicz-Taylor not implemented for this order.");
  }
  if (celltype == cell::type::tetrahedron)
  {
    if (m == 0 or m == 1)
    {
      // Scheme from Zienkiewicz and Taylor, 1 point, degree of precision 1
      return {{{0.25, 0.25, 0.25}}, {1.0 / 6.0}};
    }
    else if (m == 2)
    {
      // Scheme from Zienkiewicz and Taylor, 4 points, degree of precision 2
      constexpr double a = 0.585410196624969, b = 0.138196601125011;
      xt::xtensor<double, 2> x = {{a, b, b}, {b, a, b}, {b, b, a}, {b, b, b}};
      return {x, {1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0}};
    }
    else if (m == 3)
    {
      // Scheme from Zienkiewicz and Taylor, 5 points, degree of precision 3
      // Note : this scheme has a negative weight
      xt::xtensor<double, 2> x{
          {0.2500000000000000, 0.2500000000000000, 0.2500000000000000},
          {0.5000000000000000, 0.1666666666666666, 0.1666666666666666},
          {0.1666666666666666, 0.5000000000000000, 0.1666666666666666},
          {0.1666666666666666, 0.1666666666666666, 0.5000000000000000},
          {0.1666666666666666, 0.1666666666666666, 0.1666666666666666}};
      return {x, {-0.8 / 6.0, 0.45 / 6.0, 0.45 / 6.0, 0.45 / 6.0, 0.45 / 6.0}};
    }
    else
      throw std::runtime_error(
          "Zienkiewicz-Taylor not implemented for this order.");
  }
  throw std::runtime_error(
      "Zienkiewicz-Taylor not implemented for this cell type.");
}
//-----------------------------------------------------------------------------
std::pair<xt::xarray<double>, std::vector<double>>
make_keast_quadrature(cell::type celltype, std::size_t m)
{
  if (celltype == cell::type::tetrahedron)
  {
    if (m == 4)
    {
      // Keast rule, 14 points, degree of precision 4
      // Values taken from
      // http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
      // (KEAST5)
      xt::xtensor<double, 2> x
          = {{0.0000000000000000, 0.5000000000000000, 0.5000000000000000},
             {0.5000000000000000, 0.0000000000000000, 0.5000000000000000},
             {0.5000000000000000, 0.5000000000000000, 0.0000000000000000},
             {0.5000000000000000, 0.0000000000000000, 0.0000000000000000},
             {0.0000000000000000, 0.5000000000000000, 0.0000000000000000},
             {0.0000000000000000, 0.0000000000000000, 0.5000000000000000},
             {0.6984197043243866, 0.1005267652252045, 0.1005267652252045},
             {0.1005267652252045, 0.1005267652252045, 0.1005267652252045},
             {0.1005267652252045, 0.1005267652252045, 0.6984197043243866},
             {0.1005267652252045, 0.6984197043243866, 0.1005267652252045},
             {0.0568813795204234, 0.3143728734931922, 0.3143728734931922},
             {0.3143728734931922, 0.3143728734931922, 0.3143728734931922},
             {0.3143728734931922, 0.3143728734931922, 0.0568813795204234},
             {0.3143728734931922, 0.0568813795204234, 0.3143728734931922}};
      std::vector<double> w
          = {0.0190476190476190, 0.0190476190476190, 0.0190476190476190,
             0.0190476190476190, 0.0190476190476190, 0.0190476190476190,
             0.0885898247429807, 0.0885898247429807, 0.0885898247429807,
             0.0885898247429807, 0.1328387466855907, 0.1328387466855907,
             0.1328387466855907, 0.1328387466855907};
      std::transform(w.cbegin(), w.cend(), w.begin(),
                     [](auto x) { return x / 6.0; });
      return {x, w};
    }
    else if (m == 5)
    {
      // Keast rule, 15 points, degree of precision 5
      // Values taken from
      // http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
      // (KEAST6)
      xt::xtensor<double, 2> x
          = {{0.2500000000000000, 0.2500000000000000, 0.2500000000000000},
             {0.0000000000000000, 0.3333333333333333, 0.3333333333333333},
             {0.3333333333333333, 0.3333333333333333, 0.3333333333333333},
             {0.3333333333333333, 0.3333333333333333, 0.0000000000000000},
             {0.3333333333333333, 0.0000000000000000, 0.3333333333333333},
             {0.7272727272727273, 0.0909090909090909, 0.0909090909090909},
             {0.0909090909090909, 0.0909090909090909, 0.0909090909090909},
             {0.0909090909090909, 0.0909090909090909, 0.7272727272727273},
             {0.0909090909090909, 0.7272727272727273, 0.0909090909090909},
             {0.4334498464263357, 0.0665501535736643, 0.0665501535736643},
             {0.0665501535736643, 0.4334498464263357, 0.0665501535736643},
             {0.0665501535736643, 0.0665501535736643, 0.4334498464263357},
             {0.0665501535736643, 0.4334498464263357, 0.4334498464263357},
             {0.4334498464263357, 0.0665501535736643, 0.4334498464263357},
             {0.4334498464263357, 0.4334498464263357, 0.0665501535736643}};
      std::vector<double> w
          = {0.1817020685825351, 0.0361607142857143, 0.0361607142857143,
             0.0361607142857143, 0.0361607142857143, 0.0698714945161738,
             0.0698714945161738, 0.0698714945161738, 0.0698714945161738,
             0.0656948493683187, 0.0656948493683187, 0.0656948493683187,
             0.0656948493683187, 0.0656948493683187, 0.0656948493683187};
      std::transform(w.cbegin(), w.cend(), w.begin(),
                     [](auto x) { return x / 6.0; });
      return {x, w};
    }
    else if (m == 6)
    {
      // Keast rule, 24 points, degree of precision 6
      // Values taken from
      // http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
      // (KEAST7)
      xt::xtensor<double, 2> x
          = {{0.3561913862225449, 0.2146028712591517, 0.2146028712591517},
             {0.2146028712591517, 0.2146028712591517, 0.2146028712591517},
             {0.2146028712591517, 0.2146028712591517, 0.3561913862225449},
             {0.2146028712591517, 0.3561913862225449, 0.2146028712591517},
             {0.8779781243961660, 0.0406739585346113, 0.0406739585346113},
             {0.0406739585346113, 0.0406739585346113, 0.0406739585346113},
             {0.0406739585346113, 0.0406739585346113, 0.8779781243961660},
             {0.0406739585346113, 0.8779781243961660, 0.0406739585346113},
             {0.0329863295731731, 0.3223378901422757, 0.3223378901422757},
             {0.3223378901422757, 0.3223378901422757, 0.3223378901422757},
             {0.3223378901422757, 0.3223378901422757, 0.0329863295731731},
             {0.3223378901422757, 0.0329863295731731, 0.3223378901422757},
             {0.2696723314583159, 0.0636610018750175, 0.0636610018750175},
             {0.0636610018750175, 0.2696723314583159, 0.0636610018750175},
             {0.0636610018750175, 0.0636610018750175, 0.2696723314583159},
             {0.6030056647916491, 0.0636610018750175, 0.0636610018750175},
             {0.0636610018750175, 0.6030056647916491, 0.0636610018750175},
             {0.0636610018750175, 0.0636610018750175, 0.6030056647916491},
             {0.0636610018750175, 0.2696723314583159, 0.6030056647916491},
             {0.2696723314583159, 0.6030056647916491, 0.0636610018750175},
             {0.6030056647916491, 0.0636610018750175, 0.2696723314583159},
             {0.0636610018750175, 0.6030056647916491, 0.2696723314583159},
             {0.2696723314583159, 0.0636610018750175, 0.6030056647916491},
             {0.6030056647916491, 0.2696723314583159, 0.0636610018750175}};
      std::vector<double> w
          = {0.0399227502581679, 0.0399227502581679, 0.0399227502581679,
             0.0399227502581679, 0.0100772110553207, 0.0100772110553207,
             0.0100772110553207, 0.0100772110553207, 0.0553571815436544,
             0.0553571815436544, 0.0553571815436544, 0.0553571815436544,
             0.0482142857142857, 0.0482142857142857, 0.0482142857142857,
             0.0482142857142857, 0.0482142857142857, 0.0482142857142857,
             0.0482142857142857, 0.0482142857142857, 0.0482142857142857,
             0.0482142857142857, 0.0482142857142857, 0.0482142857142857};
      std::transform(w.cbegin(), w.cend(), w.begin(),
                     [](auto x) { return x / 6.0; });
      return {x, w};
    }
    else if (m == 7)
    {
      // Keast rule, 31 points, degree of precision 7
      // Values taken from
      // http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
      // (KEAST8)
      xt::xtensor<double, 2> x
          = {{0.2500000000000000, 0.2500000000000000, 0.2500000000000000},
             {0.7653604230090441, 0.0782131923303186, 0.0782131923303186},
             {0.0782131923303186, 0.0782131923303186, 0.0782131923303186},
             {0.0782131923303186, 0.0782131923303186, 0.7653604230090441},
             {0.0782131923303186, 0.7653604230090441, 0.0782131923303186},
             {0.6344703500082868, 0.1218432166639044, 0.1218432166639044},
             {0.1218432166639044, 0.1218432166639044, 0.1218432166639044},
             {0.1218432166639044, 0.1218432166639044, 0.6344703500082868},
             {0.1218432166639044, 0.6344703500082868, 0.1218432166639044},
             {0.0023825066607383, 0.3325391644464206, 0.3325391644464206},
             {0.3325391644464206, 0.3325391644464206, 0.3325391644464206},
             {0.3325391644464206, 0.3325391644464206, 0.0023825066607383},
             {0.3325391644464206, 0.0023825066607383, 0.3325391644464206},
             {0.0000000000000000, 0.5000000000000000, 0.5000000000000000},
             {0.5000000000000000, 0.0000000000000000, 0.5000000000000000},
             {0.5000000000000000, 0.5000000000000000, 0.0000000000000000},
             {0.5000000000000000, 0.0000000000000000, 0.0000000000000000},
             {0.0000000000000000, 0.5000000000000000, 0.0000000000000000},
             {0.0000000000000000, 0.0000000000000000, 0.5000000000000000},
             {0.2000000000000000, 0.1000000000000000, 0.1000000000000000},
             {0.1000000000000000, 0.2000000000000000, 0.1000000000000000},
             {0.1000000000000000, 0.1000000000000000, 0.2000000000000000},
             {0.6000000000000000, 0.1000000000000000, 0.1000000000000000},
             {0.1000000000000000, 0.6000000000000000, 0.1000000000000000},
             {0.1000000000000000, 0.1000000000000000, 0.6000000000000000},
             {0.1000000000000000, 0.2000000000000000, 0.6000000000000000},
             {0.2000000000000000, 0.6000000000000000, 0.1000000000000000},
             {0.6000000000000000, 0.1000000000000000, 0.2000000000000000},
             {0.1000000000000000, 0.6000000000000000, 0.2000000000000000},
             {0.2000000000000000, 0.1000000000000000, 0.6000000000000000},
             {0.6000000000000000, 0.2000000000000000, 0.1000000000000000}};
      std::vector<double> w
          = {0.1095853407966528,  0.0635996491464850,  0.0635996491464850,
             0.0635996491464850,  0.0635996491464850,  -0.3751064406859797,
             -0.3751064406859797, -0.3751064406859797, -0.3751064406859797,
             0.0293485515784412,  0.0293485515784412,  0.0293485515784412,
             0.0293485515784412,  0.0058201058201058,  0.0058201058201058,
             0.0058201058201058,  0.0058201058201058,  0.0058201058201058,
             0.0058201058201058,  0.1653439153439105,  0.1653439153439105,
             0.1653439153439105,  0.1653439153439105,  0.1653439153439105,
             0.1653439153439105,  0.1653439153439105,  0.1653439153439105,
             0.1653439153439105,  0.1653439153439105,  0.1653439153439105,
             0.1653439153439105};

      std::transform(w.cbegin(), w.cend(), w.begin(),
                     [](auto x) { return x / 6.0; });
      return {x, w};
    }
    else if (m == 8)
    {
      // Keast rule, 45 points, degree of precision 8
      // Values taken from
      // http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
      // (KEAST9)
      xt::xtensor<double, 2> x
          = {{0.2500000000000000, 0.2500000000000000, 0.2500000000000000},
             {0.6175871903000830, 0.1274709365666390, 0.1274709365666390},
             {0.1274709365666390, 0.1274709365666390, 0.1274709365666390},
             {0.1274709365666390, 0.1274709365666390, 0.6175871903000830},
             {0.1274709365666390, 0.6175871903000830, 0.1274709365666390},
             {0.9037635088221031, 0.0320788303926323, 0.0320788303926323},
             {0.0320788303926323, 0.0320788303926323, 0.0320788303926323},
             {0.0320788303926323, 0.0320788303926323, 0.9037635088221031},
             {0.0320788303926323, 0.9037635088221031, 0.0320788303926323},
             {0.4502229043567190, 0.0497770956432810, 0.0497770956432810},
             {0.0497770956432810, 0.4502229043567190, 0.0497770956432810},
             {0.0497770956432810, 0.0497770956432810, 0.4502229043567190},
             {0.0497770956432810, 0.4502229043567190, 0.4502229043567190},
             {0.4502229043567190, 0.0497770956432810, 0.4502229043567190},
             {0.4502229043567190, 0.4502229043567190, 0.0497770956432810},
             {0.3162695526014501, 0.1837304473985499, 0.1837304473985499},
             {0.1837304473985499, 0.3162695526014501, 0.1837304473985499},
             {0.1837304473985499, 0.1837304473985499, 0.3162695526014501},
             {0.1837304473985499, 0.3162695526014501, 0.3162695526014501},
             {0.3162695526014501, 0.1837304473985499, 0.3162695526014501},
             {0.3162695526014501, 0.3162695526014501, 0.1837304473985499},
             {0.0229177878448171, 0.2319010893971509, 0.2319010893971509},
             {0.2319010893971509, 0.0229177878448171, 0.2319010893971509},
             {0.2319010893971509, 0.2319010893971509, 0.0229177878448171},
             {0.5132800333608811, 0.2319010893971509, 0.2319010893971509},
             {0.2319010893971509, 0.5132800333608811, 0.2319010893971509},
             {0.2319010893971509, 0.2319010893971509, 0.5132800333608811},
             {0.2319010893971509, 0.0229177878448171, 0.5132800333608811},
             {0.0229177878448171, 0.5132800333608811, 0.2319010893971509},
             {0.5132800333608811, 0.2319010893971509, 0.0229177878448171},
             {0.2319010893971509, 0.5132800333608811, 0.0229177878448171},
             {0.0229177878448171, 0.2319010893971509, 0.5132800333608811},
             {0.5132800333608811, 0.0229177878448171, 0.2319010893971509},
             {0.7303134278075384, 0.0379700484718286, 0.0379700484718286},
             {0.0379700484718286, 0.7303134278075384, 0.0379700484718286},
             {0.0379700484718286, 0.0379700484718286, 0.7303134278075384},
             {0.1937464752488044, 0.0379700484718286, 0.0379700484718286},
             {0.0379700484718286, 0.1937464752488044, 0.0379700484718286},
             {0.0379700484718286, 0.0379700484718286, 0.1937464752488044},
             {0.0379700484718286, 0.7303134278075384, 0.1937464752488044},
             {0.7303134278075384, 0.1937464752488044, 0.0379700484718286},
             {0.1937464752488044, 0.0379700484718286, 0.7303134278075384},
             {0.0379700484718286, 0.1937464752488044, 0.7303134278075384},
             {0.7303134278075384, 0.0379700484718286, 0.1937464752488044},
             {0.1937464752488044, 0.7303134278075384, 0.0379700484718286}};
      std::vector<double> w
          = {-0.2359620398477557, 0.0244878963560562, 0.0244878963560562,
             0.0244878963560562,  0.0244878963560562, 0.0039485206398261,
             0.0039485206398261,  0.0039485206398261, 0.0039485206398261,
             0.0263055529507371,  0.0263055529507371, 0.0263055529507371,
             0.0263055529507371,  0.0263055529507371, 0.0263055529507371,
             0.0829803830550589,  0.0829803830550589, 0.0829803830550589,
             0.0829803830550589,  0.0829803830550589, 0.0829803830550589,
             0.0254426245481023,  0.0254426245481023, 0.0254426245481023,
             0.0254426245481023,  0.0254426245481023, 0.0254426245481023,
             0.0254426245481023,  0.0254426245481023, 0.0254426245481023,
             0.0254426245481023,  0.0254426245481023, 0.0254426245481023,
             0.0134324384376852,  0.0134324384376852, 0.0134324384376852,
             0.0134324384376852,  0.0134324384376852, 0.0134324384376852,
             0.0134324384376852,  0.0134324384376852, 0.0134324384376852,
             0.0134324384376852,  0.0134324384376852, 0.0134324384376852};
      std::transform(w.cbegin(), w.cend(), w.begin(),
                     [](auto x) { return x / 6.0; });
      return {x, w};
    }
    else
      throw std::runtime_error("Keast not implemented for this order.");
  }
  throw std::runtime_error("Keast not implemented for this cell type.");
}
//-----------------------------------------------------------------------------
std::pair<xt::xarray<double>, std::vector<double>>
make_xiao_gimbutas_quadrature(cell::type celltype, int m)
{
  if (celltype == cell::type::triangle)
  {
    if (m == 10)
    {
      // Scheme from Xiao Gimbutas, 25 points, degree of precision 10
      xt::xtensor<double, 2> x = {{0.3333333333333333, 0.3333333333333333},
                                  {0.4951734598011705, 0.4951734598011705},
                                  {0.019139415242841296, 0.019139415242841296},
                                  {0.18448501268524653, 0.18448501268524653},
                                  {0.42823482094371884, 0.42823482094371884},
                                  {0.4951734598011705, 0.009653080397658997},
                                  {0.019139415242841296, 0.9617211695143174},
                                  {0.18448501268524653, 0.6310299746295069},
                                  {0.42823482094371884, 0.14353035811256232},
                                  {0.009653080397658997, 0.4951734598011705},
                                  {0.9617211695143174, 0.019139415242841296},
                                  {0.6310299746295069, 0.18448501268524653},
                                  {0.14353035811256232, 0.42823482094371884},
                                  {0.03472362048232748, 0.13373475510086913},
                                  {0.03758272734119169, 0.3266931362813369},
                                  {0.8315416244168035, 0.03472362048232748},
                                  {0.6357241363774714, 0.03758272734119169},
                                  {0.13373475510086913, 0.8315416244168035},
                                  {0.3266931362813369, 0.6357241363774714},
                                  {0.13373475510086913, 0.03472362048232748},
                                  {0.3266931362813369, 0.03758272734119169},
                                  {0.8315416244168035, 0.13373475510086913},
                                  {0.6357241363774714, 0.3266931362813369},
                                  {0.03472362048232748, 0.8315416244168035},
                                  {0.03758272734119169, 0.6357241363774714}};
      std::vector<double> w
          = {0.08361487437397393,  0.009792590498418303, 0.006385359230118654,
             0.07863376974637727,  0.07524732796854398,  0.009792590498418303,
             0.006385359230118654, 0.07863376974637727,  0.07524732796854398,
             0.009792590498418303, 0.006385359230118654, 0.07863376974637727,
             0.07524732796854398,  0.028962281463256342, 0.038739049086018905,
             0.028962281463256342, 0.038739049086018905, 0.028962281463256342,
             0.038739049086018905, 0.028962281463256342, 0.038739049086018905,
             0.028962281463256342, 0.038739049086018905, 0.028962281463256342,
             0.038739049086018905};
      std::transform(w.cbegin(), w.cend(), w.begin(),
                     [](auto x) { return x / 2.0; });
      return {x, w};
    }
    else if (m == 11)
    {
      // Scheme from Xiao Gimbutas, 28 points, degree of precision 11
      xt::xtensor<double, 2> x = {{0.3333333333333333, 0.3333333333333333},
                                  {0.030846895635588123, 0.030846895635588123},
                                  {0.49878016517846074, 0.49878016517846074},
                                  {0.11320782728669404, 0.11320782728669404},
                                  {0.4366550163931761, 0.4366550163931761},
                                  {0.21448345861926937, 0.21448345861926937},
                                  {0.030846895635588123, 0.9383062087288238},
                                  {0.49878016517846074, 0.0024396696430785125},
                                  {0.11320782728669404, 0.7735843454266119},
                                  {0.4366550163931761, 0.12668996721364778},
                                  {0.21448345861926937, 0.5710330827614613},
                                  {0.9383062087288238, 0.030846895635588123},
                                  {0.0024396696430785125, 0.49878016517846074},
                                  {0.7735843454266119, 0.11320782728669404},
                                  {0.12668996721364778, 0.4366550163931761},
                                  {0.5710330827614613, 0.21448345861926937},
                                  {0.014366662569555624, 0.1593036198376935},
                                  {0.04766406697215078, 0.31063121631346313},
                                  {0.8263297175927509, 0.014366662569555624},
                                  {0.6417047167143861, 0.04766406697215078},
                                  {0.1593036198376935, 0.8263297175927509},
                                  {0.31063121631346313, 0.6417047167143861},
                                  {0.1593036198376935, 0.014366662569555624},
                                  {0.31063121631346313, 0.04766406697215078},
                                  {0.8263297175927509, 0.1593036198376935},
                                  {0.6417047167143861, 0.31063121631346313},
                                  {0.014366662569555624, 0.8263297175927509},
                                  {0.04766406697215078, 0.6417047167143861}};
      std::vector<double> w
          = {0.08144513470935129,  0.012249296950707964, 0.012465491873881381,
             0.04012924238130832,  0.06309487215989869,  0.06784510774369515,
             0.012249296950707964, 0.012465491873881381, 0.04012924238130832,
             0.06309487215989869,  0.06784510774369515,  0.012249296950707964,
             0.012465491873881381, 0.04012924238130832,  0.06309487215989869,
             0.06784510774369515,  0.014557623337809246, 0.04064284865588647,
             0.014557623337809246, 0.04064284865588647,  0.014557623337809246,
             0.04064284865588647,  0.014557623337809246, 0.04064284865588647,
             0.014557623337809246, 0.04064284865588647,  0.014557623337809246,
             0.04064284865588647};
      std::transform(w.cbegin(), w.cend(), w.begin(),
                     [](auto x) { return x / 2.0; });
      return {x, w};
    }
    else if (m == 12)
    {
      // Scheme from Xiao Gimbutas, 33 points, degree of precision 12
      xt::xtensor<double, 2> x = {{0.27146250701492614, 0.27146250701492614},
                                  {0.10925782765935432, 0.10925782765935432},
                                  {0.4401116486585931, 0.4401116486585931},
                                  {0.4882037509455415, 0.4882037509455415},
                                  {0.02464636343633564, 0.02464636343633564},
                                  {0.27146250701492614, 0.45707498597014773},
                                  {0.10925782765935432, 0.7814843446812914},
                                  {0.4401116486585931, 0.11977670268281382},
                                  {0.4882037509455415, 0.02359249810891695},
                                  {0.02464636343633564, 0.9507072731273287},
                                  {0.45707498597014773, 0.27146250701492614},
                                  {0.7814843446812914, 0.10925782765935432},
                                  {0.11977670268281382, 0.4401116486585931},
                                  {0.02359249810891695, 0.4882037509455415},
                                  {0.9507072731273287, 0.02464636343633564},
                                  {0.1162960196779266, 0.25545422863851736},
                                  {0.021382490256170623, 0.12727971723358936},
                                  {0.023034156355267166, 0.29165567973834094},
                                  {0.6282497516835561, 0.1162960196779266},
                                  {0.85133779251024, 0.021382490256170623},
                                  {0.6853101639063919, 0.023034156355267166},
                                  {0.25545422863851736, 0.6282497516835561},
                                  {0.12727971723358936, 0.85133779251024},
                                  {0.29165567973834094, 0.6853101639063919},
                                  {0.25545422863851736, 0.1162960196779266},
                                  {0.12727971723358936, 0.021382490256170623},
                                  {0.29165567973834094, 0.023034156355267166},
                                  {0.6282497516835561, 0.25545422863851736},
                                  {0.85133779251024, 0.12727971723358936},
                                  {0.6853101639063919, 0.29165567973834094},
                                  {0.1162960196779266, 0.6282497516835561},
                                  {0.021382490256170623, 0.85133779251024},
                                  {0.023034156355267166, 0.6853101639063919}};
      std::vector<double> w
          = {0.06254121319590276,  0.02848605206887755,  0.04991833492806095,
             0.024266838081452035, 0.007931642509973639, 0.06254121319590276,
             0.02848605206887755,  0.04991833492806095,  0.024266838081452035,
             0.007931642509973639, 0.06254121319590276,  0.02848605206887755,
             0.04991833492806095,  0.024266838081452035, 0.007931642509973639,
             0.04322736365941421,  0.015083677576511441, 0.02178358503860756,
             0.04322736365941421,  0.015083677576511441, 0.02178358503860756,
             0.04322736365941421,  0.015083677576511441, 0.02178358503860756,
             0.04322736365941421,  0.015083677576511441, 0.02178358503860756,
             0.04322736365941421,  0.015083677576511441, 0.02178358503860756,
             0.04322736365941421,  0.015083677576511441, 0.02178358503860756};
      std::transform(w.cbegin(), w.cend(), w.begin(),
                     [](auto x) { return x / 2.0; });

      return {x, w};
    }
    else if (m == 13)
    {
      // Scheme from Xiao Gimbutas, 37 points, degree of precision 13
      xt::xtensor<double, 2> x = {{0.3333333333333333, 0.3333333333333333},
                                  {0.4961358947410461, 0.4961358947410461},
                                  {0.4696086896534919, 0.4696086896534919},
                                  {0.23111028494908226, 0.23111028494908226},
                                  {0.4144775702790546, 0.4144775702790546},
                                  {0.11355991257213327, 0.11355991257213327},
                                  {0.024895931491216494, 0.024895931491216494},
                                  {0.4961358947410461, 0.007728210517907841},
                                  {0.4696086896534919, 0.06078262069301621},
                                  {0.23111028494908226, 0.5377794301018355},
                                  {0.4144775702790546, 0.17104485944189085},
                                  {0.11355991257213327, 0.7728801748557335},
                                  {0.024895931491216494, 0.950208137017567},
                                  {0.007728210517907841, 0.4961358947410461},
                                  {0.06078262069301621, 0.4696086896534919},
                                  {0.5377794301018355, 0.23111028494908226},
                                  {0.17104485944189085, 0.4144775702790546},
                                  {0.7728801748557335, 0.11355991257213327},
                                  {0.950208137017567, 0.024895931491216494},
                                  {0.01898800438375904, 0.2920786885766364},
                                  {0.09773603106601653, 0.26674525331035115},
                                  {0.021966344206529244, 0.1267997757838373},
                                  {0.6889333070396046, 0.01898800438375904},
                                  {0.6355187156236324, 0.09773603106601653},
                                  {0.8512338800096335, 0.021966344206529244},
                                  {0.2920786885766364, 0.6889333070396046},
                                  {0.26674525331035115, 0.6355187156236324},
                                  {0.1267997757838373, 0.8512338800096335},
                                  {0.2920786885766364, 0.01898800438375904},
                                  {0.26674525331035115, 0.09773603106601653},
                                  {0.1267997757838373, 0.021966344206529244},
                                  {0.6889333070396046, 0.2920786885766364},
                                  {0.6355187156236324, 0.26674525331035115},
                                  {0.8512338800096335, 0.1267997757838373},
                                  {0.01898800438375904, 0.6889333070396046},
                                  {0.09773603106601653, 0.6355187156236324},
                                  {0.021966344206529244, 0.8512338800096335}};
      std::vector<double> w
          = {0.05162264666429082,  0.009941476361072588, 0.03278124160372298,
             0.04606240959277825,  0.0469470955421552,   0.030903097975759793,
             0.008029399795258423, 0.009941476361072588, 0.03278124160372298,
             0.04606240959277825,  0.0469470955421552,   0.030903097975759793,
             0.008029399795258423, 0.009941476361072588, 0.03278124160372298,
             0.04606240959277825,  0.0469470955421552,   0.030903097975759793,
             0.008029399795258423, 0.01812549864620088,  0.037211960457261536,
             0.015393072683782177, 0.01812549864620088,  0.037211960457261536,
             0.015393072683782177, 0.01812549864620088,  0.037211960457261536,
             0.015393072683782177, 0.01812549864620088,  0.037211960457261536,
             0.015393072683782177, 0.01812549864620088,  0.037211960457261536,
             0.015393072683782177, 0.01812549864620088,  0.037211960457261536,
             0.015393072683782177};
      std::transform(w.cbegin(), w.cend(), w.begin(),
                     [](auto x) { return x / 2.0; });
      return {x, w};
    }
    else if (m == 14)
    {
      // Scheme from Xiao Gimbutas, 42 points, degree of precision 14
      xt::xtensor<double, 2> x = {

          {0.41764471934045394, 0.41764471934045394},
          {0.0617998830908727, 0.0617998830908727},
          {0.2734775283088387, 0.2734775283088387},
          {0.1772055324125435, 0.1772055324125435},
          {0.0193909612487011, 0.0193909612487011},
          {0.4889639103621786, 0.4889639103621786},
          {0.41764471934045394, 0.16471056131909212},
          {0.0617998830908727, 0.8764002338182546},
          {0.2734775283088387, 0.4530449433823226},
          {0.1772055324125435, 0.645588935174913},
          {0.0193909612487011, 0.9612180775025978},
          {0.4889639103621786, 0.022072179275642756},
          {0.16471056131909212, 0.41764471934045394},
          {0.8764002338182546, 0.0617998830908727},
          {0.4530449433823226, 0.2734775283088387},
          {0.645588935174913, 0.1772055324125435},
          {0.9612180775025978, 0.0193909612487011},
          {0.022072179275642756, 0.4889639103621786},
          {0.014646950055654471, 0.29837288213625773},
          {0.09291624935697185, 0.336861459796345},
          {0.05712475740364799, 0.17226668782135557},
          {0.001268330932872076, 0.11897449769695682},
          {0.6869801678080878, 0.014646950055654471},
          {0.5702222908466832, 0.09291624935697185},
          {0.7706085547749965, 0.05712475740364799},
          {0.8797571713701712, 0.001268330932872076},
          {0.29837288213625773, 0.6869801678080878},
          {0.336861459796345, 0.5702222908466832},
          {0.17226668782135557, 0.7706085547749965},
          {0.11897449769695682, 0.8797571713701712},
          {0.29837288213625773, 0.014646950055654471},
          {0.336861459796345, 0.09291624935697185},
          {0.17226668782135557, 0.05712475740364799},
          {0.11897449769695682, 0.001268330932872076},
          {0.6869801678080878, 0.29837288213625773},
          {0.5702222908466832, 0.336861459796345},
          {0.7706085547749965, 0.17226668782135557},
          {0.8797571713701712, 0.11897449769695682},
          {0.014646950055654471, 0.6869801678080878},
          {0.09291624935697185, 0.5702222908466832},
          {0.05712475740364799, 0.7706085547749965},
          {0.001268330932872076, 0.8797571713701712}};
      std::vector<double> w
          = {0.032788353544125355, 0.014433699669776668, 0.051774104507291585,
             0.04216258873699302,  0.004923403602400082, 0.021883581369428893,
             0.032788353544125355, 0.014433699669776668, 0.051774104507291585,
             0.04216258873699302,  0.004923403602400082, 0.021883581369428893,
             0.032788353544125355, 0.014433699669776668, 0.051774104507291585,
             0.04216258873699302,  0.004923403602400082, 0.021883581369428893,
             0.014436308113533842, 0.038571510787060684, 0.024665753212563677,
             0.005010228838500672, 0.014436308113533842, 0.038571510787060684,
             0.024665753212563677, 0.005010228838500672, 0.014436308113533842,
             0.038571510787060684, 0.024665753212563677, 0.005010228838500672,
             0.014436308113533842, 0.038571510787060684, 0.024665753212563677,
             0.005010228838500672, 0.014436308113533842, 0.038571510787060684,
             0.024665753212563677, 0.005010228838500672, 0.014436308113533842,
             0.038571510787060684, 0.024665753212563677, 0.005010228838500672};
      std::transform(w.cbegin(), w.cend(), w.begin(),
                     [](auto x) { return x / 2.0; });
      return {x, w};
    }
    else if (m == 15)
    {
      // Scheme from Xiao Gimbutas, 49 points, degree of precision 15
      xt::xtensor<double, 2> x = {

          {0.3333333333333333, 0.3333333333333333},
          {0.1299782299330779, 0.1299782299330779},
          {0.4600769492970597, 0.4600769492970597},
          {0.4916858166302972, 0.4916858166302972},
          {0.22153234079514206, 0.22153234079514206},
          {0.39693373740906057, 0.39693373740906057},
          {0.0563419176961002, 0.0563419176961002},
          {0.1299782299330779, 0.7400435401338442},
          {0.4600769492970597, 0.07984610140588055},
          {0.4916858166302972, 0.016628366739405598},
          {0.22153234079514206, 0.5569353184097159},
          {0.39693373740906057, 0.20613252518187886},
          {0.0563419176961002, 0.8873161646077996},
          {0.7400435401338442, 0.1299782299330779},
          {0.07984610140588055, 0.4600769492970597},
          {0.016628366739405598, 0.4916858166302972},
          {0.5569353184097159, 0.22153234079514206},
          {0.20613252518187886, 0.39693373740906057},
          {0.8873161646077996, 0.0563419176961002},
          {0.08459422148219181, 0.18232178340719132},
          {0.016027089786345473, 0.15020038406523872},
          {0.09765044243024235, 0.32311131516371266},
          {0.018454251904633165, 0.3079476814836729},
          {0.0011135352740137417, 0.03803522930110929},
          {0.733083995110617, 0.08459422148219181},
          {0.8337725261484158, 0.016027089786345473},
          {0.5792382424060449, 0.09765044243024235},
          {0.673598066611694, 0.018454251904633165},
          {0.960851235424877, 0.0011135352740137417},
          {0.18232178340719132, 0.733083995110617},
          {0.15020038406523872, 0.8337725261484158},
          {0.32311131516371266, 0.5792382424060449},
          {0.3079476814836729, 0.673598066611694},
          {0.03803522930110929, 0.960851235424877},
          {0.18232178340719132, 0.08459422148219181},
          {0.15020038406523872, 0.016027089786345473},
          {0.32311131516371266, 0.09765044243024235},
          {0.3079476814836729, 0.018454251904633165},
          {0.03803522930110929, 0.0011135352740137417},
          {0.733083995110617, 0.18232178340719132},
          {0.8337725261484158, 0.15020038406523872},
          {0.5792382424060449, 0.32311131516371266},
          {0.673598066611694, 0.3079476814836729},
          {0.960851235424877, 0.03803522930110929},
          {0.08459422148219181, 0.733083995110617},
          {0.016027089786345473, 0.8337725261484158},
          {0.09765044243024235, 0.5792382424060449},
          {0.018454251904633165, 0.673598066611694},
          {0.0011135352740137417, 0.960851235424877}};
      std::vector<double> w = {
          0.02973041974807132,   0.0073975040670461,    0.021594087936438452,
          0.0158322763500218,    0.046287286105198076,  0.046336041391207235,
          0.015084474247597068,  0.0073975040670461,    0.021594087936438452,
          0.0158322763500218,    0.046287286105198076,  0.046336041391207235,
          0.015084474247597068,  0.0073975040670461,    0.021594087936438452,
          0.0158322763500218,    0.046287286105198076,  0.046336041391207235,
          0.015084474247597068,  0.024230008783125607,  0.01122850429887806,
          0.03107522047051095,   0.016436762092827895,  0.0024752660145579163,
          0.024230008783125607,  0.01122850429887806,   0.03107522047051095,
          0.016436762092827895,  0.0024752660145579163, 0.024230008783125607,
          0.01122850429887806,   0.03107522047051095,   0.016436762092827895,
          0.0024752660145579163, 0.024230008783125607,  0.01122850429887806,
          0.03107522047051095,   0.016436762092827895,  0.0024752660145579163,
          0.024230008783125607,  0.01122850429887806,   0.03107522047051095,
          0.016436762092827895,  0.0024752660145579163, 0.024230008783125607,
          0.01122850429887806,   0.03107522047051095,   0.016436762092827895,
          0.0024752660145579163};
      std::transform(w.cbegin(), w.cend(), w.begin(),
                     [](auto x) { return x / 2.0; });
      return {x, w};
    }
    else if (m == 16)
    {
      // Scheme from Xiao Gimbutas, 55 points, degree of precision 16
      xt::xtensor<double, 2> x = {

          {0.3333333333333333, 0.3333333333333333},
          {0.06667447224023837, 0.06667447224023837},
          {0.24132168070137838, 0.24132168070137838},
          {0.41279809595522365, 0.41279809595522365},
          {0.15006373658703515, 0.15006373658703515},
          {0.46954803099668496, 0.46954803099668496},
          {0.017041629405718517, 0.017041629405718517},
          {0.06667447224023837, 0.8666510555195233},
          {0.24132168070137838, 0.5173566385972432},
          {0.41279809595522365, 0.1744038080895527},
          {0.15006373658703515, 0.6998725268259297},
          {0.46954803099668496, 0.060903938006630076},
          {0.017041629405718517, 0.965916741188563},
          {0.8666510555195233, 0.06667447224023837},
          {0.5173566385972432, 0.24132168070137838},
          {0.1744038080895527, 0.41279809595522365},
          {0.6998725268259297, 0.15006373658703515},
          {0.060903938006630076, 0.46954803099668496},
          {0.965916741188563, 0.017041629405718517},
          {0.009664954403660254, 0.41376948582708517},
          {0.030305943355186365, 0.30417944822947973},
          {0.010812972776103751, 0.08960908902270585},
          {0.10665316053614844, 0.29661537240038294},
          {0.051354315344013114, 0.16976335515028973},
          {0.0036969427073556124, 0.21404877992584728},
          {0.5765655597692546, 0.009664954403660254},
          {0.6655146084153339, 0.030305943355186365},
          {0.8995779382011905, 0.010812972776103751},
          {0.5967314670634686, 0.10665316053614844},
          {0.7788823295056971, 0.051354315344013114},
          {0.7822542773667971, 0.0036969427073556124},
          {0.41376948582708517, 0.5765655597692546},
          {0.30417944822947973, 0.6655146084153339},
          {0.08960908902270585, 0.8995779382011905},
          {0.29661537240038294, 0.5967314670634686},
          {0.16976335515028973, 0.7788823295056971},
          {0.21404877992584728, 0.7822542773667971},
          {0.41376948582708517, 0.009664954403660254},
          {0.30417944822947973, 0.030305943355186365},
          {0.08960908902270585, 0.010812972776103751},
          {0.29661537240038294, 0.10665316053614844},
          {0.16976335515028973, 0.051354315344013114},
          {0.21404877992584728, 0.0036969427073556124},
          {0.5765655597692546, 0.41376948582708517},
          {0.6655146084153339, 0.30417944822947973},
          {0.8995779382011905, 0.08960908902270585},
          {0.5967314670634686, 0.29661537240038294},
          {0.7788823295056971, 0.16976335515028973},
          {0.7822542773667971, 0.21404877992584728},
          {0.009664954403660254, 0.5765655597692546},
          {0.030305943355186365, 0.6655146084153339},
          {0.010812972776103751, 0.8995779382011905},
          {0.10665316053614844, 0.5967314670634686},
          {0.051354315344013114, 0.7788823295056971},
          {0.0036969427073556124, 0.7822542773667971}};
      std::vector<double> w
          = {0.046227910314191344,  0.012425425595561009, 0.04118404106979255,
             0.040985219786815366,  0.02878349670274891,  0.02709366946771045,
             0.003789135238264222,  0.012425425595561009, 0.04118404106979255,
             0.040985219786815366,  0.02878349670274891,  0.02709366946771045,
             0.003789135238264222,  0.012425425595561009, 0.04118404106979255,
             0.040985219786815366,  0.02878349670274891,  0.02709366946771045,
             0.003789135238264222,  0.008182210553222139, 0.013983607124653567,
             0.005751869970497159,  0.031646061681983244, 0.017653081047103284,
             0.0046146906397291345, 0.008182210553222139, 0.013983607124653567,
             0.005751869970497159,  0.031646061681983244, 0.017653081047103284,
             0.0046146906397291345, 0.008182210553222139, 0.013983607124653567,
             0.005751869970497159,  0.031646061681983244, 0.017653081047103284,
             0.0046146906397291345, 0.008182210553222139, 0.013983607124653567,
             0.005751869970497159,  0.031646061681983244, 0.017653081047103284,
             0.0046146906397291345, 0.008182210553222139, 0.013983607124653567,
             0.005751869970497159,  0.031646061681983244, 0.017653081047103284,
             0.0046146906397291345, 0.008182210553222139, 0.013983607124653567,
             0.005751869970497159,  0.031646061681983244, 0.017653081047103284,
             0.0046146906397291345};
      std::transform(w.cbegin(), w.cend(), w.begin(),
                     [](auto x) { return x / 2.0; });
      return {x, w};
    }
    else if (m == 17)
    {
      // Scheme from Xiao Gimbutas, 60 points, degree of precision 17
      xt::xtensor<double, 2> x = {{0.4171034443615992, 0.4171034443615992},
                                  {0.18035811626637066, 0.18035811626637066},
                                  {0.2857065024365867, 0.2857065024365867},
                                  {0.06665406347959701, 0.06665406347959701},
                                  {0.014755491660754072, 0.014755491660754072},
                                  {0.46559787161889027, 0.46559787161889027},
                                  {0.4171034443615992, 0.16579311127680163},
                                  {0.18035811626637066, 0.6392837674672587},
                                  {0.2857065024365867, 0.42858699512682663},
                                  {0.06665406347959701, 0.866691873040806},
                                  {0.014755491660754072, 0.9704890166784919},
                                  {0.46559787161889027, 0.06880425676221946},
                                  {0.16579311127680163, 0.4171034443615992},
                                  {0.6392837674672587, 0.18035811626637066},
                                  {0.42858699512682663, 0.2857065024365867},
                                  {0.866691873040806, 0.06665406347959701},
                                  {0.9704890166784919, 0.014755491660754072},
                                  {0.06880425676221946, 0.46559787161889027},
                                  {0.011575175903180683, 0.07250547079900238},
                                  {0.013229672760086951, 0.41547545929522905},
                                  {0.013135870834002753, 0.27179187005535477},
                                  {0.15750547792686992, 0.29921894247697034},
                                  {0.06734937786736123, 0.3062815917461865},
                                  {0.07804234056828245, 0.16872251349525944},
                                  {0.016017642362119337, 0.15919228747279268},
                                  {0.9159193532978169, 0.011575175903180683},
                                  {0.5712948679446841, 0.013229672760086951},
                                  {0.7150722591106424, 0.013135870834002753},
                                  {0.5432755795961598, 0.15750547792686992},
                                  {0.6263690303864522, 0.06734937786736123},
                                  {0.7532351459364581, 0.07804234056828245},
                                  {0.824790070165088, 0.016017642362119337},
                                  {0.07250547079900238, 0.9159193532978169},
                                  {0.41547545929522905, 0.5712948679446841},
                                  {0.27179187005535477, 0.7150722591106424},
                                  {0.29921894247697034, 0.5432755795961598},
                                  {0.3062815917461865, 0.6263690303864522},
                                  {0.16872251349525944, 0.7532351459364581},
                                  {0.15919228747279268, 0.824790070165088},
                                  {0.07250547079900238, 0.011575175903180683},
                                  {0.41547545929522905, 0.013229672760086951},
                                  {0.27179187005535477, 0.013135870834002753},
                                  {0.29921894247697034, 0.15750547792686992},
                                  {0.3062815917461865, 0.06734937786736123},
                                  {0.16872251349525944, 0.07804234056828245},
                                  {0.15919228747279268, 0.016017642362119337},
                                  {0.9159193532978169, 0.07250547079900238},
                                  {0.5712948679446841, 0.41547545929522905},
                                  {0.7150722591106424, 0.27179187005535477},
                                  {0.5432755795961598, 0.29921894247697034},
                                  {0.6263690303864522, 0.3062815917461865},
                                  {0.7532351459364581, 0.16872251349525944},
                                  {0.824790070165088, 0.15919228747279268},
                                  {0.011575175903180683, 0.9159193532978169},
                                  {0.013229672760086951, 0.5712948679446841},
                                  {0.013135870834002753, 0.7150722591106424},
                                  {0.15750547792686992, 0.5432755795961598},
                                  {0.06734937786736123, 0.6263690303864522},
                                  {0.07804234056828245, 0.7532351459364581},
                                  {0.016017642362119337, 0.824790070165088}};
      std::vector<double> w = {

          0.027310926528102106, 0.026312630588017985, 0.03771623715279528,
          0.012459000802305444, 0.002773887577637642, 0.02501945095049736,
          0.027310926528102106, 0.026312630588017985, 0.03771623715279528,
          0.012459000802305444, 0.002773887577637642, 0.02501945095049736,
          0.027310926528102106, 0.026312630588017985, 0.03771623715279528,
          0.012459000802305444, 0.002773887577637642, 0.02501945095049736,
          0.004584348401735868, 0.010398439955839537, 0.008692214501001192,
          0.02617162593533699,  0.022487772546691067, 0.02055789832045452,
          0.007978300205929593, 0.004584348401735868, 0.010398439955839537,
          0.008692214501001192, 0.02617162593533699,  0.022487772546691067,
          0.02055789832045452,  0.007978300205929593, 0.004584348401735868,
          0.010398439955839537, 0.008692214501001192, 0.02617162593533699,
          0.022487772546691067, 0.02055789832045452,  0.007978300205929593,
          0.004584348401735868, 0.010398439955839537, 0.008692214501001192,
          0.02617162593533699,  0.022487772546691067, 0.02055789832045452,
          0.007978300205929593, 0.004584348401735868, 0.010398439955839537,
          0.008692214501001192, 0.02617162593533699,  0.022487772546691067,
          0.02055789832045452,  0.007978300205929593, 0.004584348401735868,
          0.010398439955839537, 0.008692214501001192, 0.02617162593533699,
          0.022487772546691067, 0.02055789832045452,  0.007978300205929593};
      std::transform(w.cbegin(), w.cend(), w.begin(),
                     [](auto x) { return x / 2.0; });

      return {x, w};
    }
    else if (m == 18)
    {
      // Scheme from Xiao Gimbutas, 67 points, degree of precision 18
      xt::xtensor<double, 2> x = {

          {0.3333333333333333, 0.3333333333333333},
          {0.4749182113240457, 0.4749182113240457},
          {0.15163850697260495, 0.15163850697260495},
          {0.4110671018759195, 0.4110671018759195},
          {0.2656146099053742, 0.2656146099053742},
          {0.0037589443410684376, 0.0037589443410684376},
          {0.072438705567333, 0.072438705567333},
          {0.4749182113240457, 0.05016357735190857},
          {0.15163850697260495, 0.6967229860547901},
          {0.4110671018759195, 0.177865796248161},
          {0.2656146099053742, 0.46877078018925156},
          {0.0037589443410684376, 0.9924821113178631},
          {0.072438705567333, 0.855122588865334},
          {0.05016357735190857, 0.4749182113240457},
          {0.6967229860547901, 0.15163850697260495},
          {0.177865796248161, 0.4110671018759195},
          {0.46877078018925156, 0.2656146099053742},
          {0.9924821113178631, 0.0037589443410684376},
          {0.855122588865334, 0.072438705567333},
          {0.09042704035434063, 0.3850440344131637},
          {0.012498932483495477, 0.04727614183265175},
          {0.05401173533902428, 0.30206195771287075},
          {0.010505018819241962, 0.2565061597742415},
          {0.06612245802840343, 0.17847912556588763},
          {0.14906691012577386, 0.2685733063960138},
          {0.011691824674667157, 0.41106566867461836},
          {0.014331524778941987, 0.1327788302713893},
          {0.5245289252324957, 0.09042704035434063},
          {0.9402249256838529, 0.012498932483495477},
          {0.6439263069481049, 0.05401173533902428},
          {0.7329888214065166, 0.010505018819241962},
          {0.7553984164057089, 0.06612245802840343},
          {0.5823597834782124, 0.14906691012577386},
          {0.5772425066507145, 0.011691824674667157},
          {0.8528896449496688, 0.014331524778941987},
          {0.3850440344131637, 0.5245289252324957},
          {0.04727614183265175, 0.9402249256838529},
          {0.30206195771287075, 0.6439263069481049},
          {0.2565061597742415, 0.7329888214065166},
          {0.17847912556588763, 0.7553984164057089},
          {0.2685733063960138, 0.5823597834782124},
          {0.41106566867461836, 0.5772425066507145},
          {0.1327788302713893, 0.8528896449496688},
          {0.3850440344131637, 0.09042704035434063},
          {0.04727614183265175, 0.012498932483495477},
          {0.30206195771287075, 0.05401173533902428},
          {0.2565061597742415, 0.010505018819241962},
          {0.17847912556588763, 0.06612245802840343},
          {0.2685733063960138, 0.14906691012577386},
          {0.41106566867461836, 0.011691824674667157},
          {0.1327788302713893, 0.014331524778941987},
          {0.5245289252324957, 0.3850440344131637},
          {0.9402249256838529, 0.04727614183265175},
          {0.6439263069481049, 0.30206195771287075},
          {0.7329888214065166, 0.2565061597742415},
          {0.7553984164057089, 0.17847912556588763},
          {0.5823597834782124, 0.2685733063960138},
          {0.5772425066507145, 0.41106566867461836},
          {0.8528896449496688, 0.1327788302713893},
          {0.09042704035434063, 0.5245289252324957},
          {0.012498932483495477, 0.9402249256838529},
          {0.05401173533902428, 0.6439263069481049},
          {0.010505018819241962, 0.7329888214065166},
          {0.06612245802840343, 0.7553984164057089},
          {0.14906691012577386, 0.5823597834782124},
          {0.011691824674667157, 0.5772425066507145},
          {0.014331524778941987, 0.8528896449496688}};
      std::vector<double> w = {

          0.03074852123911586,  0.013107027491738756, 0.0203183388454584,
          0.0334719940598479,   0.031116396602006133, 0.0005320056169477806,
          0.013790286604766942, 0.013107027491738756, 0.0203183388454584,
          0.0334719940598479,   0.031116396602006133, 0.0005320056169477806,
          0.013790286604766942, 0.013107027491738756, 0.0203183388454584,
          0.0334719940598479,   0.031116396602006133, 0.0005320056169477806,
          0.013790286604766942, 0.015328258194553142, 0.004217516774744443,
          0.016365908413986566, 0.007729835280006227, 0.01691165391748008,
          0.02759288648857948,  0.009586124474361505, 0.007641704972719637,
          0.015328258194553142, 0.004217516774744443, 0.016365908413986566,
          0.007729835280006227, 0.01691165391748008,  0.02759288648857948,
          0.009586124474361505, 0.007641704972719637, 0.015328258194553142,
          0.004217516774744443, 0.016365908413986566, 0.007729835280006227,
          0.01691165391748008,  0.02759288648857948,  0.009586124474361505,
          0.007641704972719637, 0.015328258194553142, 0.004217516774744443,
          0.016365908413986566, 0.007729835280006227, 0.01691165391748008,
          0.02759288648857948,  0.009586124474361505, 0.007641704972719637,
          0.015328258194553142, 0.004217516774744443, 0.016365908413986566,
          0.007729835280006227, 0.01691165391748008,  0.02759288648857948,
          0.009586124474361505, 0.007641704972719637, 0.015328258194553142,
          0.004217516774744443, 0.016365908413986566, 0.007729835280006227,
          0.01691165391748008,  0.02759288648857948,  0.009586124474361505,
          0.007641704972719637};
      std::transform(w.cbegin(), w.cend(), w.begin(),
                     [](auto x) { return x / 2.0; });

      return {x, w};
    }
    else if (m == 19)
    {
      // Scheme from Xiao Gimbutas, 73 points, degree of precision
      xt::xtensor<double, 2> x = {

          {0.3333333333333333, 0.3333333333333333},
          {0.05252627985410363, 0.05252627985410363},
          {0.11144805571699878, 0.11144805571699878},
          {0.011639027327922657, 0.011639027327922657},
          {0.25516213315312486, 0.25516213315312486},
          {0.4039697179663861, 0.4039697179663861},
          {0.17817100607962755, 0.17817100607962755},
          {0.4591943889568276, 0.4591943889568276},
          {0.4925124498658742, 0.4925124498658742},
          {0.05252627985410363, 0.8949474402917927},
          {0.11144805571699878, 0.7771038885660024},
          {0.011639027327922657, 0.9767219453441547},
          {0.25516213315312486, 0.4896757336937503},
          {0.4039697179663861, 0.19206056406722782},
          {0.17817100607962755, 0.6436579878407449},
          {0.4591943889568276, 0.08161122208634475},
          {0.4925124498658742, 0.014975100268251551},
          {0.8949474402917927, 0.05252627985410363},
          {0.7771038885660024, 0.11144805571699878},
          {0.9767219453441547, 0.011639027327922657},
          {0.4896757336937503, 0.25516213315312486},
          {0.19206056406722782, 0.4039697179663861},
          {0.6436579878407449, 0.17817100607962755},
          {0.08161122208634475, 0.4591943889568276},
          {0.014975100268251551, 0.4925124498658742},
          {0.005005142352350433, 0.1424222825711269},
          {0.009777061438676854, 0.06008389996270236},
          {0.039142449434608845, 0.13070066996053453},
          {0.129312809767979, 0.31131838322398686},
          {0.07456118930435514, 0.22143394188911344},
          {0.04088831446497813, 0.3540259269997119},
          {0.014923638907438481, 0.24189410400689262},
          {0.0020691038491023883, 0.36462041433871},
          {0.8525725750765227, 0.005005142352350433},
          {0.9301390385986208, 0.009777061438676854},
          {0.8301568806048566, 0.039142449434608845},
          {0.5593688070080342, 0.129312809767979},
          {0.7040048688065313, 0.07456118930435514},
          {0.60508575853531, 0.04088831446497813},
          {0.7431822570856689, 0.014923638907438481},
          {0.6333104818121875, 0.0020691038491023883},
          {0.1424222825711269, 0.8525725750765227},
          {0.06008389996270236, 0.9301390385986208},
          {0.13070066996053453, 0.8301568806048566},
          {0.31131838322398686, 0.5593688070080342},
          {0.22143394188911344, 0.7040048688065313},
          {0.3540259269997119, 0.60508575853531},
          {0.24189410400689262, 0.7431822570856689},
          {0.36462041433871, 0.6333104818121875},
          {0.1424222825711269, 0.005005142352350433},
          {0.06008389996270236, 0.009777061438676854},
          {0.13070066996053453, 0.039142449434608845},
          {0.31131838322398686, 0.129312809767979},
          {0.22143394188911344, 0.07456118930435514},
          {0.3540259269997119, 0.04088831446497813},
          {0.24189410400689262, 0.014923638907438481},
          {0.36462041433871, 0.0020691038491023883},
          {0.8525725750765227, 0.1424222825711269},
          {0.9301390385986208, 0.06008389996270236},
          {0.8301568806048566, 0.13070066996053453},
          {0.5593688070080342, 0.31131838322398686},
          {0.7040048688065313, 0.22143394188911344},
          {0.60508575853531, 0.3540259269997119},
          {0.7431822570856689, 0.24189410400689262},
          {0.6333104818121875, 0.36462041433871},
          {0.005005142352350433, 0.8525725750765227},
          {0.009777061438676854, 0.9301390385986208},
          {0.039142449434608845, 0.8301568806048566},
          {0.129312809767979, 0.5593688070080342},
          {0.07456118930435514, 0.7040048688065313},
          {0.04088831446497813, 0.60508575853531},
          {0.014923638907438481, 0.7431822570856689},
          {0.0020691038491023883, 0.6333104818121875}

      };
      std::vector<double> w = {
          0.034469160850905275,  0.007109393622794947,  0.015234956517004836,
          0.0017651924183085402, 0.03175285458752998,   0.03153735864523962,
          0.02465198105358483,   0.022983570977123252,  0.010321882182418864,
          0.007109393622794947,  0.015234956517004836,  0.0017651924183085402,
          0.03175285458752998,   0.03153735864523962,   0.02465198105358483,
          0.022983570977123252,  0.010321882182418864,  0.007109393622794947,
          0.015234956517004836,  0.0017651924183085402, 0.03175285458752998,
          0.03153735864523962,   0.02465198105358483,   0.022983570977123252,
          0.010321882182418864,  0.0029256924878800715, 0.0033273888405939045,
          0.009695519081624202,  0.026346264707445364,  0.018108074590430505,
          0.016102209460939428,  0.00845592483909348,   0.0032821375148397378,
          0.0029256924878800715, 0.0033273888405939045, 0.009695519081624202,
          0.026346264707445364,  0.018108074590430505,  0.016102209460939428,
          0.00845592483909348,   0.0032821375148397378, 0.0029256924878800715,
          0.0033273888405939045, 0.009695519081624202,  0.026346264707445364,
          0.018108074590430505,  0.016102209460939428,  0.00845592483909348,
          0.0032821375148397378, 0.0029256924878800715, 0.0033273888405939045,
          0.009695519081624202,  0.026346264707445364,  0.018108074590430505,
          0.016102209460939428,  0.00845592483909348,   0.0032821375148397378,
          0.0029256924878800715, 0.0033273888405939045, 0.009695519081624202,
          0.026346264707445364,  0.018108074590430505,  0.016102209460939428,
          0.00845592483909348,   0.0032821375148397378, 0.0029256924878800715,
          0.0033273888405939045, 0.009695519081624202,  0.026346264707445364,
          0.018108074590430505,  0.016102209460939428,  0.00845592483909348,
          0.0032821375148397378};
      std::transform(w.cbegin(), w.cend(), w.begin(),
                     [](auto x) { return x / 2.0; });

      return {x, w};
    }
    else if (m == 20)
    {
      // Scheme from Xiao Gimbutas, 79 points, degree of precision 20
      xt::xtensor<double, 2> x = {

          {0.3333333333333333, 0.3333333333333333},
          {0.18629499774454095, 0.18629499774454095},
          {0.037310880598884766, 0.037310880598884766},
          {0.476245611540499, 0.476245611540499},
          {0.4455510569559248, 0.4455510569559248},
          {0.25457926767333916, 0.25457926767333916},
          {0.39342534781709987, 0.39342534781709987},
          {0.01097614102839789, 0.01097614102839789},
          {0.10938359671171471, 0.10938359671171471},
          {0.18629499774454095, 0.6274100045109181},
          {0.037310880598884766, 0.9253782388022305},
          {0.476245611540499, 0.047508776919002016},
          {0.4455510569559248, 0.10889788608815043},
          {0.25457926767333916, 0.4908414646533217},
          {0.39342534781709987, 0.21314930436580026},
          {0.01097614102839789, 0.9780477179432042},
          {0.10938359671171471, 0.7812328065765706},
          {0.6274100045109181, 0.18629499774454095},
          {0.9253782388022305, 0.037310880598884766},
          {0.047508776919002016, 0.476245611540499},
          {0.10889788608815043, 0.4455510569559248},
          {0.4908414646533217, 0.25457926767333916},
          {0.21314930436580026, 0.39342534781709987},
          {0.9780477179432042, 0.01097614102839789},
          {0.7812328065765706, 0.10938359671171471},
          {0.004854937607623827, 0.06409058560843404},
          {0.10622720472027006, 0.2156070573900944},
          {0.007570780504696579, 0.15913370765706722},
          {0.13980807199179993, 0.317860123835772},
          {0.04656036490766434, 0.19851813222878817},
          {0.038363684775374655, 0.09995229628813862},
          {0.009831548292802588, 0.42002375881622406},
          {0.05498747914298685, 0.33313481730958744},
          {0.01073721285601111, 0.2805814114236652},
          {0.9310544767839422, 0.004854937607623827},
          {0.6781657378896355, 0.10622720472027006},
          {0.8332955118382361, 0.007570780504696579},
          {0.5423318041724281, 0.13980807199179993},
          {0.7549215028635474, 0.04656036490766434},
          {0.8616840189364867, 0.038363684775374655},
          {0.5701446928909732, 0.009831548292802588},
          {0.6118777035474257, 0.05498747914298685},
          {0.7086813757203236, 0.01073721285601111},
          {0.06409058560843404, 0.9310544767839422},
          {0.2156070573900944, 0.6781657378896355},
          {0.15913370765706722, 0.8332955118382361},
          {0.317860123835772, 0.5423318041724281},
          {0.19851813222878817, 0.7549215028635474},
          {0.09995229628813862, 0.8616840189364867},
          {0.42002375881622406, 0.5701446928909732},
          {0.33313481730958744, 0.6118777035474257},
          {0.2805814114236652, 0.7086813757203236},
          {0.06409058560843404, 0.004854937607623827},
          {0.2156070573900944, 0.10622720472027006},
          {0.15913370765706722, 0.007570780504696579},
          {0.317860123835772, 0.13980807199179993},
          {0.19851813222878817, 0.04656036490766434},
          {0.09995229628813862, 0.038363684775374655},
          {0.42002375881622406, 0.009831548292802588},
          {0.33313481730958744, 0.05498747914298685},
          {0.2805814114236652, 0.01073721285601111},
          {0.9310544767839422, 0.06409058560843404},
          {0.6781657378896355, 0.2156070573900944},
          {0.8332955118382361, 0.15913370765706722},
          {0.5423318041724281, 0.317860123835772},
          {0.7549215028635474, 0.19851813222878817},
          {0.8616840189364867, 0.09995229628813862},
          {0.5701446928909732, 0.42002375881622406},
          {0.6118777035474257, 0.33313481730958744},
          {0.7086813757203236, 0.2805814114236652},
          {0.004854937607623827, 0.9310544767839422},
          {0.10622720472027006, 0.6781657378896355},
          {0.007570780504696579, 0.8332955118382361},
          {0.13980807199179993, 0.5423318041724281},
          {0.04656036490766434, 0.7549215028635474},
          {0.038363684775374655, 0.8616840189364867},
          {0.009831548292802588, 0.5701446928909732},
          {0.05498747914298685, 0.6118777035474257},
          {0.01073721285601111, 0.7086813757203236}};
      std::vector<double> w = {

          0.027820221402906232,  0.01834692594850583,   0.0043225508213311555,
          0.014203650606816881,  0.018904799866464896,  0.028166402615040498,
          0.027576101258140917,  0.00159768158213324,   0.01566046155214907,
          0.01834692594850583,   0.0043225508213311555, 0.014203650606816881,
          0.018904799866464896,  0.028166402615040498,  0.027576101258140917,
          0.00159768158213324,   0.01566046155214907,   0.01834692594850583,
          0.0043225508213311555, 0.014203650606816881,  0.018904799866464896,
          0.028166402615040498,  0.027576101258140917,  0.00159768158213324,
          0.01566046155214907,   0.002259739204251731,  0.015445215644198462,
          0.004405794837116996,  0.02338349146365547,   0.01197279715790938,
          0.008291423055227716,  0.007391363000510596,  0.01733445113443867,
          0.007156400476915371,  0.002259739204251731,  0.015445215644198462,
          0.004405794837116996,  0.02338349146365547,   0.01197279715790938,
          0.008291423055227716,  0.007391363000510596,  0.01733445113443867,
          0.007156400476915371,  0.002259739204251731,  0.015445215644198462,
          0.004405794837116996,  0.02338349146365547,   0.01197279715790938,
          0.008291423055227716,  0.007391363000510596,  0.01733445113443867,
          0.007156400476915371,  0.002259739204251731,  0.015445215644198462,
          0.004405794837116996,  0.02338349146365547,   0.01197279715790938,
          0.008291423055227716,  0.007391363000510596,  0.01733445113443867,
          0.007156400476915371,  0.002259739204251731,  0.015445215644198462,
          0.004405794837116996,  0.02338349146365547,   0.01197279715790938,
          0.008291423055227716,  0.007391363000510596,  0.01733445113443867,
          0.007156400476915371,  0.002259739204251731,  0.015445215644198462,
          0.004405794837116996,  0.02338349146365547,   0.01197279715790938,
          0.008291423055227716,  0.007391363000510596,  0.01733445113443867,
          0.007156400476915371};
      std::transform(w.cbegin(), w.cend(), w.begin(),
                     [](auto x) { return x / 2.0; });
      return {x, w};
    }
    else
    {
      throw std::runtime_error("Xiao-Gimbutas not implemented for this order.");
    }
  }
  else
  {
    throw std::runtime_error(
        "Xiao-Gimbutas is only implemented for triangles.");
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
    else if (m >= 10 and m <= 20)
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
    else
      return type::gauss_jacobi;
  }
  else
    return type::gauss_jacobi;
}
//-----------------------------------------------------------------------------
std::pair<xt::xarray<double>, std::vector<double>>
quadrature::make_quadrature(quadrature::type rule, cell::type celltype, int m)
{
  switch (rule)
  {
  case quadrature::type::Default:
    return make_quadrature(get_default_rule(celltype, m), celltype, m);
  case quadrature::type::gauss_jacobi:
    return make_gauss_jacobi_quadrature(celltype, m);
  case quadrature::type::gll:
    return make_gll_quadrature(celltype, m);
  case quadrature::type::xiao_gimbutas:
    return make_xiao_gimbutas_quadrature(celltype, m);
  case quadrature::type::zienkiewicz_taylor:
    return make_zienkiewicz_taylor_quadrature(celltype, m);
  case quadrature::type::keast:
    return make_keast_quadrature(celltype, m);
  case quadrature::type::strang_fix:
    return make_strang_fix_quadrature(celltype, m);
  default:
    throw std::runtime_error("Unknown quadrature rule");
  }
}
//-----------------------------------------------------------------------------
std::pair<xt::xarray<double>, std::vector<double>>
quadrature::make_quadrature(cell::type celltype, int m)
{
  return make_quadrature(quadrature::type::Default, celltype, m);
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 1> quadrature::get_gl_points(int m)
{
  std::vector<double> _pts = compute_gauss_jacobi_points(0, m);
  std::array<std::size_t, 1> shape = {_pts.size()};
  xt::xtensor<double, 1> pts(shape);
  for (std::size_t i = 0; i < _pts.size(); ++i)
    pts(i) = 0.5 + 0.5 * _pts[i];
  return pts;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 1> quadrature::get_gll_points(int m)
{
  [[maybe_unused]] auto [pts, wts] = make_gll_line(m);
  return pts;
}
//-----------------------------------------------------------------------------
