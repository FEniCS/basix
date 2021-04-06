// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "quadrature.h"
#include "span.hpp"
#include <cmath>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xview.hpp>

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

  xt::xtensor<double, 2> A = xt::diag(_alpha);
  auto tmp = xt::sqrt(xt::view(_beta, xt::range(1, _)));
  A += xt::diag(xt::sqrt(tmp), 1) + xt::diag(xt::sqrt(tmp), -1);

  auto [evals, evecs] = xt::linalg::eigh(A);

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
std::pair<xt::xarray<double>, std::vector<double>>
make_gauss_jacobi_quadrature(cell::type celltype, std::size_t m)
{
  switch (celltype)
  {
  case cell::type::interval:
    return quadrature::make_quadrature_line(m);
  case cell::type::quadrilateral:
  {
    auto [QptsL, QwtsL] = quadrature::make_quadrature_line(m);
    xt::xtensor<double, 2> Qpts({m * m, 2});
    std::vector<double> Qwts(m * m);
    int c = 0;
    for (std::size_t j = 0; j < m; ++j)
    {
      for (std::size_t i = 0; i < m; ++i)
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
    auto [QptsL, QwtsL] = quadrature::make_quadrature_line(m);
    xt::xtensor<double, 2> Qpts({m * m * m, 3});
    std::vector<double> Qwts(m * m * m);
    int c = 0;
    for (std::size_t k = 0; k < m; ++k)
    {
      for (std::size_t j = 0; j < m; ++j)
      {
        for (std::size_t i = 0; i < m; ++i)
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
    auto [QptsL, QwtsL] = quadrature::make_quadrature_line(m);
    auto [QptsT, QwtsT] = quadrature::make_quadrature_triangle_collapsed(m);
    xt::xtensor<double, 2> Qpts({m * QptsT.shape(0), 3});
    std::vector<double> Qwts(m * QptsT.shape(0));
    int c = 0;
    for (std::size_t k = 0; k < m; ++k)
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
    return quadrature::make_quadrature_triangle_collapsed(m);
  case cell::type::tetrahedron:
    return quadrature::make_quadrature_tetrahedron_collapsed(m);
  default:
    throw std::runtime_error("Unsupported celltype for make_quadrature");
  }
}
//-----------------------------------------------------------------------------
std::pair<xt::xarray<double>, std::vector<double>>
make_gll_quadrature(cell::type celltype, std::size_t m)
{
  switch (celltype)
  {
  case cell::type::interval:
    return quadrature::make_gll_line(m);
  case cell::type::quadrilateral:
  {
    auto [QptsL, QwtsL] = quadrature::make_gll_line(m);
    xt::xtensor<double, 2> Qpts({m * m, 2});
    std::vector<double> Qwts(m * m);
    int c = 0;
    for (std::size_t j = 0; j < m; ++j)
    {
      for (std::size_t i = 0; i < m; ++i)
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
    auto [QptsL, QwtsL] = quadrature::make_gll_line(m);
    xt::xtensor<double, 2> Qpts({m * m * m, 3});
    std::vector<double> Qwts(m * m * m);
    int c = 0;
    for (std::size_t k = 0; k < m; ++k)
    {
      for (std::size_t j = 0; j < m; ++j)
      {
        for (std::size_t i = 0; i < m; ++i)
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
make_default_tetrahedron_quadrature(int m)
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
    xt::xtensor_fixed<double, xt::xshape<4, 3>> x
        = {{a, b, b}, {b, a, b}, {b, b, a}, {b, b, b}};
    return {x, {1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0}};
  }
  else if (m == 3)
  {
    // Scheme from Zienkiewicz and Taylor, 5 points, degree of precision 3
    // Note : this scheme has a negative weight
    xt::xtensor_fixed<double, xt::xshape<5, 3>> x{
        {0.2500000000000000, 0.2500000000000000, 0.2500000000000000},
        {0.5000000000000000, 0.1666666666666666, 0.1666666666666666},
        {0.1666666666666666, 0.5000000000000000, 0.1666666666666666},
        {0.1666666666666666, 0.1666666666666666, 0.5000000000000000},
        {0.1666666666666666, 0.1666666666666666, 0.1666666666666666}};
    return {x, {-0.8 / 6.0, 0.45 / 6.0, 0.45 / 6.0, 0.45 / 6.0, 0.45 / 6.0}};
  }
  else if (m == 4)
  {
    // Keast rule, 14 points, degree of precision 4
    // Values taken from
    // http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
    // (KEAST5)
    xt::xtensor_fixed<double, xt::xshape<14, 3>> x
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
    xt::xtensor_fixed<double, xt::xshape<15, 3>> x
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
    xt::xtensor_fixed<double, xt::xshape<24, 3>> x
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
    xt::xtensor_fixed<double, xt::xshape<31, 3>> x
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
    xt::xtensor_fixed<double, xt::xshape<45, 3>> x
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
  {
    const int np = (m + 2) / 2;
    return quadrature::make_quadrature_tetrahedron_collapsed(np);
  }
}
//-----------------------------------------------------------------------------
std::pair<xt::xarray<double>, std::vector<double>>
make_default_triangle_quadrature(int m)
{
  if (m == 0 or m == 1)
  {
    // Scheme from Zienkiewicz and Taylor, 1 point, degree of precision 1
    return {{{1.0 / 3.0, 1.0 / 3.0}}, {0.5}};
  }
  else if (m == 2)
  {
    // Scheme from Strang and Fix, 3 points, degree of precision 2
    xt::xtensor_fixed<double, xt::xshape<3, 2>> x = {
        {1.0 / 6.0, 1.0 / 6.0}, {1.0 / 6.0, 2.0 / 3.0}, {2.0 / 3.0, 1.0 / 6.0}};
    return {x, {1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0}};
  }
  else if (m == 3)
  {
    // Scheme from Strang and Fix, 6 points, degree of precision 3
    xt::xtensor_fixed<double, xt::xshape<6, 2>> x
        = {{0.659027622374092, 0.231933368553031},
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
    xt::xtensor_fixed<double, xt::xshape<6, 2>> x
        = {{0.816847572980459, 0.091576213509771},
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
    xt::xtensor_fixed<double, xt::xshape<7, 2>> x
        = {{0.33333333333333333, 0.33333333333333333},
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
    xt::xtensor_fixed<double, xt::xshape<12, 2>> x
        = {{0.873821971016996, 0.063089014491502},
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
  {
    const int np = (m + 2) / 2;
    return quadrature::make_quadrature_triangle_collapsed(np);
  }
}

} // namespace
//-----------------------------------------------------------------------------
xt::xtensor<double, 2>
quadrature::compute_jacobi_deriv(double a, std::size_t n, std::size_t nderiv,
                                 const tcb::span<const double>& x)
{
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

    xt::view(J, i, xt::all(), xt::all()) = Jd;
    // J.push_back(Jd);
  }

  xt::xtensor<double, 2> result({nderiv + 1, x.size()});
  for (std::size_t i = 0; i < nderiv + 1; ++i)
    xt::row(result, i) = xt::view(J, i, n, xt::all());

  return result;
}
//-----------------------------------------------------------------------------
std::vector<double> quadrature::compute_gauss_jacobi_points(double a, int m)
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
      tcb::span<const double> _x(&x[k], 1);
      const xt::xtensor<double, 2> f
          = quadrature::compute_jacobi_deriv(a, m, 1, _x);
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
quadrature::compute_gauss_jacobi_rule(double a, int m)
{
  /// @note Computes on [-1, 1]
  std::vector<double> _pts = quadrature::compute_gauss_jacobi_points(a, m);
  auto pts = xt::adapt(_pts);

  const xt::xtensor<double, 1> Jd
      = xt::row(quadrature::compute_jacobi_deriv(a, m, 1, pts), 1);

  const double a1 = std::pow(2.0, a + 1.0);
  const double a3 = std::tgamma(m + 1.0);
  // factorial(m)
  double a5 = 1.0;
  for (int i = 0; i < m; ++i)
    a5 *= (i + 1);
  const double a6 = a1 * a3 / a5;

  std::vector<double> wts(m);
  for (int i = 0; i < m; ++i)
  {
    const double x = pts[i];
    const double f = Jd[i];
    wts[i] = a6 / (1.0 - x * x) / (f * f);
  }

  std::cout << "weights:\n" << xt::adapt(wts) << std::endl;
  std::cout << "x:\n" << pts << std::endl;

  return {pts, wts};
}
//-----------------------------------------------------------------------------
std::pair<xt::xarray<double>, std::vector<double>>
quadrature::compute_gll_rule(int m)
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
std::pair<xt::xarray<double>, std::vector<double>>
quadrature::make_quadrature_line(int m)
{
  auto [ptx, wx] = quadrature::compute_gauss_jacobi_rule(0.0, m);
  std::transform(wx.begin(), wx.end(), wx.begin(),
                 [](auto x) { return 0.5 * x; });
  return {0.5 * (ptx + 1.0), wx};
}
//-----------------------------------------------------------------------------
std::pair<xt::xarray<double>, std::vector<double>>
quadrature::make_gll_line(int m)
{
  auto [ptx, wx] = quadrature::compute_gll_rule(m);
  std::transform(wx.begin(), wx.end(), wx.begin(),
                 [](auto x) { return 0.5 * x; });
  return {0.5 * (ptx + 1.0), wx};
}
//-----------------------------------------------------------------------------
std::pair<xt::xarray<double>, std::vector<double>>
quadrature::make_quadrature_triangle_collapsed(std::size_t m)
{
  auto [ptx, wx] = quadrature::compute_gauss_jacobi_rule(0.0, m);
  auto [pty, wy] = quadrature::compute_gauss_jacobi_rule(1.0, m);

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
quadrature::make_quadrature_tetrahedron_collapsed(std::size_t m)
{
  auto [ptx, wx] = quadrature::compute_gauss_jacobi_rule(0.0, m);
  auto [pty, wy] = quadrature::compute_gauss_jacobi_rule(1.0, m);
  auto [ptz, wz] = quadrature::compute_gauss_jacobi_rule(2.0, m);

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
quadrature::make_quadrature(const std::string& rule, cell::type celltype, int m)
{
  if (rule == "" or rule == "default")
  {
    if (celltype == cell::type::triangle)
      return make_default_triangle_quadrature(m);
    else if (celltype == cell::type::tetrahedron)
      return make_default_tetrahedron_quadrature(m);
    else
    {
      const int np = (m + 2) / 2;
      return make_gauss_jacobi_quadrature(celltype, np);
    }
  }
  else if (rule == "Gauss-Jacobi")
  {
    const int np = (m + 2) / 2;
    return make_gauss_jacobi_quadrature(celltype, np);
  }
  else if (rule == "GLL")
  {
    const int np = (m + 4) / 2;
    return make_gll_quadrature(celltype, np);
  }
  else
    throw std::runtime_error("Unknown quadrature rule \"" + rule + "\"");
}
//-----------------------------------------------------------------------------
