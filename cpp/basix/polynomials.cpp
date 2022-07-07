// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "polynomials.h"
#include "polyset.h"
#include <xtensor/xview.hpp>

using namespace basix;
namespace stdex = std::experimental;

namespace
{
//-----------------------------------------------------------------------------
int single_choose(int n, int k)
{
  int out = 1;
  for (int i = k + 1; i <= n; ++i)
    out *= i;
  for (int i = 1; i <= n - k; ++i)
    out /= i;
  return out;
}
//-----------------------------------------------------------------------------
int choose(int n, const std::vector<int>& powers)
{
  int out = 1;
  for (std::size_t i = 0; i < powers.size(); ++i)
  {
    out *= single_choose(n, powers[i]);
    n -= powers[i];
  }
  return out;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> tabulate_bernstein(cell::type celltype, int d,
                                          const xt::xtensor<double, 2>& x)
{
  if (celltype != cell::type::interval and celltype != cell::type::triangle
      and celltype != cell::type::tetrahedron)
  {
    throw std::runtime_error("not implemented yet");
  }

  // TODO: implement a better Bernstein evaluation algorithm here

  const std::size_t pdim = dim(polynomials::type::bernstein, celltype, d);

  xt::xtensor<double, 2> values({pdim, x.shape(0)});

  xt::xtensor<double, 2> lambdas({x.shape(1) + 1, x.shape(0)});
  for (std::size_t j = 0; j < lambdas.shape(1); ++j)
    lambdas(0, j) = 1.0;
  for (std::size_t i = 0; i < x.shape(1); ++i)
  {
    for (std::size_t j = 0; j < x.shape(0); ++j)
    {
      lambdas(0, j) -= x(j, i);
      lambdas(i + 1, j) = x(j, i);
    }
  }

  std::vector<int> powers(lambdas.shape(0), 0);
  powers[0] = d;

  int n = 0;
  while (powers[0] >= 0)
  {
    {
      const int p = choose(d, powers);
      for (std::size_t j = 0; j < values.shape(1); ++j)
        values(n, j) = p;
    }

    for (std::size_t l = 0; l < lambdas.shape(0); ++l)
      for (int a = 0; a < powers[l]; ++a)
        for (std::size_t j = 0; j < values.shape(1); ++j)
          values(n, j) *= lambdas(l, j);

    powers[0] -= 1;
    powers[1] += 1;
    for (std::size_t i = 1; powers[0] < 0 and i + 1 < lambdas.shape(0); ++i)
    {
      powers[i] = 0;
      powers[i + 1] += 1;
      powers[0] = d;
      for (std::size_t j = 1; j < lambdas.shape(0); ++j)
        powers[0] -= powers[j];
    }

    ++n;
  }

  return values;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
xt::xtensor<double, 2> polynomials::tabulate(polynomials::type polytype,
                                             cell::type celltype, int d,
                                             const xt::xtensor<double, 2>& x)
{
  switch (polytype)
  {
  case polynomials::type::legendre:
    return xt::view(polyset::tabulate(celltype, d, 0, x), 0, xt::all(),
                    xt::all());
  case polynomials::type::bernstein:
    return tabulate_bernstein(celltype, d, x);
  default:
    throw std::runtime_error("not implemented yet");
  }
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 2>>
polynomials::tabulate(
    polynomials::type polytype, cell::type celltype, int d,
    std::experimental::mdspan<const double,
                              std::experimental::dextents<std::size_t, 2>>
        x)
{
  xt::xtensor<double, 2> _x({x.extent(0), x.extent(1)});
  std::copy_n(x.data(), x.size(), _x.data());
  xt::xtensor<double, 2> p = tabulate(polytype, celltype, d, _x);
  return {std::vector(p.data(), p.data() + p.size()), {p.shape(0), p.shape(1)}};
}
//-----------------------------------------------------------------------------
int polynomials::dim(polynomials::type, cell::type cell, int d)
{
  return polyset::dim(cell, d);
}
//-----------------------------------------------------------------------------
