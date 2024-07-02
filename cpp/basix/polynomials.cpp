// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "polynomials.h"
#include "mdspan.hpp"
#include "polyset.h"
#include <concepts>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace basix;
namespace stdex
    = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;

namespace
{
template <typename T, std::size_t d>
using mdarray_t
    = stdex::mdarray<T,
                     MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, d>>;
template <typename T, std::size_t d>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, d>>;

//-----------------------------------------------------------------------------
constexpr int single_choose(int n, int k)
{
  int out = 1;
  for (int i = n + 1 - k; i <= n; ++i)
    out *= i;
  for (int i = 1; i <= k; ++i)
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
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
tabulate_bernstein(cell::type celltype, int d, mdspan_t<const T, 2> x)
{
  if (celltype != cell::type::interval and celltype != cell::type::triangle
      and celltype != cell::type::tetrahedron)
  {
    throw std::runtime_error("not implemented yet");
  }

  // TODO: implement a better Bernstein evaluation algorithm here

  const std::size_t pdim = dim(polynomials::type::bernstein, celltype, d);

  std::array<std::size_t, 2> shape = {pdim, x.extent(0)};
  std::vector<T> values_b(shape[0] * shape[1]);
  mdspan_t<T, 2> values(values_b.data(), shape);

  mdarray_t<T, 2> lambdas(x.extent(1) + 1, x.extent(0));
  for (std::size_t j = 0; j < lambdas.extent(1); ++j)
    lambdas[0, j] = 1.0;
  for (std::size_t i = 0; i < x.extent(1); ++i)
  {
    for (std::size_t j = 0; j < x.extent(0); ++j)
    {
      lambdas[0, j] -= x[j, i];
      lambdas[i + 1, j] = x[j, i];
    }
  }

  std::vector<int> powers(lambdas.extent(0), 0);
  powers[0] = d;

  int n = 0;
  while (powers[0] >= 0)
  {
    {
      const int p = choose(d, powers);
      for (std::size_t j = 0; j < values.extent(1); ++j)
        values[n, j] = p;
    }

    for (std::size_t l = 0; l < lambdas.extent(0); ++l)
      for (int a = 0; a < powers[l]; ++a)
        for (std::size_t j = 0; j < values.extent(1); ++j)
          values[n, j] *= lambdas[l, j];

    powers[0] -= 1;
    powers[1] += 1;
    for (std::size_t i = 1; powers[0] < 0 and i + 1 < powers.size(); ++i)
    {
      powers[i] = 0;
      powers[i + 1] += 1;
      powers[0] = d;
      for (std::size_t j = 1; j < powers.size(); ++j)
        powers[0] -= powers[j];
    }

    ++n;
  }

  return {std::move(values_b), std::move(shape)};
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>> polynomials::tabulate(
    polynomials::type polytype, cell::type celltype, int d,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x)
{
  switch (polytype)
  {
  case polynomials::type::legendre:
  {
    auto [values, shape]
        = polyset::tabulate(celltype, polyset::type::standard, d, 0, x);
    assert(shape[0] == 1);
    return {std::move(values), {shape[1], shape[2]}};
  }
  case polynomials::type::bernstein:
  {
    auto [values, shape] = tabulate_bernstein(celltype, d, x);
    return {std::move(values), std::move(shape)};
  }
  default:
    throw std::runtime_error("not implemented yet");
  }
}
//-----------------------------------------------------------------------------
int polynomials::dim(polynomials::type, cell::type cell, int d)
{
  return polyset::dim(cell, polyset::type::standard, d);
}
//-----------------------------------------------------------------------------
/// @cond
template std::pair<std::vector<float>, std::array<std::size_t, 2>>
polynomials::tabulate(
    polynomials::type, cell::type, int,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const float, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>);
template std::pair<std::vector<double>, std::array<std::size_t, 2>>
polynomials::tabulate(
    polynomials::type, cell::type, int,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const double,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>);
/// @endcond
//-----------------------------------------------------------------------------
