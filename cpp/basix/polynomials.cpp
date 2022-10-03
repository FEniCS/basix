// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "polynomials.h"
#include "mdspan.hpp"
#include "polyset.h"
#include <iostream>
#include <utility>
#include <vector>

using namespace basix;
namespace stdex = std::experimental;
using mdarray2_t = stdex::mdarray<double, stdex::dextents<std::size_t, 2>>;
using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
using cmdspan3_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 3>>;

namespace
{
//-----------------------------------------------------------------------------
constexpr int single_choose(int n, int k)
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
std::pair<std::vector<double>, std::array<std::size_t, 2>>
tabulate_bernstein(cell::type celltype, int d, cmdspan2_t x)
{
  if (celltype != cell::type::interval and celltype != cell::type::triangle
      and celltype != cell::type::tetrahedron)
  {
    throw std::runtime_error("not implemented yet");
  }

  // TODO: implement a better Bernstein evaluation algorithm here

  const std::size_t pdim = dim(polynomials::type::bernstein, celltype, d);
  const std::size_t tdim = cell::topological_dimension(celltype);
  if (tdim != x.extent(1))
    throw std::runtime_error("x does not match cell dimension");

  std::array<std::size_t, 2> shape = {pdim, x.extent(0)};
  std::vector<double> values_b(shape[0] * shape[1]);
  mdspan2_t values(values_b.data(), shape);

  // Barycentric coordinate lambda (transposing x)
  mdarray2_t lambdas(tdim + 1, x.extent(0));
  for (std::size_t j = 0; j < lambdas.extent(1); ++j)
    lambdas(0, j) = 1.0;
  for (std::size_t i = 0; i < tdim; ++i)
  {
    for (std::size_t j = 0; j < x.extent(0); ++j)
    {
      lambdas(0, j) -= x(j, i);
      lambdas(i + 1, j) = x(j, i);
    }
  }

  // Recurrence relation: B(i,j,k,n) = lambda_0 * B(i,j,k,n-1) + lambda_1 *
  // B(i-1,j,k,n-1) + lambda_2 * B(i, j-1, k, n-1) + lambda_3 * B(i, j, k-1,
  // n-1)

  std::vector<double> B_vec(shape[0] * shape[1]);
  mdspan2_t B(B_vec.data(), shape);
  // Set B0 = 1.0
  for (std::size_t i = 0; i < shape[1]; ++i)
    B(0, i) = 1.0;

  // 2D version
  auto idx = [&d](int i, int j) { return i + ((2 * d + 3) * j - j * j) / 2; };

  // Work up through n
  for (std::size_t n = 1; n < d + 1; ++n)
  {
    // Start with i+j=n and work back down to i=j=0
    for (std::size_t w = 0; w < n + 1; ++w)
    {
      for (std::size_t i = 0; i < n - w + 1; ++i)
      {
        const std::size_t j = n - w - i;
        const int index = idx(i, j);
        // B(i, j, n) = lambdas[0] * B(i, j, n-1) + lambdas[1] * B(i-1, j, n-1)
        // + lambdas[2] * B(i, j-1, n-1)

        if (i + j < n)
          for (std::size_t p = 0; p < shape[1]; ++p)
            B(index, p) *= lambdas(0, p);
        if (i > 0)
        {
          const int idxi1 = idx(i - 1, j);
          for (std::size_t p = 0; p < shape[1]; ++p)
            B(index, p) += lambdas(1, p) * B(idxi1, p);
        }
        if (j > 0)
        {
          const int idxj1 = idx(i, j - 1);
          for (std::size_t p = 0; p < shape[1]; ++p)
            B(index, p) += lambdas(2, p) * B(idxj1, p);
        }

        //   for (std::size_t p = 0; p < shape[1]; ++p)
        //   {
        //     double Bij = (i + j < n) ? B(index, p) : 0.0;
        //     double Bim1 = (i > 0) ? B(idx(i - 1, j), p) : 0.0;
        //     double Bjm1 = (j > 0) ? B(idx(i, j - 1), p) : 0.0;
        //     B(index, p) = lambdas(0, p) * Bij + lambdas(1, p) * Bim1
        //                   + lambdas(2, p) * Bjm1;
        //     std::cout << "n = " << n << "  (" << i << "," << j << ") =
        //     Index["
        //               << index << "] = l0*" << Bij << " + l1*" << Bim1 << "+
        //               l2*"
        //               << Bjm1 << "\n";
        //   }
      }
    }
  }

  std::vector<int> powers(lambdas.extent(0), 0);
  powers[0] = d;

  int n = 0;
  while (powers[0] >= 0)
  {
    const int p = choose(d, powers);
    for (std::size_t j = 0; j < values.extent(1); ++j)
      values(n, j) = p;

    for (std::size_t l = 0; l < lambdas.extent(0); ++l)
      for (int a = 0; a < powers[l]; ++a)
        for (std::size_t j = 0; j < values.extent(1); ++j)
          values(n, j) *= lambdas(l, j);

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

  for (int i = 0; i < values.size(); ++i)
    std::cout << values_b[i] << " = " << B_vec[i] << "\n";

  return {std::move(B_vec), std::move(shape)};
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 2>>
polynomials::tabulate(
    polynomials::type polytype, cell::type celltype, int d,
    std::experimental::mdspan<const double,
                              std::experimental::dextents<std::size_t, 2>>
        x)
{
  switch (polytype)
  {
  case polynomials::type::legendre:
  {
    auto [values, shape] = polyset::tabulate(celltype, d, 0, x);
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
  return polyset::dim(cell, d);
}
//-----------------------------------------------------------------------------
