// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "span.hpp"
#include <string>
#include <vector>
#include <xtensor/xexpression.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

/// Matrix vector and other products
namespace basix
{
/// Calculate a matrix-matrix product.
xt::xtensor<double, 2> dot22(xt::xtensor<double, 2> A, xt::xtensor<double, 2> B)
{
  assert(A.shape(1) == B.shape(0));

  std::array<std::size_t, 2> s = {A.shape(0), B.shape(1)};

  xt::xtensor<double, 2> r(s);
  for (std::size_t i = 0; i < s[0]; ++i)
  {
    xt::view(r, i, xt::all()) = 0;
    for (std::size_t k = 0; k < A.shape(1); ++k)
      xt::view(r, i, xt::all()) += A(i, k) * xt::view(B, k, xt::all());
  }
  return r;
}
//-----------------------------------------------------------------------------
/// Calculate a matrix-vector product.
xt::xtensor<double, 1> dot21(xt::xtensor<double, 2> A, xt::xtensor<double, 1> B)
{
  assert(A.shape(1) == B.shape(0));

  std::array<std::size_t, 1> s = {A.shape(0)};

  xt::xtensor<double, 1> r(s);
  for (std::size_t i = 0; i < s[0]; ++i)
  {
    xt::view(r, i, xt::all()) = 0;
    for (std::size_t k = 0; k < A.shape(1); ++k)
      xt::view(r, i, xt::all()) += A(i, k) * xt::view(B, k, xt::all());
  }
  return r;
}
//-----------------------------------------------------------------------------
} // namespace basix
