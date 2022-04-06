// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xtensor.hpp>

/// Mathematical functions

/// @note The functions in this namespace are designed to be called multiple
/// times at runtime, so their performance is critical.
namespace basix::math
{

/// Compute the outer product of vectors u and v
/// @param u The first vector. It must has size 3.
/// @param v The second vector. It must has size 3.
/// @return The outer product. The type will be the same as `u`.
template <typename U, typename V>
xt::xtensor<typename U::value_type, 2> outer(const U& u, const V& v)
{
  xt::xtensor<typename U::value_type, 2> results({u.size(), v.size()});
  for (std::size_t i = 0; i < u.size(); i++)
    for (std::size_t j = 0; j < u.size(); j++)
      results(i, j) = u(i) * v(j);

  return results;
}

/// Compute the cross product u x v
/// @param u The first vector. It must has size 3.
/// @param v The second vector. It must has size 3.
/// @return The cross product `u x v`. The type will be the same as `u`.
template <typename U, typename V>
xt::xtensor_fixed<typename U::value_type, xt::xshape<3>> cross(const U& u,
                                                               const V& v)
{
  assert(u.size() == 3);
  assert(v.size() == 3);
  return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
          u[0] * v[1] - u[1] * v[0]};
}

/// Compute C = A * B
/// @param[in] A Input matrix
/// @param[in] B Input matrix
/// return A * B
template <typename U, typename V>
xt::xtensor<typename U::value_type, 2> dot(const U& A, const V& B)
{
  xt::xtensor<typename U::value_type, 2> C
      = xt::zeros<typename U::value_type>({A.shape(0), B.shape(1)});

  assert(A.shape(1) == B.shape(0));
  for (std::size_t i = 0; i < A.shape(0); i++)
    for (std::size_t j = 0; j < B.shape(1); j++)
      for (std::size_t k = 0; k < A.shape(1); k++)
        C(i, j) += A(i, k) * B(k, j);

  return C;
}

/// Compute the eigenvalues and eigenvectors of a square Hermitian matrix A
/// @param[in] A Input matrix
/// @return Eigenvalues and eigenvectors
std::pair<xt::xtensor<double, 1>,
          xt::xtensor<double, 2, xt::layout_type::column_major>>
eigh(const xt::xtensor<double, 2>& A);

/// Solve A X = B
/// @param[in] A The matrix
/// @param[in] B Right-hand side matrix/vector
/// @return A^{-1} B
xt::xarray<double, xt::layout_type::column_major>
solve(const xt::xtensor<double, 2>& A, const xt::xarray<double>& B);

/// Check if A is a singular matrix
/// @param[in] A The matrix
/// @return A bool indicating if the matrix is singular
bool is_singular(const xt::xtensor<double, 2>& A);

} // namespace basix::math
