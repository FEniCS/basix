// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "mdspan.hpp"
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

/// Compute the outer product of vectors u and v
/// @param u The first vector. It must has size 3.
/// @param v The second vector. It must has size 3.
/// @return The outer product. The type will be the same as `u`.
template <typename U, typename V>
std::pair<std::vector<typename U::value_type>, std::array<std::size_t, 2>>
outer_new(const U& u, const V& v)
{
  std::vector<typename U::value_type> result(u.size() * v.size());
  for (std::size_t i = 0; i < u.size(); i++)
    for (std::size_t j = 0; j < u.size(); j++)
      result[i * v.size() + j] = u[i] * v[j];

  return {std::move(result), {u.size(), v.size()}};
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

/// Compute the cross product u x v
/// @param u The first vector. It must has size 3.
/// @param v The second vector. It must has size 3.
/// @return The cross product `u x v`. The type will be the same as `u`.
template <typename U, typename V>
std::array<typename U::value_type, 3> cross_new(const U& u, const V& v)
{
  assert(u.size() == 3);
  assert(v.size() == 3);
  return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
          u[0] * v[1] - u[1] * v[0]};
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
xt::xtensor<double, 2> solve(const xt::xtensor<double, 2>& A,
                             const xt::xtensor<double, 2>& B);

/// Solve A X = B
/// @param[in] A The matrix
/// @param[in] B Right-hand side matrix/vector
/// @return A^{-1} B
std::vector<double>
solve(const std::experimental::mdspan<
          double, std::experimental::dextents<std::size_t, 2>>& A,
      const std::experimental::mdspan<
          double, std::experimental::dextents<std::size_t, 2>>& B);

/// Check if A is a singular matrix
/// @param[in] A The matrix
/// @return A bool indicating if the matrix is singular
bool is_singular(const xt::xtensor<double, 2>& A);

/// Compute C = A * B
/// @param[in] A Input matrix
/// @param[in] B Input matrix
/// @param[in, out] C The output matrix
void dot(const xt::xtensor<double, 2>& A, const xt::xtensor<double, 2>& B,
         xt::xtensor<double, 2>& C);

/// Compute C = A * B
/// @param[in] A Input matrix
/// @param[in] B Input matrix
/// @return A * B
xt::xtensor<double, 2> dot(const xt::xtensor<double, 2>& A,
                           const xt::xtensor<double, 2>& B);

/// Build an identity matrix
/// @param[in] n The number of rows/columns
/// @return Identity matrix using row-major storage
std::vector<double> eye(std::size_t n);

} // namespace basix::math
