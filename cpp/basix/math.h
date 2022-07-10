// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "mdspan.hpp"
#include <array>
#include <vector>
#include <xtl/xspan.hpp>

/// Mathematical functions
///
/// @note The functions in this namespace are designed to be called
/// multiple times at runtime, so their performance is critical.
namespace basix::math
{

namespace impl
{
/// Compute C = A * B using BLAS
/// @param[in] A Input matrix
/// @param[in] B Input matrix
/// @return A * B
void dot_blas(const xtl::span<const double>& A,
              std::array<std::size_t, 2> Ashape,
              const xtl::span<const double>& B,
              std::array<std::size_t, 2> Bshape, const xtl::span<double>& C);
} // namespace impl

/// @brief Compute the outer product of vectors u and v.
/// @param u The first vector
/// @param v The second vector
/// @return The outer product. The type will be the same as `u`.
template <typename U, typename V>
std::pair<std::vector<typename U::value_type>, std::array<std::size_t, 2>>
outer(const U& u, const V& v)
{
  std::vector<typename U::value_type> result(u.size() * v.size());
  for (std::size_t i = 0; i < u.size(); ++i)
    for (std::size_t j = 0; j < v.size(); ++j)
      result[i * v.size() + j] = u[i] * v[j];

  return {std::move(result), {u.size(), v.size()}};
}

/// Compute the cross product u x v
/// @param u The first vector. It must has size 3.
/// @param v The second vector. It must has size 3.
/// @return The cross product `u x v`. The type will be the same as `u`.
template <typename U, typename V>
std::array<typename U::value_type, 3> cross(const U& u, const V& v)
{
  assert(u.size() == 3);
  assert(v.size() == 3);
  return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
          u[0] * v[1] - u[1] * v[0]};
}

/// Compute the eigenvalues and eigenvectors of a square Hermitian matrix A
/// @param[in] A Input matrix, row-major storage
/// @param[in] n Number of rows
/// @return Eigenvalues (0) and eigenvectors (1). The eigenvector array
/// uses column-major storage, which each column being an eigenvector.
/// @pre The matrix `A` must be symmetric
std::pair<std::vector<double>, std::vector<double>>
eigh(const xtl::span<const double>& A, std::size_t n);

/// Solve A X = B
/// @param[in] A The matrix
/// @param[in] B Right-hand side matrix/vector
/// @return A^{-1} B
std::vector<double>
solve(const std::experimental::mdspan<
          const double, std::experimental::dextents<std::size_t, 2>>& A,
      const std::experimental::mdspan<
          const double, std::experimental::dextents<std::size_t, 2>>& B);

/// Check if A is a singular matrix
/// @param[in] A The matrix
/// @return A bool indicating if the matrix is singular
bool is_singular(const std::experimental::mdspan<
                 const double, std::experimental::dextents<std::size_t, 2>>& A);

/// Compute C = A * B
/// @param[in] A Input matrix
/// @param[in] B Input matrix
/// @param[out] C Output matrix. Must be sized correctly before calling
/// this function.
template <typename U, typename V, typename W>
void dot(const U& A, const V& B, W&& C)
{
  assert(A.extent(1) == B.extent(0));
  assert(C.extent(0) == C.extent(0));
  assert(C.extent(1) == B.extent(1));

  int M = A.extent(0);
  int N = B.extent(1);
  int K = A.extent(1);

  if (M * N * K < 4096)
  {
    for (std::size_t i = 0; i < A.extent(0); ++i)
      for (std::size_t j = 0; j < B.extent(1); ++j)
        for (std::size_t k = 0; k < A.extent(1); ++k)
          C(i, j) += A(i, k) * B(k, j);
  }
  else
  {
    impl::dot_blas(A, {A.extent(0), A.extent(1)}, B, {B.extent(0), B.extent(1)},
                   C);
  }
}

/// Build an identity matrix
/// @param[in] n The number of rows/columns
/// @return Identity matrix using row-major storage
std::vector<double> eye(std::size_t n);

} // namespace basix::math
