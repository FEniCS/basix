// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "mdspan.hpp"
#include <array>
#include <concepts>
#include <span>
#include <string>
#include <utility>
#include <vector>

extern "C"
{
  void ssyevd_(char* jobz, char* uplo, int* n, float* a, int* lda, float* w,
               float* work, int* lwork, int* iwork, int* liwork, int* info);
  void dsyevd_(char* jobz, char* uplo, int* n, double* a, int* lda, double* w,
               double* work, int* lwork, int* iwork, int* liwork, int* info);

  void sgesv_(int* N, int* NRHS, float* A, int* LDA, int* IPIV, float* B,
              int* LDB, int* INFO);
  void dgesv_(int* N, int* NRHS, double* A, int* LDA, int* IPIV, double* B,
              int* LDB, int* INFO);

  void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha,
              double* a, int* lda, double* b, int* ldb, double* beta, double* c,
              int* ldc);

  int sgetrf_(const int* m, const int* n, float* a, const int* lda, int* lpiv,
              int* info);
  int dgetrf_(const int* m, const int* n, double* a, const int* lda, int* lpiv,
              int* info);
}

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
void dot_blas(const std::span<const double>& A,
              std::array<std::size_t, 2> Ashape,
              const std::span<const double>& B,
              std::array<std::size_t, 2> Bshape, const std::span<double>& C);
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
template <std::floating_point T>
std::pair<std::vector<T>, std::vector<T>> eigh(const std::span<const T>& A,
                                               std::size_t n)
{
  static_assert(std::is_same_v<T, float> or std::is_same_v<T, double>);

  // Copy A
  std::vector<T> M(A.begin(), A.end());

  // Allocate storage for eigenvalues
  std::vector<T> w(n, 0);

  int N = n;
  char jobz = 'V'; // Compute eigenvalues and eigenvectors
  char uplo = 'L'; // Lower
  int ldA = n;
  int lwork = -1;
  int liwork = -1;
  int info;
  std::vector<T> work(1);
  std::vector<int> iwork(1);

  // Query optimal workspace size
  if constexpr (std::is_same_v<T, float>)
  {
    ssyevd_(&jobz, &uplo, &N, M.data(), &ldA, w.data(), work.data(), &lwork,
            iwork.data(), &liwork, &info);
  }
  else if constexpr (std::is_same_v<T, double>)
  {
    dsyevd_(&jobz, &uplo, &N, M.data(), &ldA, w.data(), work.data(), &lwork,
            iwork.data(), &liwork, &info);
  }

  if (info != 0)
    throw std::runtime_error("Could not find workspace size for syevd.");

  // Solve eigen problem
  work.resize(work[0]);
  iwork.resize(iwork[0]);
  lwork = work.size();
  liwork = iwork.size();
  if constexpr (std::is_same_v<T, float>)
  {
    ssyevd_(&jobz, &uplo, &N, M.data(), &ldA, w.data(), work.data(), &lwork,
            iwork.data(), &liwork, &info);
  }
  else if constexpr (std::is_same_v<T, double>)
  {
    dsyevd_(&jobz, &uplo, &N, M.data(), &ldA, w.data(), work.data(), &lwork,
            iwork.data(), &liwork, &info);
  }
  if (info != 0)
    throw std::runtime_error("Eigenvalue computation did not converge.");

  return {std::move(w), std::move(M)};
}

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

/// Compute the LU decomposition of the transpose of a square matrix A
/// @param[in,out] A The matrix
/// @return The LU permutation, in prepared format (see
/// `basix::precompute::prepare_permutation`)
template <std::floating_point T>
std::vector<std::size_t>
transpose_lu(std::pair<std::vector<T>, std::array<std::size_t, 2>>& A)
{
  static_assert(std::is_same_v<T, float> or std::is_same_v<T, double>);

  std::size_t dim = A.second[0];
  assert(dim == A.second[1]);
  int N = dim;
  int info;
  std::vector<int> lu_perm(dim);

  // Comput LU decomposition of M
  if constexpr (std::is_same_v<T, float>)
    sgetrf_(&N, &N, A.first.data(), &N, lu_perm.data(), &info);
  else if constexpr (std::is_same_v<T, double>)
    dgetrf_(&N, &N, A.first.data(), &N, lu_perm.data(), &info);

  if (info != 0)
  {
    throw std::runtime_error("LU decomposition failed: "
                             + std::to_string(info));
  }

  std::vector<std::size_t> perm(dim);
  for (std::size_t i = 0; i < dim; ++i)
    perm[i] = static_cast<std::size_t>(lu_perm[i] - 1);

  return perm;
}

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
  if (A.extent(0) * B.extent(1) * A.extent(1) < 4096)
  {
    std::fill_n(C.data_handle(), C.extent(0) * C.extent(1), 0);
    for (std::size_t i = 0; i < A.extent(0); ++i)
      for (std::size_t j = 0; j < B.extent(1); ++j)
        for (std::size_t k = 0; k < A.extent(1); ++k)
          C(i, j) += A(i, k) * B(k, j);
  }
  else
  {
    impl::dot_blas(
        std::span(A.data_handle(), A.size()), {A.extent(0), A.extent(1)},
        std::span(B.data_handle(), B.size()), {B.extent(0), B.extent(1)},
        std::span(C.data_handle(), C.size()));
  }
}

/// Build an identity matrix
/// @param[in] n The number of rows/columns
/// @return Identity matrix using row-major storage
std::vector<double> eye(std::size_t n);

} // namespace basix::math
