// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mdspan.hpp"

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

  void sgemm_(char* transa, char* transb, int* m, int* n, int* k, float* alpha,
              float* a, int* lda, float* b, int* ldb, float* beta, float* c,
              int* ldc);
  void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha,
              double* a, int* lda, double* b, int* ldb, double* beta, double* c,
              int* ldc);

  int sgetrf_(const int* m, const int* n, float* a, const int* lda, int* lpiv,
              int* info);
  int dgetrf_(const int* m, const int* n, double* a, const int* lda, int* lpiv,
              int* info);
}

/// @brief Mathematical functions.
///
/// @note The functions in this namespace are designed to be called
/// multiple times at runtime, so their performance is critical.
namespace basix::math
{
namespace impl
{
/// @brief Compute C = A * B using BLAS.
/// @param[in] A Input matrix
/// @param[in] B Input matrix
/// @return A * B
template <std::floating_point T>
void dot_blas(std::span<const T> A, std::array<std::size_t, 2> Ashape,
              std::span<const T> B, std::array<std::size_t, 2> Bshape,
              std::span<T> C)
{
  static_assert(std::is_same_v<T, float> or std::is_same_v<T, double>);

  assert(Ashape[1] == Bshape[0]);
  assert(C.size() == Ashape[0] * Bshape[1]);

  int M = Ashape[0];
  int N = Bshape[1];
  int K = Ashape[1];

  T alpha = 1;
  T beta = 0;
  int lda = K;
  int ldb = N;
  int ldc = N;
  char trans = 'N';
  if constexpr (std::is_same_v<T, float>)
  {
    sgemm_(&trans, &trans, &N, &M, &K, &alpha, const_cast<T*>(B.data()), &ldb,
           const_cast<T*>(A.data()), &lda, &beta, C.data(), &ldc);
  }
  else if constexpr (std::is_same_v<T, double>)
  {
    dgemm_(&trans, &trans, &N, &M, &K, &alpha, const_cast<T*>(B.data()), &ldb,
           const_cast<T*>(A.data()), &lda, &beta, C.data(), &ldc);
  }
}

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
std::pair<std::vector<T>, std::vector<T>> eigh(std::span<const T> A,
                                               std::size_t n)
{
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

/// @brief Solve A X = B.
/// @param[in] A The matrix
/// @param[in] B Right-hand side matrix/vector
/// @return A^{-1} B
template <std::floating_point T>
std::vector<T>
solve(MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
          const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
          A,
      MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
          const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
          B)
{
  namespace stdex
      = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;

  // Copy A and B to column-major storage
  stdex::mdarray<T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>,
                 MDSPAN_IMPL_STANDARD_NAMESPACE::layout_left>
      _A(A.extents()), _B(B.extents());
  for (std::size_t i = 0; i < A.extent(0); ++i)
    for (std::size_t j = 0; j < A.extent(1); ++j)
      _A[i, j] = A[i, j];
  for (std::size_t i = 0; i < B.extent(0); ++i)
    for (std::size_t j = 0; j < B.extent(1); ++j)
      _B[i, j] = B[i, j];

  int N = _A.extent(0);
  int nrhs = _B.extent(1);
  int lda = _A.extent(0);
  int ldb = _B.extent(0);
  // Pivot indices that define the permutation matrix for the LU solver
  std::vector<int> piv(N);
  int info;
  if constexpr (std::is_same_v<T, float>)
    sgesv_(&N, &nrhs, _A.data(), &lda, piv.data(), _B.data(), &ldb, &info);
  else if constexpr (std::is_same_v<T, double>)
    dgesv_(&N, &nrhs, _A.data(), &lda, piv.data(), _B.data(), &ldb, &info);
  if (info != 0)
    throw std::runtime_error("Call to dgesv failed: " + std::to_string(info));

  // Copy result to row-major storage
  std::vector<T> rb(_B.extent(0) * _B.extent(1));
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      r(rb.data(), _B.extents());
  for (std::size_t i = 0; i < _B.extent(0); ++i)
    for (std::size_t j = 0; j < _B.extent(1); ++j)
      r[i, j] = _B[i, j];

  return rb;
}

/// @brief Check if A is a singular matrix,
/// @param[in] A The matrix
/// @return A bool indicating if the matrix is singular
template <std::floating_point T>
bool is_singular(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        A)
{
  // Copy to column major matrix
  namespace stdex
      = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;
  stdex::mdarray<T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>,
                 MDSPAN_IMPL_STANDARD_NAMESPACE::layout_left>
      _A(A.extents());
  for (std::size_t i = 0; i < A.extent(0); ++i)
    for (std::size_t j = 0; j < A.extent(1); ++j)
      _A[i, j] = A[i, j];

  std::vector<T> B(A.extent(1), 1);
  int N = _A.extent(0);
  int nrhs = 1;
  int lda = _A.extent(0);
  int ldb = B.size();

  // Pivot indices that define the permutation matrix for the LU solver
  std::vector<int> piv(N);
  int info;
  if constexpr (std::is_same_v<T, float>)
    sgesv_(&N, &nrhs, _A.data(), &lda, piv.data(), B.data(), &ldb, &info);
  else if constexpr (std::is_same_v<T, double>)
    dgesv_(&N, &nrhs, _A.data(), &lda, piv.data(), B.data(), &ldb, &info);

  if (info < 0)
  {
    throw std::runtime_error("dgesv failed due to invalid value: "
                             + std::to_string(info));
  }
  else if (info > 0)
    return true;
  else
    return false;
}

/// @brief Compute the LU decomposition of the transpose of a square
/// matrix A.
/// @param[in,out] A The matrix
/// @return The LU permutation, in prepared format (see
/// precompute::prepare_permutation)
template <std::floating_point T>
std::vector<std::size_t>
transpose_lu(std::pair<std::vector<T>, std::array<std::size_t, 2>>& A)
{
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

/// @brief Compute C = A * B.
/// @param[in] A Input matrix
/// @param[in] B Input matrix
/// @param[out] C Output matrix. Must be sized correctly before calling
/// this function.
template <typename U, typename V, typename W>
void dot(const U& A, const V& B, W&& C)
{
  assert(A.extent(1) == B.extent(0));
  assert(C.extent(0) == A.extent(0));
  assert(C.extent(1) == B.extent(1));
  if (A.extent(0) * B.extent(1) * A.extent(1) < 512)
  {
    std::fill_n(C.data_handle(), C.extent(0) * C.extent(1), 0);
    for (std::size_t i = 0; i < A.extent(0); ++i)
      for (std::size_t j = 0; j < B.extent(1); ++j)
        for (std::size_t k = 0; k < A.extent(1); ++k)
          C[i, j] += A[i, k] * B[k, j];
  }
  else
  {
    using T = typename std::decay_t<U>::value_type;
    impl::dot_blas<T>(
        std::span(A.data_handle(), A.size()), {A.extent(0), A.extent(1)},
        std::span(B.data_handle(), B.size()), {B.extent(0), B.extent(1)},
        std::span(C.data_handle(), C.size()));
  }
}

/// @brief Build an identity matrix.
/// @param[in] n The number of rows/columns
/// @return Identity matrix using row-major storage
template <std::floating_point T>
std::vector<T> eye(std::size_t n)
{
  std::vector<T> I(n * n, 0);
  namespace stdex
      = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      Iview(I.data(), n, n);
  for (std::size_t i = 0; i < n; ++i)
    Iview[i, i] = 1;
  return I;
}

/// @brief Orthogonalise the rows of a matrix (in place).
/// @param[in] wcoeffs The matrix
/// @param[in] start The row to start from. The rows before this should
/// already be orthogonal.
template <std::floating_point T>
void orthogonalise(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        wcoeffs,
    std::size_t start = 0)
{
  for (std::size_t i = start; i < wcoeffs.extent(0); ++i)
  {
    T norm = 0;
    for (std::size_t k = 0; k < wcoeffs.extent(1); ++k)
      norm += wcoeffs[i, k] * wcoeffs[i, k];

    norm = std::sqrt(norm);
    if (norm < 2 * std::numeric_limits<T>::epsilon())
    {
      throw std::runtime_error(
          "Cannot orthogonalise the rows of a matrix with incomplete row rank");
    }

    for (std::size_t k = 0; k < wcoeffs.extent(1); ++k)
      wcoeffs[i, k] /= norm;

    for (std::size_t j = i + 1; j < wcoeffs.extent(0); ++j)
    {
      T a = 0;
      for (std::size_t k = 0; k < wcoeffs.extent(1); ++k)
        a += wcoeffs[i, k] * wcoeffs[j, k];
      for (std::size_t k = 0; k < wcoeffs.extent(1); ++k)
        wcoeffs[j, k] -= a * wcoeffs[i, k];
    }
  }
}
} // namespace basix::math
