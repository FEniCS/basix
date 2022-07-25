// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "math.h"
#include "mdspan.hpp"
#include <string>
#include <vector>

namespace stdex = std::experimental;

extern "C"
{
  void dsyevd_(char* jobz, char* uplo, int* n, double* a, int* lda, double* w,
               double* work, int* lwork, int* iwork, int* liwork, int* info);

  void dgesv_(int* N, int* NRHS, double* A, int* LDA, int* IPIV, double* B,
              int* LDB, int* INFO);

  void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha,
              double* a, int* lda, double* b, int* ldb, double* beta, double* c,
              int* ldc);

  int dgetrf_(const int* m, const int* n, double* a, const int* lda, int* lpiv,
              int* info);
}

//------------------------------------------------------------------
void basix::math::impl::dot_blas(const std::span<const double>& A,
                                 std::array<std::size_t, 2> Ashape,
                                 const std::span<const double>& B,
                                 std::array<std::size_t, 2> Bshape,
                                 const std::span<double>& C)
{
  assert(Ashape[1] == Bshape[0]);
  assert(C.size() == Ashape[0] * Bshape[1]);

  int M = Ashape[0];
  int N = Bshape[1];
  int K = Ashape[1];

  double alpha = 1;
  double beta = 0;
  int lda = K;
  int ldb = N;
  int ldc = N;
  char trans = 'N';
  dgemm_(&trans, &trans, &N, &M, &K, &alpha, const_cast<double*>(B.data()),
         &ldb, const_cast<double*>(A.data()), &lda, &beta, C.data(), &ldc);
}
//------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double>>
basix::math::eigh(const std::span<const double>& A, std::size_t n)
{
  // Copy A
  std::vector<double> M(A.begin(), A.end());

  // Allocate storage for eigenvalues
  std::vector<double> w(n, 0);

  int N = n;
  char jobz = 'V'; // Compute eigenvalues and eigenvectors
  char uplo = 'L'; // Lower
  int ldA = n;
  int lwork = -1;
  int liwork = -1;
  int info;
  std::vector<double> work(1);
  std::vector<int> iwork(1);

  // Query optimal workspace size
  dsyevd_(&jobz, &uplo, &N, M.data(), &ldA, w.data(), work.data(), &lwork,
          iwork.data(), &liwork, &info);
  if (info != 0)
    throw std::runtime_error("Could not find workspace size for syevd.");

  // Solve eigen problem
  work.resize(work[0]);
  iwork.resize(iwork[0]);
  lwork = work.size();
  liwork = iwork.size();
  dsyevd_(&jobz, &uplo, &N, M.data(), &ldA, w.data(), work.data(), &lwork,
          iwork.data(), &liwork, &info);
  if (info != 0)
    throw std::runtime_error("Eigenvalue computation did not converge.");

  return {std::move(w), std::move(M)};
}
//------------------------------------------------------------------
std::vector<double> basix::math::solve(
    const std::experimental::mdspan<
        const double, std::experimental::dextents<std::size_t, 2>>& A,
    const std::experimental::mdspan<
        const double, std::experimental::dextents<std::size_t, 2>>& B)
{
  // Copy A and B to column-major storage
  stdex::mdarray<double, stdex::dextents<std::size_t, 2>, stdex::layout_left>
      _A(A.extents()), _B(B.extents());
  for (std::size_t i = 0; i < A.extent(0); ++i)
    for (std::size_t j = 0; j < A.extent(1); ++j)
      _A(i, j) = A(i, j);
  for (std::size_t i = 0; i < B.extent(0); ++i)
    for (std::size_t j = 0; j < B.extent(1); ++j)
      _B(i, j) = B(i, j);

  int N = _A.extent(0);
  int nrhs = _B.extent(1);
  int lda = _A.extent(0);
  int ldb = _B.extent(0);
  // Pivot indices that define the permutation matrix for the LU solver
  std::vector<int> piv(N);
  int info;
  dgesv_(&N, &nrhs, _A.data(), &lda, piv.data(), _B.data(), &ldb, &info);
  if (info != 0)
    throw std::runtime_error("Call to dgesv failed: " + std::to_string(info));

  // Copy result to row-major storage
  std::vector<double> rb(_B.extent(0) * _B.extent(1));
  stdex::mdspan<double, stdex::dextents<std::size_t, 2>> r(rb.data(),
                                                           _B.extents());
  for (std::size_t i = 0; i < _B.extent(0); ++i)
    for (std::size_t j = 0; j < _B.extent(1); ++j)
      r(i, j) = _B(i, j);

  return rb;
}
//------------------------------------------------------------------
bool basix::math::is_singular(
    const std::experimental::mdspan<
        const double, std::experimental::dextents<std::size_t, 2>>& A)
{
  // Copy to column major matrix
  stdex::mdarray<double, stdex::dextents<std::size_t, 2>, stdex::layout_left>
      _A(A.extents());
  for (std::size_t i = 0; i < A.extent(0); ++i)
    for (std::size_t j = 0; j < A.extent(1); ++j)
      _A(i, j) = A(i, j);

  std::vector<double> B(A.extent(1), 1);
  int N = _A.extent(0);
  int nrhs = 1;
  int lda = _A.extent(0);
  int ldb = B.size();

  // Pivot indices that define the permutation matrix for the LU solver
  std::vector<int> piv(N);
  int info;
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
//------------------------------------------------------------------
std::vector<std::size_t> basix::math::lu_permutation(
    const std::experimental::mdspan<
        const double, std::experimental::dextents<std::size_t, 2>>& A)
{
  assert(A.extent(0) == A.extent(1));

  const std::size_t dim = A.extent(0);

  // Copy A
  std::vector<double> M(dim * dim);
  for (std::size_t i = 0; i < dim; ++i)
    for (std::size_t j = 0; j < dim; ++j)
      M[i * A.extent(1) + j] = A(i, j);

  int N = dim;
  int info;
  std::vector<int> lu_perm(dim);

  // Comput LU decomposition of M
  dgetrf_(&N, &N, M.data(), &N, lu_perm.data(), &info);

  std::vector<std::size_t> perm(dim);
  for (std::size_t i = 0; i < dim; ++i)
    perm[i] = static_cast<std::size_t>(lu_perm[i] - 1);

  return perm;
}
//------------------------------------------------------------------
std::vector<double> basix::math::eye(std::size_t n)
{
  std::vector<double> I(n * n, 0);
  stdex::mdspan<double, stdex::dextents<std::size_t, 2>> Iview(I.data(), n, n);
  for (std::size_t i = 0; i < n; ++i)
    Iview(i, i) = 1.0;
  return I;
}
//------------------------------------------------------------------
