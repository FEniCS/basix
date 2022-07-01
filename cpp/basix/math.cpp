// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "math.h"
#include "mdspan.hpp"
#include <vector>
#include <xtensor/xadapt.hpp>

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
}

//------------------------------------------------------------------
std::pair<xt::xtensor<double, 1>,
          xt::xtensor<double, 2, xt::layout_type::column_major>>
basix::math::eigh(const xt::xtensor<double, 2>& A)
{
  // Copy to column major matrix
  xt::xtensor<double, 2, xt::layout_type::column_major> M(A.shape());
  M.assign(A);
  int N = A.shape(0);
  xt::xtensor<double, 1> w = xt::zeros<double>({N});

  char jobz = 'V'; // Compute eigenvalues and eigenvectors
  char uplo = 'L'; // Lower
  int ldA = A.shape(1);
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
std::pair<std::vector<double>, std::vector<double>>
basix::math::eigh(const xtl::span<const double>& A, std::size_t n)
{
  // Copy to column major matrix
  // xt::xtensor<double, 2, xt::layout_type::column_major> M(A.shape());
  // M.assign(A);
  // int N = A.shape(0);
  std::vector<double> w(n, 0);
  std::vector<double> M(A.begin(), A.end());

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
xt::xtensor<double, 2> basix::math::solve(const xt::xtensor<double, 2>& A,
                                          const xt::xtensor<double, 2>& B)
{
  // Copy to column major matrix
  xt::xtensor<double, 2, xt::layout_type::column_major> _A(A.shape());
  _A.assign(A);
  xt::xtensor<double, 2, xt::layout_type::column_major> _B(B.shape());
  _B.assign(B);

  int N = _A.shape(0);
  int nrhs = _B.shape(1);
  int lda = _A.shape(0);
  int ldb = B.shape(0);
  // Pivot indices that define the permutation matrix for the LU solver
  std::vector<int> piv(N);
  int info;
  dgesv_(&N, &nrhs, _A.data(), &lda, piv.data(), _B.data(), &ldb, &info);
  if (info != 0)
    throw std::runtime_error("Call to dgesv failed: " + std::to_string(info));

  // Note: using assign, instead of returning the object, to get around an
  // xtensor bug with Intel Compilers
  // https://github.com/xtensor-stack/xtensor/issues/2351
  xt::xtensor<double, 2> out(_B.shape());
  out.assign(_B);

  return out;
}
//------------------------------------------------------------------
std::vector<double>
basix::math::solve(const std::experimental::mdspan<
                       double, std::experimental::dextents<std::size_t, 2>>& A,
                   const std::experimental::mdspan<
                       double, std::experimental::dextents<std::size_t, 2>>& B)
{
  auto _A = xt::adapt(A.data(), A.size(), xt::no_ownership(),
                      std::array<std::size_t, 2>{A.extent(0), A.extent(1)});
  auto _B = xt::adapt(B.data(), B.size(), xt::no_ownership(),
                      std::array<std::size_t, 2>{B.extent(0), B.extent(1)});

  xt::xtensor<double, 2> C = solve(_A, _B);
  return std::vector<double>(C.data(), C.data() + C.size());
}
//------------------------------------------------------------------
bool basix::math::is_singular(const xt::xtensor<double, 2>& A)
{
  // Copy to column major matrix
  xt::xtensor<double, 2, xt::layout_type::column_major> _A(A.shape());
  _A.assign(A);
  const std::array<std::size_t, 1> sh = {A.shape(1)};
  xt::xtensor<double, 1> B(sh);
  B.fill(1.0);

  int N = _A.shape(0);
  int nrhs = 1;
  int lda = _A.shape(0);
  int ldb = B.shape(0);
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
void basix::math::dot(const xt::xtensor<double, 2>& A,
                      const xt::xtensor<double, 2>& B,
                      xt::xtensor<double, 2>& C)
{
  assert(A.shape(1) == B.shape(0));
  assert(C.shape(0) == C.shape(0));
  assert(C.shape(1) == B.shape(1));

  int M = A.shape(0);
  int N = B.shape(1);
  int K = A.shape(1);

  if (M * N * K < 4096)
  {
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j)
        for (int k = 0; k < K; ++k)
          C(i, j) += A(i, k) * B(k, j);
  }
  else
  {
    double alpha = 1;
    double beta = 0;
    int lda = K;
    int ldb = N;
    int ldc = N;
    char trans = 'N';
    dgemm_(&trans, &trans, &N, &M, &K, &alpha, const_cast<double*>(B.data()),
           &ldb, const_cast<double*>(A.data()), &lda, &beta,
           const_cast<double*>(C.data()), &ldc);
  }
}
//------------------------------------------------------------------
xt::xtensor<double, 2> basix::math::dot(const xt::xtensor<double, 2>& A,
                                        const xt::xtensor<double, 2>& B)
{
  xt::xtensor<double, 2> C = xt::zeros<double>({A.shape(0), B.shape(1)});
  dot(A, B, C);
  return C;
}
//------------------------------------------------------------------
std::vector<double> basix::math::eye(std::size_t n)
{
  std::vector<double> I(n * n, 0);
  std::experimental::mdspan<double, std::experimental::dextents<std::size_t, 2>>
      Iview(I.data(), n, n);
  for (std::size_t i = 0; i < n; ++i)
    Iview(i, i) = 1.0;
  return I;
}
//------------------------------------------------------------------
