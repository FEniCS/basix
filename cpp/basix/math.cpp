// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <array>
#include <cmath>
#include <type_traits>
#include <xtensor/xfixed.hpp>
#include <xtensor/xtensor.hpp>

extern "C"
{
  void dsyevd_(char* jobz, char* uplo, int* n, double* a, int* lda, double* w,
               double* work, int* lwork, int* iwork, int* liwork, int* info);
}

namespace basix::math
{

/// Compute the eigenvalues and eigenvectors of a square Hermitian matrix A
/// @param[in] A Input matrix
/// @param[out] C Filled to be C = A * B
std::pair<xt::xtensor<double, 1>,
          xt::xtensor<double, 2, xt::layout_type::column_major>>
eigh(const xt::xtensor<double, 2>& A)
{
  // Copy to column major matrix
  xt::xtensor<double, 2, xt::layout_type::column_major> M(A.shape());
  M.assign(A);
  int N = A.shape(0);
  xt::xtensor<double, 1> w = xt::zeros<double>({N});

  char jobz = 'V'; // Vectors
  char uplo = 'L'; // Lower
  int ldA = A.shape(1);
  int lwork = -1;
  int liwork = -1;
  int info;

  std::vector<double> work(1);
  std::vector<int> iwork(1);

  // Query and allocate the optimal workspace
  dsyevd_(&jobz, &uplo, &N, M.data(), &ldA, w.data(), work.data(), &lwork,
          iwork.data(), &liwork, &info);

  if (info != 0)
    throw std::runtime_error("Could not find workspace size for syevd.");

  // Solve eigenproblem
  work.resize(static_cast<std::size_t>(work[0]));
  iwork.resize(static_cast<std::size_t>(iwork[0]));

  dsyevd_(&jobz, &uplo, &N, M.data(), &ldA, w.data(), work.data(), &lwork,
          iwork.data(), &liwork, &info);

  if (info != 0)
    throw std::runtime_error("Eigenvalue computation did not converge.");

  return {std::move(w), std::move(M)};
}

} // namespace basix::math