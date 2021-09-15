// Copyright (C) 2021 Igor Baratta
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "math.h"
#include <array>
#include <cmath>

extern "C"
{
  void dsyevd(char* jobz, char* uplo, int* n, double* a, int* lda, double* w,
              double* work, int* lwork, int* iwork, int* liwork, int* info);
}

std::pair<xt::xtensor<double, 1>,
          xt::xtensor<double, 2, xt::layout_type::column_major>>
basix::math::eigh(const xt::xtensor<double, 2>& A)
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
  dsyevd(&jobz, &uplo, &N, M.data(), &ldA, w.data(), work.data(), &lwork,
         iwork.data(), &liwork, &info);

  if (info != 0)
  {
    throw std::runtime_error("Could not find workspace size for syevd.");
  }

  // Solve eigenproblem
  work.resize(static_cast<std::size_t>(work[0]));
  iwork.resize(static_cast<std::size_t>(iwork[0]));

  dsyevd(&jobz, &uplo, &N, M.data(), &ldA, w.data(), work.data(), &lwork,
         iwork.data(), &liwork, &info);

  if (info != 0)
  {
    throw std::runtime_error("Eigenvalue computation did not converge.");
  }

  return {std::move(w), std::move(M)};
}