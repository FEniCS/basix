// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cmath>
#include <type_traits>
#include <xtensor/xfixed.hpp>
#include <xtensor/xtensor.hpp>

extern "C"
{
  void dsyevd(char* jobz, char* uplo, int* n, double* a, int* lda, double* w,
              double* work, int* lwork, int* iwork, int* liwork, int* info);
}

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
/// @param[out] C Filled to be C = A * B
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
/// @param[out] C Filled to be C = A * B
template <typename U>
auto eigh(const U& A)
{
  // Copy to column major matrix
  xt::xtensor<double, 2, xt::layout_type::column_major> M(A.shape());
  M.assign(A);
  std::array<int, 1> N = {A.shape(0)};
  xt::xtensor<double, 1> w(N);

  char jobz = 'V'; // Vectors
  char uplo = 'L'; // Lower
  int ldA = A.shape(1);
  int lwork = -1;
  int liwork = -1;
  int info;

  std::vector<double> work(1);
  std::vector<int> iwork(1);

  // Query and allocate the optimal workspace
  dsyevd(&jobz, &uplo, N.data(), M.data(), &ldA, w.data(), work.data(), &lwork,
         iwork.data(), &liwork, &info);

  if (info != 0)
  {
    throw std::runtime_error("Could not find workspace size for syevd.");
  }

  // Solve eigenproblem
  work.resize(static_cast<std::size_t>(work[0]));
  iwork.resize(static_cast<std::size_t>(iwork[0]));

  dsyevd(&jobz, &uplo, N.data(), A.data(), &ldA, w.data(), work.data(), &lwork,
         iwork.data(), &liwork, &info);

  if (info != 0)
  {
    throw std::runtime_error("Eigenvalue computation did not converge.");
  }

  return std::make_tuple(std::move(w), std::move(M));
}

} // namespace basix::math