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
std::vector<std::size_t> basix::math::transpose_lu(
    std::pair<std::vector<double>, std::array<std::size_t, 2>>& A)
{
  const std::size_t dim = A.second[0];
  assert(dim == A.second[1]);

  int N = dim;
  int info;
  std::vector<int> lu_perm(dim);

  // Comput LU decomposition of M
  dgetrf_(&N, &N, A.first.data(), &N, lu_perm.data(), &info);

  if (info != 0)
    throw std::runtime_error("LU decomposition failed: "
                             + std::to_string(info));

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
