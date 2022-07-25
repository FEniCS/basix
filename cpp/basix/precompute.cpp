// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "precompute.h"
#include "math.h"
#include <numeric>

using namespace basix;

namespace stdex = std::experimental;
using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
using mdarray2_t = stdex::mdarray<double, stdex::dextents<std::size_t, 2>>;

//-----------------------------------------------------------------------------
std::vector<std::size_t>
precompute::prepare_permutation(const std::span<const std::size_t>& perm)
{
  std::vector<std::size_t> f_perm(perm.size());
  for (std::size_t row = 0; row < perm.size(); ++row)
  {
    std::size_t row2 = perm[row];
    while (row2 < row)
      row2 = f_perm[row2];
    f_perm[row] = row2;
  }
  return f_perm;
}
//-----------------------------------------------------------------------------
std::tuple<std::vector<std::size_t>, std::vector<double>,
           std::pair<std::vector<double>, std::array<std::size_t, 2>>>
precompute::prepare_matrix(
    const std::experimental::mdspan<
        const double, std::experimental::dextents<std::size_t, 2>>& matrix)
{
  using T = double;
  const std::size_t dim = matrix.extent(0);
  assert(dim == matrix.extent(1));
  mdarray2_t permuted_matrix(dim, dim);
  std::vector<T> diag(dim);

  // Permute the matrix so that all the top left blocks are invertible
  std::vector<std::size_t> perm = math::lu_permutation(matrix);

  for (std::size_t i = 0; i < dim; ++i)
    for (std::size_t j = 0; j < dim; ++j)
      permuted_matrix(j, i) = matrix(j, perm[i]);

  // Create the precomputed representation of the matrix
  std::vector<double> prepared_matrix_b(dim * dim);
  mdspan2_t prepared_matrix(prepared_matrix_b.data(), dim, dim);
  std::vector<double> Ab(dim * dim), Bb(dim * dim), tb(dim);
  for (std::size_t i = 0; i < dim; ++i)
  {
    diag[i] = permuted_matrix(i, i);
    prepared_matrix(i, i) = 0;
    if (i < dim - 1)
    {
      for (std::size_t k = i + 1; k < dim; ++k)
        prepared_matrix(i, k) = permuted_matrix(i, k);
    }

    if (i > 0)
    {
      mdspan2_t A(Ab.data(), i, i);
      for (std::size_t k0 = 0; k0 < A.extent(1); ++k0)
        for (std::size_t k1 = 0; k1 < A.extent(0); ++k1)
          A(k1, k0) = permuted_matrix(k0, k1);

      mdspan2_t B(Bb.data(), i, 1);
      for (std::size_t k1 = 0; k1 < B.extent(0); ++k1)
        B(k1, 0) = permuted_matrix(i, k1);

      const std::vector<double> v = math::solve(A, B);

      for (std::size_t k1 = 0; k1 < i; ++k1)
        prepared_matrix(i, k1) = v[k1];

      std::span t(tb.data(), i);
      for (std::size_t k = 0; k < i; ++k)
        t[k] = permuted_matrix(k, i);
      diag[i] -= std::transform_reduce(v.begin(), v.end(), t.begin(), 0.0);

      for (std::size_t j = i + 1; j < dim; ++j)
      {
        for (std::size_t k = 0; k < i; ++k)
          t[k] = permuted_matrix(k, j);
        prepared_matrix(i, j)
            -= std::transform_reduce(v.begin(), v.end(), t.begin(), 0.0);
      }
    }
  }

  return {prepare_permutation(perm),
          std::move(diag),
          {std::move(prepared_matrix_b),
           {prepared_matrix.extent(0), prepared_matrix.extent(1)}}};
}
//-----------------------------------------------------------------------------
