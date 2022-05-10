// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "precompute.h"
#include "math.h"
#include <xtensor/xview.hpp>

using namespace basix;

//-----------------------------------------------------------------------------
std::vector<std::size_t>
precompute::prepare_permutation(const std::vector<std::size_t>& perm)
{
  std::vector<std::size_t> f_perm(perm.size());
  for (std::size_t row = 0; row < perm.size(); ++row)
  {
    std::size_t row2 = perm[row];
    while (row2 < row)
      row2 = perm[row2];
    f_perm[row] = row2;
  }
  return f_perm;
}
//-----------------------------------------------------------------------------
std::tuple<std::vector<std::size_t>, std::vector<double>,
           xt::xtensor<double, 2>>
precompute::prepare_matrix(const xt::xtensor<double, 2>& matrix)
{
  using T = double;
  const std::size_t dim = matrix.shape(0);
  std::vector<std::size_t> perm(dim);
  xt::xtensor<T, 2> permuted_matrix({dim, dim});
  std::vector<T> diag(dim);

  // Permute the matrix so that all the top left blocks are invertible
  for (std::size_t i = 0; i < dim; ++i)
  {
    double max_eval = 0;
    std::size_t col = 0;
    for (std::size_t j = 0; j < dim; ++j)
    {
      const bool used = std::find(perm.begin(), std::next(perm.begin(), i), j)
                        != std::next(perm.begin(), i);
      if (!used)
      {
        xt::col(permuted_matrix, i) = xt::col(matrix, j);
        xt::xtensor<double, 2> mat = xt::view(
            permuted_matrix, xt::range(0, i + 1), xt::range(0, i + 1));
        xt::xtensor<double, 2> mat_t = xt::transpose(mat);
        xt::xtensor<double, 2> mat2 = math::dot(mat, mat_t);
        auto [evals, evecs] = math::eigh(mat2);
        if (double lambda = std::abs(evals.front()); lambda > max_eval)
        {
          max_eval = lambda;
          col = j;
        }
      }
    }
    if (xt::allclose(max_eval, 0))
      throw std::runtime_error("Singular matrix");

    xt::col(permuted_matrix, i) = xt::col(matrix, col);
    perm[i] = col;
  }

  // Create the precomputed representation of the matrix
  xt::xtensor<T, 2> prepared_matrix({dim, dim});
  xt::xtensor<T, 2> A, B;
  for (std::size_t i = 0; i < dim; ++i)
  {
    diag[i] = permuted_matrix(i, i);
    prepared_matrix(i, i) = 0;
    if (i < dim - 1)
    {
      xt::view(prepared_matrix, i, xt::range(i + 1, dim))
          = xt::view(permuted_matrix, i, xt::range(i + 1, dim));
    }

    if (i > 0)
    {

      A.assign(xt::transpose(
          xt::view(permuted_matrix, xt::range(0, i), xt::range(0, i))));
      B.assign(xt::transpose(
          xt::view(permuted_matrix, xt::range(i, i + 1), xt::range(0, i))));
      xt::xtensor<T, 2> v = math::solve(A, B);
      xt::view(prepared_matrix, i, xt::range(0, i)).assign(xt::col(v, 0));

      xt::xtensor<T, 1> t = xt::view(permuted_matrix, xt::range(0, i), i);
      diag[i] -= std::transform_reduce(v.begin(), v.end(), t.begin(), 0.);

      for (std::size_t j = i + 1; j < dim; ++j)
      {
        t = xt::view(permuted_matrix, xt::range(0, i), j);
        prepared_matrix(i, j)
            -= std::transform_reduce(v.begin(), v.end(), t.begin(), 0.);
      }
    }
  }

  return {prepare_permutation(perm), std::move(diag),
          std::move(prepared_matrix)};
}
//-----------------------------------------------------------------------------
