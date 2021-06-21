// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "precompute.h"
#include <xtensor-blas/xlinalg.hpp>

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
    double max_det = 0;
    std::size_t col;
    for (std::size_t j = 0; j < dim; ++j)
    {
      bool used = false;
      for (std::size_t k = 0; k < i; ++k)
        if (perm[k] == j)
          used = true;

      if (!used)
      {
        xt::view(permuted_matrix, xt::all(), i)
            = xt::view(matrix, xt::all(), j);
        double det = std::abs(xt::linalg::det(xt::view(
            permuted_matrix, xt::range(0, i + 1), xt::range(0, i + 1))));
        if (det > max_det)
        {
          max_det = det;
          col = j;
        }
      }
    }
    xt::view(permuted_matrix, xt::all(), i) = xt::view(matrix, xt::all(), col);
    perm[i] = col;
  }

  // Create the precomputed representation of the matrix
  xt::xtensor<T, 2> prepared_matrix({dim, dim});

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
      xt::xtensor<T, 1> v = xt::linalg::solve(
          xt::transpose(
              xt::view(permuted_matrix, xt::range(0, i), xt::range(0, i))),
          xt::view(permuted_matrix, i, xt::range(0, i)));

      xt::view(prepared_matrix, i, xt::range(0, i)) = v;

      diag[i] -= xt::linalg::dot(
          v, xt::view(permuted_matrix, xt::range(0, i), i))(0);
      for (std::size_t j = i + 1; j < dim; ++j)
      {
        prepared_matrix(i, j) -= xt::linalg::dot(
            v, xt::view(permuted_matrix, xt::range(0, i), j))(0);
      }
    }
  }

  return {prepare_permutation(perm), std::move(diag),
          std::move(prepared_matrix)};
}
//-----------------------------------------------------------------------------
