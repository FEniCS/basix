// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "precompute.h"
#include "math.h"
#include <numeric>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

//-----------------------------------------------------------------------------
std::vector<std::size_t>
precompute::prepare_permutation(const xtl::span<const std::size_t>& perm)
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
        for (std::size_t k = 0; k < matrix.shape(0); ++k)
          permuted_matrix(k, i) = matrix(k, j);

        xt::xtensor<double, 2> mat({i + 1, i + 1});
        for (std::size_t k0 = 0; k0 < mat.shape(0); ++k0)
          for (std::size_t k1 = 0; k1 < mat.shape(1); ++k1)
            mat(k0, k1) = permuted_matrix(k0, k1);

        xt::xtensor<double, 2> mat_t({mat.shape(1), mat.shape(0)});
        for (std::size_t k0 = 0; k0 < mat.shape(0); ++k0)
          for (std::size_t k1 = 0; k1 < mat.shape(1); ++k1)
            mat_t(k1, k0) = mat(k0, k1);

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

    for (std::size_t k = 0; k < matrix.shape(0); ++k)
      permuted_matrix(k, i) = matrix(k, col);

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
      for (std::size_t k = i + 1; k < dim; ++k)
        prepared_matrix(i, k) = permuted_matrix(i, k);
    }

    if (i > 0)
    {
      xt::xtensor<T, 2> A({i, i});
      for (std::size_t k0 = 0; k0 < A.shape(0); ++k0)
        for (std::size_t k1 = 0; k1 < A.shape(1); ++k1)
          A(k0, k1) = permuted_matrix(k0, k1);

      xt::xtensor<T, 2> B({1, i});
      for (std::size_t k1 = 0; k1 < B.shape(1); ++k1)
        B(0, k1) = permuted_matrix(i, k1);

      std::vector<double> v(i);
      auto _v = xt::adapt(v, std::vector<std::size_t>{1, i});
      _v = math::solve(xt::transpose(A), xt::transpose(B));

      for (std::size_t k1 = 0; k1 < i; ++k1)
        prepared_matrix(i, k1) = v[k1];

      std::vector<double> t(i);
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

  return {prepare_permutation(perm), std::move(diag),
          std::move(prepared_matrix)};
}
//-----------------------------------------------------------------------------
