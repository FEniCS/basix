// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "precompute.h"
#include "math.h"
#include <numeric>
#include <xtensor/xadapt.hpp>

using namespace basix;

namespace stdex = std::experimental;
using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
using mdarray2_t = stdex::mdarray<double, stdex::dextents<std::size_t, 2>>;

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
           std::pair<std::vector<double>, std::array<std::size_t, 2>>>
precompute::prepare_matrix(const xt::xtensor<double, 2>& matrix)
{
  using T = double;
  const std::size_t dim = matrix.shape(0);
  std::vector<std::size_t> perm(dim);
  mdarray2_t permuted_matrix(dim, dim);
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

        mdarray2_t mat(i + 1, i + 1);
        for (std::size_t k0 = 0; k0 < mat.extent(0); ++k0)
          for (std::size_t k1 = 0; k1 < mat.extent(1); ++k1)
            mat(k0, k1) = permuted_matrix(k0, k1);

        mdarray2_t mat_t(mat.extent(1), mat.extent(0));
        for (std::size_t k0 = 0; k0 < mat.extent(0); ++k0)
          for (std::size_t k1 = 0; k1 < mat.extent(1); ++k1)
            mat_t(k1, k0) = mat(k0, k1);

        auto [mat2_data, shape] = math::dot_new(mat, mat_t);
        cmdspan2_t mat2(mat2_data.data(), shape);

        auto [evals, _] = math::eigh(mat2, mat2.extent(0));
        if (double lambda = std::abs(evals.front()); lambda > max_eval)
        {
          max_eval = lambda;
          col = j;
        }
      }
    }

    if (std::abs(max_eval) < 1.0e-9)
      throw std::runtime_error("Singular matrix");

    for (std::size_t k = 0; k < matrix.shape(0); ++k)
      permuted_matrix(k, i) = matrix(k, col);

    perm[i] = col;
  }

  // Create the precomputed representation of the matrix
  std::vector<double> prepared_matrix_b(dim * dim);
  mdspan2_t prepared_matrix(prepared_matrix_b.data(), dim, dim);
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
      std::vector<double> Ab(i * i);
      mdspan2_t A(Ab.data(), i, i);
      for (std::size_t k0 = 0; k0 < A.extent(1); ++k0)
        for (std::size_t k1 = 0; k1 < A.extent(0); ++k1)
          A(k1, k0) = permuted_matrix(k0, k1);

      std::vector<double> Bb(i);
      mdspan2_t B(Bb.data(), i, 1);
      for (std::size_t k1 = 0; k1 < B.extent(0); ++k1)
        B(k1, 0) = permuted_matrix(i, k1);

      const std::vector<double> v = math::solve(A, B);

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

  return {prepare_permutation(perm),
          std::move(diag),
          {std::move(prepared_matrix_b),
           {prepared_matrix.extent(0), prepared_matrix.extent(1)}}};
}
//-----------------------------------------------------------------------------
