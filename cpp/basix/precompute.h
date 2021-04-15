// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xspan.hpp>

namespace basix
{

/// ## Matrix and permutation precomputation
/// These functions generate precomputed version of matrices to allow
/// application without temporary memory assignment later
namespace precompute
{
/// Prepare a permutation
///
/// @param[in] perm A permutation
/// @return The precomputed representation of the permutation
std::vector<std::size_t>
prepare_permutation(const std::vector<std::size_t> perm);

/// Apply a (precomputed) permutation
///
/// @param[in] perm A permutation in precomputed form (as returned by
/// `prepare_permutation()`)
/// @param[in,out] data The data to apply the permutation to
/// @param[in] offset The position in the data to start applying the permutation
/// @param[in] block_size The block size of the data
template <typename E>
void apply_permutation(const std::vector<std::size_t> perm, xtl::span<E>& data,
                       const std::size_t offset = 0,
                       const std::size_t block_size = 1);

/// Prepare a matrix
///
/// @param[in] matrix A matrix
/// @return The precomputed representation of the matrix
template <typename T>
std::tuple<std::vector<std::size_t>, std::vector<T>, xt::xtensor<T, 2>>
prepare_matrix(const xt::xtensor<T, 2> matrix);

/// Apply a (precomputed) matrix
///
/// @param[in] matrix A matrix in precomputed form (as returned by
/// `prepare_matrix()`)
/// @param[in,out] data The data to apply the permutation to
/// @param[in] offset The position in the data to start applying the permutation
/// @param[in] block_size The block size of the data
template <typename T, typename E>
void apply_matrix(const std::tuple<std::vector<std::size_t>, std::vector<T>,
                                   xt::xtensor<T, 2>>
                      matrix,
                  xtl::span<E>& data, const std::size_t offset = 0,
                  const std::size_t block_size = 1);
} // namespace precompute

//-----------------------------------------------------------------------------
template <typename E>
void precompute::apply_permutation(const std::vector<std::size_t> perm,
                                   xtl::span<E>& data, const std::size_t offset,
                                   const std::size_t block_size)
{
  for (std::size_t b = 0; b < block_size; ++b)
    for (std::size_t i = 0; i < perm.size(); ++i)
      std::swap(data[block_size * (offset + i) + b],
                data[block_size * (offset + perm[i]) + b]);
}
//-----------------------------------------------------------------------------
template <typename T>
std::tuple<std::vector<std::size_t>, std::vector<T>, xt::xtensor<T, 2>>
precompute::prepare_matrix(const xt::xtensor<T, 2> matrix)
{
  const std::size_t dim = matrix.shape(0);

  std::vector<std::size_t> perm(dim);
  xt::xtensor<T, 2> permuted_matrix({dim, dim});
  std::vector<T> diag(dim);

  // Permute the matrix so that all the top left blocks are invertible
  for (std::size_t i = 0; i < dim; ++i)
  {
    double max_det = 0;
    std::size_t col = -1;
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
        prepared_matrix(i, j) -= xt::linalg::dot(
            v, xt::view(permuted_matrix, xt::range(0, i), j))(0);
    }
  }

  return std::make_tuple(prepare_permutation(perm), diag, prepared_matrix);
}
//-----------------------------------------------------------------------------
template <typename T, typename E>
void precompute::apply_matrix(
    const std::tuple<std::vector<std::size_t>, std::vector<T>,
                     xt::xtensor<T, 2>>
        matrix,
    xtl::span<E>& data, const std::size_t offset, const std::size_t block_size)
{
  const std::size_t dim = std::get<0>(matrix).size();

  apply_permutation(std::get<0>(matrix), data, offset, block_size);
  for (std::size_t b = 0; b < block_size; ++b)
  {
    for (std::size_t i = 0; i < dim; ++i)
    {
      data[block_size * (offset + i) + b] *= std::get<1>(matrix)[i];
      for (std::size_t j = 0; j < dim; ++j)
      {
        data[block_size * (offset + i) + b]
            += std::get<2>(matrix)(i, j) * data[block_size * (offset + j) + b];
      }
    }
  }
}
//-----------------------------------------------------------------------------

} // namespace basix
