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
/// This computes a representation of the permutation that allows the
/// permutations to be applied without any temporary memory assignment.
///
/// In pseudo code, this function does the following:
///
/// \code{.pseudo}
/// FOR index, entry IN perm:
///     new_index = entry
///     WHILE new_index < index:
///         new_index = perm[new_index]
///     OUT[index] = new_index
/// \endcode
///
/// Example
/// -------
/// As an example, consider the permutation `P = [1, 4, 0, 5, 2, 3]`.
///
/// First, we look at the 0th entry. `P[0]` is 1. This is greater than 0, so the
/// 0th entry of the output is 1.
///
/// Next, we look at the 1st entry. `P[1]` is 4. This is greater than 1, so the
/// 1st entry of the output is 4.
///
/// Next, we look at the 2nd entry. `P[2]` is 0. This is less than 2, so we look
/// at `P[0]. `P[0]` is 1. This is less than 2, so we look at `P[1]`. `P[1]`
/// is 4. This is greater than 2, so the 2nd entry of the output is 4.
///
/// Next, we look at the 3rd entry. `P[3]` is 5. This is greater than 3, so the
/// 3rd entry of the output is 5.
///
/// Next, we look at the 4th entry. `P[4]` is 2. This is less than 4, so we look
/// at `P[2]`. `P[2]` is 0. This is less than 4, so we look at `P[0]`. `P[0]`
/// is 1. This is less than 4, so we look at `P[1]`. `P[1]` is 4. This is
/// greater than (or equal to) 4, so the 4th entry of the output is 4.
///
/// Next, we look at the 5th entry. `P[5]` is 3. This is less than 5, so we look
/// at `P[3]`. `P[3]` is 5. This is greater than (or equal to) 5, so the 5th
/// entry of the output is 5.
///
/// Hence, the output of this function in this case is `[1, 4, 4, 5, 4, 5]`.
///
/// For an example of how the permutation in this form is applied, see
/// `apply_permutation()`.
///
/// @param[in] perm A permutation
/// @return The precomputed representation of the permutation
std::vector<std::size_t>
prepare_permutation(const std::vector<std::size_t> perm);

/// Apply a (precomputed) permutation
///
/// This uses the representation returned by `prepare_permutation()` to apply a
/// permutation without needing any temporary memory.
///
/// In pseudo code, this function does the following:
///
/// \code{.pseudo}
/// FOR index, entry IN perm:
///     SWAP(INPUT[index], INPUT[entry]
/// \endcode
///
/// Example
/// -------
/// As an example, consider the permutation `P = [1, 4, 0, 5, 2, 3]`.
/// In the documentation of `prepare_permutation()`, we saw that the precomputed
/// representation of this permutation is `P2 = [1, 4, 4, 5, 4, 5]`. In this
/// example, we look at how this representation can be used to apply this
/// permutation to the array `A = [a, b, c, d, e, f]`.
///
/// `P2[0]` is 1, so we swap `A[0]` and `A[1]`. After this, `A = [b, a, c, d, e,
/// f]`.
///
/// `P2[1]` is 4, so we swap `A[1]` and `A[4]`. After this, `A = [b, e, c, d, a,
/// f]`.
///
/// `P2[2]` is 4, so we swap `A[2]` and `A[4]`. After this, `A = [b, e, a, d, c,
/// f]`.
///
/// `P2[3]` is 5, so we swap `A[3]` and `A[5]`. After this, `A = [b, e, a, f, c,
/// d]`.
///
/// `P2[4]` is 4, so we swap `A[4]` and `A[4]`. This changes nothing.
///
/// `P2[5]` is 5, so we swap `A[5]` and `A[5]`. This changes nothing.
///
/// Therefore the result of applying this permutation is `[b, e, a, f, c, d]`
/// (which is what we get if we apply the permutation directly.
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
