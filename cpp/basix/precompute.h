// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "mdspan.hpp"
#include <span>
#include <tuple>
#include <vector>

/// Matrix and permutation precomputation
namespace basix::precompute
{

/// Prepare a permutation
///
/// This computes a representation of the permutation that allows the
/// permutation to be applied without any temporary memory assignment.
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
/// @param[in,out] perm A permutation
void prepare_permutation(const std::span<std::size_t>& perm);

/// Apply a (precomputed) permutation
///
/// This uses the representation returned by `prepare_permutation()` to apply a
/// permutation without needing any temporary memory.
///
/// In pseudo code, this function does the following:
///
/// \code{.pseudo}
/// FOR index, entry IN perm:
///     SWAP(data[index], data[entry])
/// \endcode
///
/// If `block_size` is set, this will apply the permutation to every block.
/// The `offset` is set, this will start applying the permutation at the
/// `offset`th block.
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
/// (which is what we get if we apply the permutation directly).
///
/// @note This function is designed to be called at runtime, so its performance
/// is critical.
///
/// @param[in] perm A permutation in precomputed form (as returned by
/// `prepare_permutation()`)
/// @param[in,out] data The data to apply the permutation to
/// @param[in] offset The position in the data to start applying the permutation
/// @param[in] block_size The block size of the data
template <typename E>
void apply_permutation(const std::span<const std::size_t>& perm,
                       const std::span<E>& data, std::size_t offset = 0,
                       std::size_t block_size = 1)
{
  for (std::size_t b = 0; b < block_size; ++b)
  {
    for (std::size_t i = 0; i < perm.size(); ++i)
    {
      std::swap(data[block_size * (offset + i) + b],
                data[block_size * (offset + perm[i]) + b]);
    }
  }
}

/// Apply a (precomputed) permutation to some transposed data
///
/// @note This function is designed to be called at runtime, so its performance
/// is critical.
///
/// see `apply_permutation()`.
template <typename E>
void apply_permutation_to_transpose(const std::span<const std::size_t>& perm,
                                    const std::span<E>& data,
                                    std::size_t offset = 0,
                                    std::size_t block_size = 1)
{
  const std::size_t dim = perm.size();
  const std::size_t data_size
      = (data.size() + (dim < block_size ? block_size - dim : 0)) / block_size;
  for (std::size_t b = 0; b < block_size; ++b)
  {
    for (std::size_t i = 0; i < dim; ++i)
    {
      std::swap(data[data_size * b + offset + i],
                data[data_size * b + offset + perm[i]]);
    }
  }
}

/// Prepare a square matrix
///
/// This computes a representation of the matrix that allows the matrix to be
/// applied without any temporary memory assignment.
///
/// This function will first permute the matrix's columns so that the top left
/// @f$n\times n@f$ blocks are invertible (for all @f$n@f$). Let @f$A@f$ be the
/// input matrix after the permutation is applied. The output vector @f$D@f$ and
/// matrix @f$M@f$ are then given by:
///  @f{align*}{
///  M_{i,j} &= \begin{cases}
///         A_{i,:i}A_{:i,:i}^{-1}e_j & j < i\\
///         A_{i, i} - A_{i,:i}A_{:i,:i}^{-1}A_{:i,i} & j = i\\
///         A_{i, i} - A_{i,:i}A_{:i,:i}^{-1}A_{:i,j} & j > i = 0
///        \end{cases},
/// @f}
/// where @f$e_j@f$ is the @f$j@f$th coordinate vector, we index all the
/// matrices and vector starting at 0, and we use numpy-slicing-stying notation
/// in the subscripts: for example, @f$A_{:i,j}@f$ represents the first @f$i@f$
/// entries in the @f$j@f$th column of @f$A@f$
///
/// This function returns the permutation (precomputed as in
/// `prepare_permutation()`), the vector @f$D@f$, and the matrix @f$M@f$ as a
/// tuple.
///
/// For an example of how the permutation in this form is applied, see
/// `apply_matrix()`.
///
/// @param[in,out] A The matrix's data
/// @param[in] dim The number of rows/columns of the matrix
/// @return The three parts of a precomputed representation of the matrix.
/// These are (as described above):
/// - A permutation (precomputed as in `prepare_permutation()`);
/// - the vector @f$D@f$;
std::vector<std::size_t>
prepare_matrix(std::pair<std::vector<double>, std::array<std::size_t, 2>>& A);

/// @brief Apply a (precomputed) matrix.
///
/// This uses the representation returned by `prepare_matrix()` to apply a
/// matrix without needing any temporary memory.
///
/// In pseudo code, this function does the following:
///
/// \code{.pseudo}
/// perm, diag, mat = matrix
/// apply_permutation(perm, data)
/// FOR index IN RANGE(dim):
///     data[index] *= mat[index, index]
///     FOR j IN RANGE(dim):
///         IF j != index:
///             data[index] *= mat[index, j] * data[j]
/// \endcode
///
/// If `block_size` is set, this will apply the permutation to every block.
/// The `offset` is set, this will start applying the permutation at the
/// `offset`th block.
///
/// @note This function is designed to be called at runtime, so its
/// performance is critical.
///
/// @param[in] v_size_t A permutaion, as computed by
/// precompute::prepare_matrix
/// @param[in] M The vector created by precompute::prepare_matrix
/// @param[in,out] data The data to apply the permutation to
/// @param[in] offset The position in the data to start applying the
/// permutation
/// @param[in] block_size The block size of the data
template <typename T, typename E>
void apply_matrix(const std::span<const std::size_t>& v_size_t,
                  const std::experimental::mdspan<
                      const T, std::experimental::dextents<std::size_t, 2>>& M,
                  const std::span<E>& data, std::size_t offset = 0,
                  std::size_t block_size = 1)
{
  const std::size_t dim = v_size_t.size();
  apply_permutation(v_size_t, data, offset, block_size);
  for (std::size_t b = 0; b < block_size; ++b)
  {
    for (std::size_t i = 0; i < dim; ++i)
    {
      for (std::size_t j = i + 1; j < dim; ++j)
      {
        data[block_size * (offset + i) + b]
            += M(i, j) * data[block_size * (offset + j) + b];
      }
    }
    for (std::size_t i = 1; i <= dim; ++i)
    {
      data[block_size * (offset + dim - i) + b] *= M(dim - i, dim - i);
      for (std::size_t j = 0; j < dim - i; ++j)
      {
        data[block_size * (offset + dim - i) + b]
            += M(dim - i, j) * data[block_size * (offset + j) + b];
      }
    }
  }
}

/// @brief Apply a (precomputed) matrix to some transposed data.
///
/// @note This function is designed to be called at runtime, so its
/// performance is critical.
///
/// See `apply_matrix()`.
template <typename T, typename E>
void apply_matrix_to_transpose(
    const std::span<const std::size_t>& v_size_t,
    const std::experimental::mdspan<
        const T, std::experimental::dextents<std::size_t, 2>>& M,
    const std::span<E>& data, std::size_t offset = 0,
    std::size_t block_size = 1)
{
  const std::size_t dim = v_size_t.size();
  const std::size_t data_size
      = (data.size() + (dim < block_size ? block_size - dim : 0)) / block_size;
  apply_permutation_to_transpose(v_size_t, data, offset, block_size);
  for (std::size_t b = 0; b < block_size; ++b)
  {
    for (std::size_t i = 0; i < dim; ++i)
    {
      for (std::size_t j = i + 1; j < dim; ++j)
      {
        data[data_size * b + offset + i]
            += M(i, j) * data[data_size * b + offset + j];
      }
    }
    for (std::size_t i = 1; i <= dim; ++i)
    {
      data[data_size * b + offset + dim - i] *= M(dim - i, dim - i);
      for (std::size_t j = 0; j < dim - i; ++j)
      {
        data[data_size * b + offset + dim - i]
            += M(dim - i, j) * data[data_size * b + offset + j];
      }
    }
  }
}

} // namespace basix::precompute
