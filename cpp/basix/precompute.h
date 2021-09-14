// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include <vector>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xspan.hpp>

/// ## Matrix and permutation precomputation
/// These functions generate precomputed version of matrices to allow
/// application without temporary memory assignment later
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
/// @param[in] perm A permutation
/// @return The precomputed representation of the permutation
std::vector<std::size_t>
prepare_permutation(const std::vector<std::size_t>& perm);

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
/// @param[in] perm A permutation in precomputed form (as returned by
/// `prepare_permutation()`)
/// @param[in,out] data The data to apply the permutation to
/// @param[in] offset The position in the data to start applying the permutation
/// @param[in] block_size The block size of the data
template <typename E>
void apply_permutation(const std::vector<std::size_t>& perm,
                       const xtl::span<E>& data, std::size_t offset = 0,
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
/// see `apply_permutation()`.
template <typename E>
void apply_permutation_to_transpose(const std::vector<std::size_t>& perm,
                                    const xtl::span<E>& data,
                                    std::size_t offset = 0,
                                    std::size_t block_size = 1)
{
  const std::size_t dim = perm.size();
  const std::size_t data_size = (data.size() + block_size - dim) / block_size;
  for (std::size_t b = 0; b < block_size; ++b)
  {
    for (std::size_t i = 0; i < dim; ++i)
    {
      std::swap(data[data_size * b + offset + i],
                data[data_size * b + offset + perm[i]]);
    }
  }
}

/// Prepare a matrix
///
/// This computes a representation of the matrix that allows the matrix to be
/// applied without any temporary memory assignment.
///
/// This function will first permute the matrix's columns so that the top left
/// @f$n\times n@f$ blocks are invertible (for all @f$n@f$). Let @f$A@f$ be the
/// input matrix after the permutation is applied. The output vector @f$D@f$ and
/// matrix @f$M@f$ are then given by:
///  @f{align*}{
///  D_i &= \begin{cases}
///         A_{i, i} & i = 0\\
///         A_{i, i} - A_{i,:i}A_{:i,:i}^{-1}A_{:i,i} & i \not= 0
///        \end{cases},\\
///  M_{i,j} &= \begin{cases}
///         A_{i,:i}A_{:i,:i}^{-1}e_j & j < i\\
///         0 & j = i\\
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
/// Example
/// -------
/// As an example, consider the matrix @f$A = @f$ `[[-1, 0, 1], [1, 1, 0], [2,
/// 0, 2]]`. For this matrix, no permutation is needed, so the first item in the
/// output will represent the identity permutation. We now compute the output
/// vector @f$D@f$ and matrix @f$M@f$.
///
/// First, we set @f$D_0 = A_{0,0}=-1@f$,
/// set the diagonal of @f$M@f$ to be 0
/// and set @f$M_{0, 1:} = A_{0, 1:}=\begin{bmatrix}0&1\end{bmatrix}@f$.
/// The output so far is
///  @f{align*}{ D &= \begin{bmatrix}-1\\?\\?\end{bmatrix},\\
///  \quad M &= \begin{bmatrix}
///             0&0&1\\
///             ?&0&?\\
///             ?&?&0
///          \end{bmatrix}. @f}
///
/// Next, we set:
///  @f{align*}{ D_1 &= A_{1,1} - A_{1, :1}A_{:1,:1}^{-1}A_{:1, 1}\\
///          &= 1 -
///          \begin{bmatrix}-1\end{bmatrix}\cdot\begin{bmatrix}0\end{bmatrix}\\
///          &= 1,\\
/// M_{2,0} &= A_{1, :1}A_{:1,:1}^{-1}e_0\\
///         &= \begin{bmatrix}1\end{bmatrix}\begin{bmatrix}-1\end{bmatrix}^{-1}
///            \begin{bmatrix}1\end{bmatrix}\\
///         &= \begin{bmatrix}-1\end{bmatrix}
///  M_{2,3} &= A_{1,2}-A_{1, :1}A_{:1,:1}^{-1}A_{:1, 1}\\
///          &=
///          0-\begin{bmatrix}1\end{bmatrix}\begin{bmatrix}-1\end{bmatrix}^{-1}
///            \begin{bmatrix}1\end{bmatrix},\\
///          &= 1.
/// @f}
/// The output so far is
///  @f{align*}{ D &= \begin{bmatrix}-1\\1\\?\end{bmatrix},\\
///  \quad M &= \begin{bmatrix}
///             0&0&1\\
///             -1&0&1\\
///             ?&?&0
///          \end{bmatrix}. @f}
///
/// Next, we set:
///  @f{align*}{ D_2 &= A_{2,2} - A_{2, :2}A_{:2,:2}^{-1}A_{:2, 2}\\
///          &= 2 -
///          \begin{bmatrix}2&0\end{bmatrix}
///          \begin{bmatrix}-1&0\\1&1\end{bmatrix}^{-1}
///          \begin{bmatrix}1\\0\end{bmatrix}\\
///          &= 4,\\
/// M_{2,0} &= A_{2, :2}A_{:2,:2}^{-1}e_0\\ &= -2.\\
/// M_{2,1} &= A_{2, :2}A_{:2,:2}^{-1}e_1\\ &= 0.\\
/// @f}
/// The output is
///  @f{align*}{ D &= \begin{bmatrix}-1\\1\\4\end{bmatrix},\\
///  \quad M &= \begin{bmatrix}
///             0&0&1\\
///             -1&0&1\\
///             -2&0&0
///          \end{bmatrix}. @f}
///
/// For an example of how the permutation in this form is applied, see
/// `apply_matrix()`.
///
/// @param[in] matrix A matrix
/// @return The precomputed representation of the matrix
std::tuple<std::vector<std::size_t>, std::vector<double>,
           xt::xtensor<double, 2>>
prepare_matrix(const xt::xtensor<double, 2>& matrix);

/// Apply a (precomputed) matrix
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
///     data[index] *= diag[index]
///     FOR j IN RANGE(dim):
///         data[index] *= mat[index, j] * data[j]
/// \endcode
///
/// If `block_size` is set, this will apply the permutation to every block.
/// The `offset` is set, this will start applying the permutation at the
/// `offset`th block.
///
/// Example
/// -------
/// As an example, consider the matrix @f$A = @f$ `[[-1, 0, 1], [1, 1, 0], [2,
/// 0, 2]]`. In the documentation of `prepare_matrix()`, we saw that the
/// precomputed representation of this matrix is the identity permutation,
///  @f{align*}{ D &= \begin{bmatrix}-1\\1\\4\end{bmatrix},\\
///  \quad M &= \begin{bmatrix}
///             0&0&1\\
///             -1&0&1\\
///             -2&0&0
///          \end{bmatrix}. @f}
/// In this example, we look at how this representation can be used to
/// apply this matrix to the vector @f$v = @f$ `[3, -1, 2]`.
///
/// No permutation is necessary, so first, we multiply @f$v_0@f$ by
/// @f$D_0=-1@f$. After this, @f$v@f$ is `[-3, -1, 2]`.
///
/// Next, we add @f$M_{0,i}v_i@f$ to @f$v_0@f$ for all @f$i@f$: in this case, we
/// add @f$0\times-3 + 0\times-1 + 1\times2 = 2@f$. After this, @f$v@f$ is `[-1,
/// -1, 2]`.
///
/// Next, we multiply @f$v_1@f$ by @f$D_1=1@f$. After this, @f$v@f$ is `[-1, -1,
/// 2]`.
///
/// Next, we add @f$M_{1,i}v_i@f$ to @f$v_1@f$ for all @f$i@f$: in this case, we
/// add @f$-1\times-1 + 0\times-1 + 1\times2 = 3@f$. After this, @f$v@f$ is
/// `[-1, 2, 2]`.
///
/// Next, we multiply @f$v_2@f$ by @f$D_2=4@f$. After this, @f$v@f$ is `[-1, 2,
/// 8]`.
///
/// Next, we add @f$M_{2,i}v_i@f$ to @f$v_2@f$ for all @f$i@f$: in this case, we
/// add @f$-2\times-1 + 0\times2 + 0\times8 = 2@f$. After this, @f$v@f$ is `[-1,
/// 2, 10]`. This final value of @f$v@f$ is what the result of @f$Av@f$
///
/// @param[in] matrix A matrix in precomputed form (as returned by
/// `prepare_matrix()`)
/// @param[in,out] data The data to apply the permutation to
/// @param[in] offset The position in the data to start applying the permutation
/// @param[in] block_size The block size of the data
template <typename T, typename E>
void apply_matrix(const std::tuple<std::vector<std::size_t>, std::vector<T>,
                                   xt::xtensor<T, 2>>& matrix,
                  const xtl::span<E>& data, std::size_t offset = 0,
                  std::size_t block_size = 1)
{
  const std::vector<std::size_t>& v_size_t = std::get<0>(matrix);
  const std::vector<T>& v_t = std::get<1>(matrix);
  const xt::xtensor<T, 2>& M = std::get<2>(matrix);

  const std::size_t dim = v_size_t.size();
  apply_permutation(v_size_t, data, offset, block_size);
  for (std::size_t b = 0; b < block_size; ++b)
  {
    for (std::size_t i = 0; i < dim; ++i)
    {
      data[block_size * (offset + i) + b] *= v_t[i];
      for (std::size_t j = 0; j < dim; ++j)
      {
        data[block_size * (offset + i) + b]
            += M(i, j) * data[block_size * (offset + j) + b];
      }
    }
  }
}

/// Apply a (precomputed) matrix to some transposed data.
///
/// See `apply_matrix()`.
template <typename T, typename E>
void apply_matrix_to_transpose(
    const std::tuple<std::vector<std::size_t>, std::vector<T>,
                     xt::xtensor<T, 2>>& matrix,
    const xtl::span<E>& data, std::size_t offset = 0,
    std::size_t block_size = 1)
{
  const std::vector<std::size_t>& v_size_t = std::get<0>(matrix);
  const std::vector<T>& v_t = std::get<1>(matrix);
  const xt::xtensor<T, 2>& M = std::get<2>(matrix);

  const std::size_t dim = v_size_t.size();
  const std::size_t data_size = (data.size() + block_size - dim) / block_size;
  apply_permutation_to_transpose(v_size_t, data, offset, block_size);
  for (std::size_t b = 0; b < block_size; ++b)
  {
    for (std::size_t i = 0; i < dim; ++i)
    {
      data[data_size * b + offset + i] *= v_t[i];
      for (std::size_t j = 0; j < dim; ++j)
      {
        data[data_size * b + offset + i]
            += M(i, j) * data[data_size * b + offset + j];
      }
    }
  }
}

} // namespace basix::precompute
