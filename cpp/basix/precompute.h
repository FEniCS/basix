// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "math.h"
#include "mdspan.hpp"
#include <concepts>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

/// Matrix and permutation pre-computation
namespace basix::precompute
{
namespace impl
{
/// @private These structs are used to get the float/value type from a
/// template argument, including support for complex types.
template <typename T, typename = void>
struct scalar_value_type
{
  /// @internal
  typedef T value_type;
};
/// @private
template <typename T>
struct scalar_value_type<T, std::void_t<typename T::value_type>>
{
  typedef typename T::value_type value_type;
};
/// @private Convenience typedef
template <typename T>
using scalar_value_type_t = typename scalar_value_type<T>::value_type;
} // namespace impl

/// @brief Prepare a permutation.
///
/// Computes  representation of the permutation that allows the
/// permutation to be applied without any temporary memory assignment.
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
/// Consider the permutation `P = [1, 4, 0, 5, 2, 3]`.
///
/// -# First, we look at the 0th entry. `P[0]` is 1. This is greater
/// than 0, so the 0th entry of the output is 1.
///
/// -# Next, we look at the 1st entry. `P[1]` is 4. This is greater than
/// 1, so the 1st entry of the output is 4.
///
/// -# Next, we look at the 2nd entry. `P[2]` is 0. This is less than 2,
/// so we look at `P[0]. `P[0]` is 1. This is less than 2, so we look at
/// `P[1]`. `P[1]` is 4. This is greater than 2, so the 2nd entry of the
/// output is 4.
///
/// -# Next, we look at the 3rd entry. `P[3]` is 5. This is greater than 3,
/// so the 3rd entry of the output is 5.
///
/// -# Next, we look at the 4th entry. `P[4]` is 2. This is less than 4, so
/// we look at `P[2]`. `P[2]` is 0. This is less than 4, so we look at
/// `P[0]`. `P[0]` is 1. This is less than 4, so we look at `P[1]`.
/// `P[1]` is 4. This is greater than (or equal to) 4, so the 4th entry
/// of the output is 4.
///
/// -# Next, we look at the 5th entry. `P[5]` is 3. This is less than 5,
/// so we look at `P[3]`. `P[3]` is 5. This is greater than (or equal
/// to) 5, so the 5th entry of the output is 5.
///
/// Hence, the output of this function in this case is `[1, 4, 4, 5, 4,
/// 5]`.
///
/// For an example of how the permutation in this form is applied, see
/// apply_permutation().
///
/// @param[in,out] perm A permutation
void prepare_permutation(std::span<std::size_t> perm);

/// @brief Apply a (precomputed) permutation \f$v = P u\f$.
///
/// This uses the representation returned by prepare_permutation() to
/// apply a permutation without needing any temporary memory. In pseudo
/// code, this function does the following:
///
/// \code{.pseudo}
/// FOR index, entry IN perm:
///     SWAP(data[index], data[entry])
/// \endcode
///
/// If `n` is set, this will apply the permutation to every block. The
/// `offset` is set, this will start applying the permutation at the
/// `offset`th block.
///
/// Example
/// -------
/// Consider the permutation `P = [1, 4, 0, 5, 2, 3]`. In the
/// documentation of prepare_permutation(), we saw that the precomputed
/// representation of this permutation is `P2 = [1, 4, 4, 5, 4, 5]`. In
/// this example, we look at how this representation can be used to
/// apply this permutation to the array `A = [a, b, c, d, e, f]`.
///
/// - `P2[0]` is 1, so we swap `A[0]` and `A[1]`. After this, `A = [b,
/// a, c, d, e, f]`.
///
/// - `P2[1]` is 4, so we swap `A[1]` and `A[4]`. After this, `A = [b,
/// e, c, d, a, f]`.
///
/// - `P2[2]` is 4, so we swap `A[2]` and `A[4]`. After this, `A = [b, e,
/// a, d, c, f]`.
///
/// - `P2[3]` is 5, so we swap `A[3]` and `A[5]`. After this, `A = [b,
/// e, a, f, c, d]`.
///
/// - `P2[4]` is 4, so we swap `A[4]` and `A[4]`. This changes nothing.
///
/// - `P2[5]` is 5, so we swap `A[5]` and `A[5]`. This changes nothing.
///
/// Therefore the result of applying this permutation is `[b, e, a, f,
/// c, d]` (which is what we get if we apply the permutation directly).
///
/// @note This function is designed to be called at runtime, so its
/// performance is critical.
///
/// @param[in] perm A permutation in precomputed form (as returned by
/// prepare_permutation()).
/// @param[in,out] data The data to apply the permutation to. It has
/// shape `(m, n)` (uses row-major storage), where the permutation
/// matrix has shape `(m, m)`.
/// @param[in] offset The position in the data to start applying the
/// permutation.
/// @param[in] n The block size of the data.
template <typename E>
void apply_permutation(std::span<const std::size_t> perm, std::span<E> data,
                       std::size_t offset = 0, std::size_t n = 1)
{
  for (std::size_t i = 0; i < perm.size(); ++i)
    for (std::size_t b = 0; b < n; ++b)
      std::swap(data[n * (offset + i) + b], data[n * (offset + perm[i]) + b]);
}

/// @brief Permutation of mapped data.
template <typename E>
void apply_permutation_mapped(std::span<const std::size_t> perm,
                              std::span<E> data, std::span<const int> emap,
                              std::size_t n = 1)
{
  for (std::size_t i = 0; i < perm.size(); ++i)
    for (std::size_t b = 0; b < n; ++b)
      std::swap(data[n * emap[i] + b], data[n * emap[perm[i]] + b]);
}

/// @brief Apply a (precomputed) permutation to some transposed data.
///
/// Applies \f$v = u P^{T}\f$.
///
/// @note This function is designed to be called at runtime, so its
/// performance is critical.
///
/// See apply_permutation().
template <typename E>
void apply_inv_permutation_right(std::span<const std::size_t> perm,
                                 std::span<E> data, std::size_t offset = 0,
                                 std::size_t n = 1)
{
  const std::size_t dim = perm.size();
  const std::size_t data_size = (data.size() + (dim < n ? n - dim : 0)) / n;
  for (std::size_t b = 0; b < n; ++b)
  {
    for (std::size_t i = 0; i < dim; ++i)
    {
      std::swap(data[data_size * b + offset + i],
                data[data_size * b + offset + perm[i]]);
    }
  }
}

/// @brief Prepare a square matrix.
///
/// Computes the LU decomposition of the transpose of the matrix.
///
/// This function returns the permutation @f$P@f$P in the representation
/// @f$PA^t=LU@f$ (precomputed as in prepare_permutation()). The LU
/// decomposition of @f$A^t@f$ is computed in-place
///
/// For an example of how the permutation in this form is applied, see
/// apply_matrix().
///
/// @param[in,out] A The matrix data.
/// @return The three parts of a precomputed representation of the
/// matrix. These are (as described above):
/// - A permutation (precomputed as in prepare_permutation());
/// - the vector @f$D@f$;
template <std::floating_point T>
std::vector<std::size_t>
prepare_matrix(std::pair<std::vector<T>, std::array<std::size_t, 2>>& A)
{
  return math::transpose_lu<T>(A);
}

/// @brief Apply a (precomputed) matrix.
///
/// This uses the representation returned by prepare_matrix() to apply a
/// matrix without needing any temporary memory. In pseudo code, this
/// function does the following:
///
/// \code{.pseudo}
/// INPUT perm, mat, data
/// apply_permutation(perm, data)
/// FOR i IN RANGE(dim):
///     FOR j IN RANGE(i+1, dim):
///         data[i] += mat[i, j] * data[j]
/// FOR i IN RANGE(dim - 1, -1, -1):
///     data[i] *= M[i, i]
///     FOR j in RANGE(i):
///         data[i] += mat[i, j] * data[j]
/// \endcode
///
/// If `n` is set, this will apply the permutation to every block. The
/// `offset` is set, this will start applying the permutation at the
/// `offset`th block.
///
/// @note This function is designed to be called at runtime, so its
/// performance is critical.
///
/// @param[in] v_size_t A permutation, as computed by prepare_matrix().
/// @param[in] M The vector created by prepare_matrix().
/// @param[in,out] data The data to apply the permutation to
/// @param[in] offset The position in the data to start applying the
/// permutation
/// @param[in] n The block size of the data
template <typename T, typename E>
void apply_matrix(
    std::span<const std::size_t> v_size_t,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        M,
    std::span<E> data, std::size_t offset = 0, std::size_t n = 1)
{
  using U = typename impl::scalar_value_type_t<E>;

  const std::size_t dim = v_size_t.size();
  apply_permutation(v_size_t, data, offset, n);
  for (std::size_t b = 0; b < n; ++b)
  {
    for (std::size_t i = 0; i < dim; ++i)
    {
      for (std::size_t j = i + 1; j < dim; ++j)
      {
        data[n * (offset + i) + b]
            += static_cast<U>(M[i, j]) * data[n * (offset + j) + b];
      }
    }
    for (std::size_t i = 1; i <= dim; ++i)
    {
      data[n * (offset + dim - i) + b] *= static_cast<U>(M[dim - i, dim - i]);
      for (std::size_t j = 0; j < dim - i; ++j)
      {
        data[n * (offset + dim - i) + b]
            += static_cast<U>(M[dim - i, j]) * data[n * (offset + j) + b];
      }
    }
  }
}

/// @brief Apply a (precomputed) matrix to some transposed data.
///
/// Computes \f$v^{T} = M u^{T}\f$ (or equivalently \f$v = u M^{T}\f$).
///
/// @note This function is designed to be called at runtime, so its
/// performance is critical.
///
/// See apply_matrix().
template <typename T, typename E>
void apply_tranpose_matrix_right(
    std::span<const std::size_t> v_size_t,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        M,
    std::span<E> data, std::size_t offset = 0, std::size_t n = 1)
{
  using U = typename impl::scalar_value_type_t<E>;

  const std::size_t dim = v_size_t.size();
  const std::size_t data_size = (data.size() + (dim < n ? n - dim : 0)) / n;
  apply_inv_permutation_right(v_size_t, data, offset, n);
  for (std::size_t b = 0; b < n; ++b)
  {
    for (std::size_t i = 0; i < dim; ++i)
    {
      for (std::size_t j = i + 1; j < dim; ++j)
      {
        data[data_size * b + offset + i]
            += static_cast<U>(M[i, j]) * data[data_size * b + offset + j];
      }
    }
    for (std::size_t i = 1; i <= dim; ++i)
    {
      data[data_size * b + offset + dim - i]
          *= static_cast<U>(M[dim - i, dim - i]);
      for (std::size_t j = 0; j < dim - i; ++j)
      {
        data[data_size * b + offset + dim - i]
            += static_cast<U>(M[dim - i, j]) * data[data_size * b + offset + j];
      }
    }
  }
}

} // namespace basix::precompute
