// Copyright (c) 2021-2022 Matthew Scroggs and Garth N. Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "mdspan.hpp"
#include <algorithm>
#include <stdexcept>
#include <type_traits>

/// Information about finite element maps
namespace basix::maps
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

/// Map type
enum class type
{
  identity = 0,
  L2Piola = 1,
  covariantPiola = 2,
  contravariantPiola = 3,
  doubleCovariantPiola = 4,
  doubleContravariantPiola = 5,
};

/// @brief L2 Piola map
template <typename O, typename P, typename Q, typename R>
void l2_piola(O&& r, const P& U, const Q& /*J*/, double detJ, const R& /*K*/)
{
  assert(U.extent(0) == r.extent(0));
  assert(U.extent(1) == r.extent(1));
  for (std::size_t i = 0; i < U.extent(0); ++i)
    for (std::size_t j = 0; j < U.extent(1); ++j)
      r(i, j) = U(i, j) / detJ;
}

/// @brief  Covariant Piola map
template <typename O, typename P, typename Q, typename R>
void covariant_piola(O&& r, const P& U, const Q& /*J*/, double /*detJ*/,
                     const R& K)
{
  using T = typename std::decay_t<O>::value_type;
  using Z = typename impl::scalar_value_type_t<T>;
  for (std::size_t p = 0; p < U.extent(0); ++p)
  {
    // r_p = K^T U_p, where p indicates the p-th row
    for (std::size_t i = 0; i < r.extent(1); ++i)
    {
      T acc = 0;
      for (std::size_t k = 0; k < K.extent(0); ++k)
        acc += static_cast<Z>(K[k, i]) * U[p, k];
      r[p, i] = acc;
    }
  }
}

/// @brief Contravariant Piola map
template <typename O, typename P, typename Q, typename R>
void contravariant_piola(O&& r, const P& U, const Q& J, double detJ,
                         const R& /*K*/)
{
  using T = typename std::decay_t<O>::value_type;
  using Z = typename impl::scalar_value_type_t<T>;
  for (std::size_t p = 0; p < U.extent(0); ++p)
  {
    for (std::size_t i = 0; i < r.extent(1); ++i)
    {
      T acc = 0;
      for (std::size_t k = 0; k < J.extent(1); ++k)
        acc += static_cast<Z>(J[i, k]) * U[p, k];
      r[p, i] = acc;
    }
  }

  std::transform(r.data_handle(), r.data_handle() + r.size(), r.data_handle(),
                 [detJ](auto ri) { return ri / static_cast<Z>(detJ); });
}

/// @brief Double covariant Piola map
template <typename O, typename P, typename Q, typename R>
void double_covariant_piola(O&& r, const P& U, const Q& J, double /*detJ*/,
                            const R& K)
{
  namespace stdex
      = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;
  using T = typename std::decay_t<O>::value_type;
  using Z = typename impl::scalar_value_type_t<T>;
  for (std::size_t p = 0; p < U.extent(0); ++p)
  {
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        _U(U.data_handle() + p * U.extent(1), J.extent(1), J.extent(1));
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        _r(r.data_handle() + p * r.extent(1), K.extent(1), K.extent(1));
    // _r = K^T _U K
    for (std::size_t i = 0; i < _r.extent(0); ++i)
    {
      for (std::size_t j = 0; j < _r.extent(1); ++j)
      {
        T acc = 0;
        for (std::size_t k = 0; k < K.extent(0); ++k)
          for (std::size_t l = 0; l < _U.extent(1); ++l)
            acc += static_cast<Z>(K[k, i]) * _U[k, l] * static_cast<Z>(K[l, j]);
        _r[i, j] = acc;
      }
    }
  }
}

/// @brief Double contravariant Piola map
template <typename O, typename P, typename Q, typename R>
void double_contravariant_piola(O&& r, const P& U, const Q& J, double detJ,
                                const R& /*K*/)
{
  namespace stdex
      = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;
  using T = typename std::decay_t<O>::value_type;
  using Z = typename impl::scalar_value_type_t<T>;
  for (std::size_t p = 0; p < U.extent(0); ++p)
  {
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        _U(U.data_handle() + p * U.extent(1), J.extent(1), J.extent(1));
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        _r(r.data_handle() + p * r.extent(1), J.extent(0), J.extent(0));

    // _r = J U J^T
    for (std::size_t i = 0; i < _r.extent(0); ++i)
    {
      for (std::size_t j = 0; j < _r.extent(1); ++j)
      {
        T acc = 0;
        for (std::size_t k = 0; k < J.extent(1); ++k)
          for (std::size_t l = 0; l < _U.extent(1); ++l)
            acc += static_cast<Z>(J[i, k]) * _U[k, l] * static_cast<Z>(J[j, l]);
        _r[i, j] = acc;
      }
    }
  }

  std::transform(r.data_handle(), r.data_handle() + r.size(), r.data_handle(),
                 [detJ](auto ri) { return ri / static_cast<Z>(detJ * detJ); });
}

} // namespace basix::maps
