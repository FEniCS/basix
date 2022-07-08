// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "mdspan.hpp"
#include <stdexcept>
#include <type_traits>
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xspan.hpp>

#include <xtensor/xio.hpp>

/// Information about finite element maps
namespace basix::maps
{

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

namespace impl
{
template <typename O, typename Mat0, typename Mat1, typename Mat2>
void dot22(O&& r, const Mat0& A, const Mat1& B, const Mat2& C)
{
  assert(A.shape(1) == B.shape(0));
  using T = typename std::decay_t<O>::value_type;
  for (std::size_t i = 0; i < r.shape(0); ++i)
  {
    for (std::size_t j = 0; j < r.shape(1); ++j)
    {
      T acc = 0;
      for (std::size_t k = 0; k < A.shape(1); ++k)
        for (std::size_t l = 0; l < B.shape(1); ++l)
          acc += A(i, k) * B(k, l) * C(l, j);
      r(i, j) = acc;
    }
  }
}

template <typename Vec, typename Mat0, typename Mat1>
void dot21(Vec&& r, const Mat0& A, const Mat1& B)
{
  using T = typename std::decay_t<Vec>::value_type;
  for (std::size_t i = 0; i < r.shape(0); ++i)
  {
    T acc = 0;
    for (std::size_t k = 0; k < A.shape(1); ++k)
      acc += A(i, k) * B[k];
    r[i] = acc;
  }
}
} // namespace impl

/// L2 Piola map
template <typename O, typename P, typename Q, typename R>
void l2_piola_old(O&& r, const P& U, const Q& /*J*/, double detJ,
                  const R& /*K*/)
{
  r.assign(U);
  std::for_each(r.begin(), r.end(), [detJ](auto& ri) { ri /= detJ; });
}

/// L2 Piola map (new)
template <typename O, typename P, typename Q, typename R>
void l2_piola(O&& r, const P& U, const Q& /*J*/, double detJ, const R& /*K*/)
{
  assert(U.extent(0) == r.extent(0));
  assert(U.extent(1) == r.extent(1));
  for (std::size_t i = 0; i < U.extent(0); ++i)
    for (std::size_t j = 0; j < U.extent(1); ++j)
      r(i, j) = U(i, j) / detJ;
}

/// Covariant Piola map
template <typename O, typename P, typename Q, typename R>
void covariant_piola_old(O&& r, const P& U, const Q& /*J*/, double /*detJ*/,
                         const R& K)
{
  auto Kt = xt::transpose(K);

  // std::cout << "Kt" << std::endl;
  // std::cout << Kt << std::endl;
  // std::cout << "r" << std::endl;
  // std::cout << r << std::endl;
  // std::cout << "U" << std::endl;
  // std::cout << U << std::endl;

  for (std::size_t p = 0; p < U.shape(0); ++p)
  {
    auto r_p = xt::row(r, p);
    auto U_p = xt::row(U, p);
    impl::dot21(r_p, Kt, U_p);
  }

  // std::cout << "r-post" << std::endl;
  // std::cout << r << std::endl;
}

/// Covariant Piola map (new)
template <typename O, typename P, typename Q, typename R>
void covariant_piola(O&& r, const P& U, const Q& /*J*/, double /*detJ*/,
                     const R& K)
{
  // std::cout << "Kt" << std::endl;
  // for (std::size_t k0 = 0; k0 < K.extent(1); ++k0)
  // {
  //   std::cout << "   ";
  //   for (std::size_t k1 = 0; k1 < K.extent(0); ++k1)
  //     std::cout << K(k1, k0) << "   ";
  //   std::cout << std::endl;
  // }

  // std::cout << "r" << std::endl;
  // for (std::size_t k0 = 0; k0 < r.extent(0); ++k0)
  // {
  //   std::cout << "   ";
  //   for (std::size_t k1 = 0; k1 < r.extent(1); ++k1)
  //     std::cout << r(k0, k1) << "   ";
  //   std::cout << std::endl;
  // }

  // std::cout << "U" << std::endl;
  // for (std::size_t k0 = 0; k0 < U.extent(0); ++k0)
  // {
  //   std::cout << "   ";
  //   for (std::size_t k1 = 0; k1 < U.extent(1); ++k1)
  //     std::cout << U(k0, k1) << "   ";
  //   std::cout << std::endl;
  // }

  using T = typename std::decay_t<O>::value_type;
  for (std::size_t p = 0; p < U.extent(0); ++p)
  {
    // r_p = K^T U_p, where p indicates the p-th row
    for (std::size_t i = 0; i < r.extent(1); ++i)
    {
      T acc = 0;
      for (std::size_t k = 0; k < K.extent(0); ++k)
        acc += K(k, i) * U(p, k);
      r(p, i) = acc;
    }
  }

  // std::cout << "r-post" << std::endl;
  // for (std::size_t k0 = 0; k0 < r.extent(0); ++k0)
  // {
  //   std::cout << "   ";
  //   for (std::size_t k1 = 0; k1 < r.extent(1); ++k1)
  //     std::cout << r(k0, k1) << "   ";
  //   std::cout << std::endl;
  // }
}

/// Contravariant Piola map
template <typename O, typename P, typename Q, typename R>
void contravariant_piola_old(O&& r, const P& U, const Q& J, double detJ,
                             const R& /*K*/)
{
  for (std::size_t p = 0; p < U.shape(0); ++p)
  {
    auto r_p = xt::row(r, p);
    auto U_p = xt::row(U, p);
    impl::dot21(r_p, J, U_p);
  }
  double inv_detJ = 1 / detJ;
  std::for_each(r.begin(), r.end(), [inv_detJ](auto& ri) { ri *= inv_detJ; });
}

/// Contravariant Piola map (new)
template <typename O, typename P, typename Q, typename R>
void contravariant_piola(O&& r, const P& U, const Q& J, double detJ,
                         const R& /*K*/)
{
  using T = typename std::decay_t<O>::value_type;
  for (std::size_t p = 0; p < U.extent(0); ++p)
  {
    for (std::size_t i = 0; i < r.extent(1); ++i)
    {
      T acc = 0;
      for (std::size_t k = 0; k < J.extent(1); ++k)
        acc += J(i, k) * U(p, k);
      r(p, i) = acc;
    }
  }

  std::transform(r.data(), r.data() + r.size(), r.data(),
                 [detJ](auto ri) { return ri / detJ; });
}

/// Double covariant Piola map
template <typename O, typename P, typename Q, typename R>
void double_covariant_piola_old(O&& r, const P& U, const Q& J, double /*detJ*/,
                                const R& K)
{
  for (std::size_t p = 0; p < U.shape(0); ++p)
  {
    auto r_p = xt::row(r, p);
    auto U_p = xt::row(U, p);
    auto _U = xt::reshape_view(U_p, {J.shape(1), J.shape(1)});
    auto _r = xt::reshape_view(r_p, {K.shape(1), K.shape(1)});
    impl::dot22(_r, xt::transpose(K), _U, K);
  }
}

/// Double covariant Piola map (new)
template <typename O, typename P, typename Q, typename R>
void double_covariant_piola(O&& r, const P& U, const Q& J, double /*detJ*/,
                            const R& K)
{
  namespace stdex = std::experimental;
  using T = typename std::decay_t<O>::value_type;
  for (std::size_t p = 0; p < U.extent(0); ++p)
  {
    stdex::mdspan<const T, stdex::dextents<std::size_t, 2>> _U(
        U.data() + p * U.extent(1), J.extent(1), J.extent(1));
    stdex::mdspan<T, stdex::dextents<std::size_t, 2>> _r(
        r.data() + p * r.extent(1), K.extent(1), K.extent(1));
    // _r = K^T _U K
    for (std::size_t i = 0; i < _r.extent(0); ++i)
    {
      for (std::size_t j = 0; j < _r.extent(1); ++j)
      {
        T acc = 0;
        for (std::size_t k = 0; k < K.extent(0); ++k)
          for (std::size_t l = 0; l < _U.extent(1); ++l)
            acc += K(k, i) * _U(k, l) * K(l, j);
        _r(i, j) = acc;
      }
    }
  }
}

/// Double contravariant Piola map
template <typename O, typename P, typename Q, typename R>
void double_contravariant_piola_old(O&& r, const P& U, const Q& J, double detJ,
                                    const R& /*K*/)
{
  auto Jt = xt::transpose(J);
  for (std::size_t p = 0; p < U.shape(0); ++p)
  {
    auto r_p = xt::row(r, p);
    auto U_p = xt::row(U, p);
    auto _U = xt::reshape_view(U_p, {J.shape(1), J.shape(1)});
    auto _r = xt::reshape_view(r_p, {J.shape(0), J.shape(0)});
    impl::dot22(_r, J, _U, Jt);
  }
  double inv_detJ = 1 / (detJ * detJ);
  std::for_each(r.begin(), r.end(), [inv_detJ](auto& ri) { ri *= inv_detJ; });
}

/// Double contravariant Piola map (new)
template <typename O, typename P, typename Q, typename R>
void double_contravariant_piola(O&& r, const P& U, const Q& J, double detJ,
                                const R& /*K*/)
{
  namespace stdex = std::experimental;
  using T = typename std::decay_t<O>::value_type;

  for (std::size_t p = 0; p < U.extent(0); ++p)
  {
    stdex::mdspan<const T, stdex::dextents<std::size_t, 2>> _U(
        U.data() + p * U.extent(1), J.extent(1), J.extent(1));
    stdex::mdspan<T, stdex::dextents<std::size_t, 2>> _r(
        r.data() + p * r.extent(1), J.extent(0), J.extent(0));

    // _r - J U J^T
    for (std::size_t i = 0; i < _r.extent(0); ++i)
    {
      for (std::size_t j = 0; j < _r.extent(1); ++j)
      {
        T acc = 0;
        for (std::size_t k = 0; k < J.extent(1); ++k)
          for (std::size_t l = 0; l < _U.extent(1); ++l)
            acc += J(i, k) * _U(k, l) * J(j, l);
        _r(i, j) = acc;
      }
    }
  }

  std::transform(r.data(), r.data() + r.size(), r.data(),
                 [detJ](auto ri) { return ri / (detJ * detJ); });
}

} // namespace basix::maps
