// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include <stdexcept>
#include <type_traits>
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xspan.hpp>

/// Information about finite element maps
namespace basix::maps
{

/// Map type
enum class type
{
  identity,
  covariantPiola,
  contravariantPiola,
  doubleCovariantPiola,
  doubleContravariantPiola,
};

namespace impl
{
template <typename O, typename Mat0, typename Mat1, typename Mat2>
void dot22(O&& r, const Mat0& A, const Mat1& B, const Mat2& C)
{
  assert(A.shape(1) == B.shape(0));
  using T = typename std::decay_t<O>::value_type;
  for (std::size_t i = 0; i < r.shape(0); ++i)
    for (std::size_t j = 0; j < r.shape(1); ++j)
    {
      T acc{};
      for (std::size_t k = 0; k < A.shape(1); ++k)
        for (std::size_t l = 0; l < B.shape(1); ++l)
          acc += A(i, k) * B(k, l) * C(l, j);
      r(i, j) = acc;
    }
}

template <typename Vec, typename Mat0, typename Mat1>
void dot21(Vec&& r, const Mat0& A, const Mat1& B)
{
  using T = typename std::decay_t<Vec>::value_type;
  for (std::size_t i = 0; i < r.shape(0); ++i)
  {
    T acc{};
    for (std::size_t k = 0; k < A.shape(1); ++k)
      acc += A(i, k) * B[k];
    r[i] = acc;
  }
}

template <typename Vec0, typename Vec1, typename Mat0, typename Mat1>
void identity(Vec0&& r, const Vec1& U, const Mat0& /*J*/, double /*detJ*/,
              const Mat1& /*K*/)
{

  r.assign(U);
}

template <typename O, typename P, typename Q, typename R>
void covariant_piola(O&& r, const P& U, const Q& /*J*/, double /*detJ*/,
                     const R& K)
{
  auto Kt = xt::transpose(K);
  for (std::size_t p = 0; p < U.shape(0); ++p)
  {
    auto r_p = xt::row(r, p);
    auto U_p = xt::row(U, p);
    dot21(r_p, Kt, U_p);
  }
}

template <typename O, typename P, typename Q, typename R>
void contravariant_piola(O&& r, const P& U, const Q& J, double detJ,
                         const R& /*K*/)
{
  for (std::size_t p = 0; p < U.shape(0); ++p)
  {
    auto r_p = xt::row(r, p);
    auto U_p = xt::row(U, p);
    dot21(r_p, J, U_p);
  }
  double inv_detJ = 1 / detJ;
  std::for_each(r.begin(), r.end(), [inv_detJ](auto& ri) { ri *= inv_detJ; });
}

template <typename O, typename P, typename Q, typename R>
void double_covariant_piola(O&& r, const P& U, const Q& J, double /*detJ*/,
                            const R& K)
{
  for (std::size_t p = 0; p < U.shape(0); ++p)
  {
    auto r_p = xt::row(r, p);
    auto U_p = xt::row(U, p);
    auto _U = xt::reshape_view(U_p, {J.shape(1), J.shape(1)});
    auto _r = xt::reshape_view(r_p, {K.shape(1), K.shape(1)});
    dot22(_r, xt::transpose(K), _U, K);
  }
}

template <typename O, typename P, typename Q, typename R>
void double_contravariant_piola(O&& r, const P& U, const Q& J, double detJ,
                                const R& /*K*/)
{
  auto Jt = xt::transpose(J);
  for (std::size_t p = 0; p < U.shape(0); ++p)
  {
    auto r_p = xt::row(r, p);
    auto U_p = xt::row(U, p);
    auto _U = xt::reshape_view(U_p, {J.shape(1), J.shape(1)});
    auto _r = xt::reshape_view(r_p, {J.shape(0), J.shape(0)});
    dot22(_r, J, _U, Jt);
  }
  double inv_detJ = 1 / (detJ * detJ);
  std::for_each(r.begin(), r.end(), [inv_detJ](auto& ri) { ri *= inv_detJ; });
}
} // namespace impl

/// Apply a map to data. Note that the required input arguments depends
/// on the type of map.
///
/// @note This function is designed to be called at runtime, so its performance
/// is critical.
/// @todo Remove the switch from this function if possible
///
/// @param[out] u The field after mapping, flattened with row-major
/// layout
/// @param[in] U The field to be mapped, flattened with row-major layout
/// @param[in] J Jacobian of the map
/// @param[in] detJ Determinant of `J`
/// @param[in] K The inverse of `J`
/// @param[in] map_type The map type
template <typename O, typename P, typename Mat0, typename Mat1>
void apply_map(O&& u, const P& U, const Mat0& J, double detJ, const Mat1& K,
               maps::type map_type)
{
  switch (map_type)
  {
  case maps::type::identity:
    return impl::identity(u, U, J, detJ, K);
  case maps::type::covariantPiola:
    return impl::covariant_piola(u, U, J, detJ, K);
  case maps::type::contravariantPiola:
    return impl::contravariant_piola(u, U, J, detJ, K);
  case maps::type::doubleCovariantPiola:
    return impl::double_covariant_piola(u, U, J, detJ, K);
  case maps::type::doubleContravariantPiola:
    return impl::double_contravariant_piola(u, U, J, detJ, K);
  default:
    throw std::runtime_error("Mapping not yet implemented");
  }
}

} // namespace basix::maps
