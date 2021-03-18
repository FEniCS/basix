// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "span.hpp"
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

/// Information about finite element maps
namespace basix::mapping
{

/// Cell type
enum class type
{
  identity,
  covariantPiola,
  contravariantPiola,
  doubleCovariantPiola,
  doubleContravariantPiola,
};

// Get the function that maps data from the reference to the physical
// cell
// @param mapping_type Mapping type
// @return The mapping function
// std::function<std::vector<double>(const tcb::span<const double>&,
//                                   const xt::xtensor<double, 2>&, double,
//                                   const xt::xtensor<double, 2>&)>
// get_forward_map(map::type mapping_type);

/// Convert mapping type enum to string
const std::string& type_to_str(map::type type);

namespace impl
{
template <typename T, typename O>
auto dot22(const T& A, const O& B)
{
  using value_type
      = std::common_type_t<typename T::value_type, typename O::value_type>;
  assert(A.shape(1) == B.shape(0));
  xt::xtensor<value_type, 2> r({A.shape(0), B.shape(1)});
  for (std::size_t i = 0; i < r.shape(0); ++i)
  {
    for (std::size_t j = 0; j < r.shape(1); ++j)
    {
      r(i, j) = 0;
      for (std::size_t k = 0; k < A.shape(1); ++k)
        r(i, j) += A(i, k) * B(k, j);
    }
  }
  return r;
}

template <typename T, typename O>
auto dot21(const T& A, const O& B)
{
  using value_type
      = std::common_type_t<typename T::value_type, typename O::value_type>;
  assert(A.shape(1) == B.shape(0));
  std::array<std::size_t, 1> s = {A.shape(0)};
  xt::xtensor<value_type, 1> r(s);
  for (std::size_t i = 0; i < r.shape(0); ++i)
  {
    r[i] = 0;
    for (std::size_t k = 0; k < A.shape(1); ++k)
      r[i] += A(i, k) * B[k];
  }
  return r;
}

template <typename T>
std::vector<T> identity(const tcb::span<const T>& U,
                        const xt::xtensor<double, 2>& /*J*/, double /*detJ*/,
                        const xt::xtensor<double, 2>& /*K*/)
{
  return std::vector<T>(U.begin(), U.end());
}

template <typename T>
std::vector<T> covariant_piola(const tcb::span<const T>& U,
                               const xt::xtensor<double, 2>& /*J*/,
                               double /*detJ*/, const xt::xtensor<double, 2>& K)
{
  std::array<std::size_t, 1> s = {U.size()};
  auto _U = xt::adapt(U.data(), U.size(), xt::no_ownership(), s);
  auto r = dot21(xt::transpose(K), _U);
  return std::vector<T>(r.begin(), r.end());
}

template <typename T>
std::vector<T> contravariant_piola(const tcb::span<const T>& U,
                                   const xt::xtensor<double, 2>& J, double detJ,
                                   const xt::xtensor<double, 2>& /*K*/)
{
  std::array<std::size_t, 1> s = {U.size()};
  auto _U = xt::adapt(U.data(), U.size(), xt::no_ownership(), s);
  auto r = 1 / detJ * dot21(J, _U);
  return std::vector<T>(r.begin(), r.end());
}

template <typename T>
std::vector<T> double_covariant_piola(const tcb::span<const T>& U,
                                      const xt::xtensor<double, 2>& J,
                                      double /*detJ*/,
                                      const xt::xtensor<double, 2>& K)
{
  std::array<std::size_t, 2> s = {J.shape(1), J.shape(1)};
  auto data_matrix = xt::adapt(U.data(), U.size(), xt::no_ownership(), s);
  auto r = dot22(xt::transpose(K), dot22(data_matrix, K));
  return std::vector<T>(r.begin(), r.end());
}

template <typename T>
std::vector<T> double_contravariant_piola(const tcb::span<const T>& U,
                                          const xt::xtensor<double, 2>& J,
                                          double detJ,
                                          const xt::xtensor<double, 2>& /*K*/)
{
  std::array<std::size_t, 2> s = {J.shape(1), J.shape(1)};
  auto data_matrix = xt::adapt(U.data(), U.size(), xt::no_ownership(), s);

  auto r = 1 / (detJ * detJ) * dot22(J, dot22(data_matrix, xt::transpose(J)));
  return std::vector<T>(r.begin(), r.end());
}
} // namespace impl

/// TODO
template <typename T, typename P, typename O>
std::vector<T> apply_map(const tcb::span<const T>& U, const P& J, double detJ,
                         const O& K, mapp::type map_type)
{
  switch (map_type)
  {
  case map::type::identity:
    return impl::identity(U, J, detJ, K);
  case map::type::covariantPiola:
    return impl::covariant_piola(U, J, detJ, K);
  case map::type::contravariantPiola:
    return impl::contravariant_piola(U, J, detJ, K);
  case map::type::doubleCovariantPiola:
    return impl::double_covariant_piola(U, J, detJ, K);
  case map::type::doubleContravariantPiola:
    return impl::double_contravariant_piola(U, J, detJ, K);
  default:
    throw std::runtime_error("Mapping not yet implemented");
  }
}

} // namespace basix::mapping
