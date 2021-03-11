// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "mappings.h"
#include <functional>
#include <map>
#include <stdexcept>
#include <xtensor/xadapt.hpp>
// #include <xtensor-blas/xlinalg.hpp>

namespace
{

//-----------------------------------------------------------------------------
// TODO: replace this with xt::linalg::dot
xt::xtensor<double, 2> dot22(xt::xtensor<double, 2> A, xt::xtensor<double, 2> B)
{
  assert(A.shape(1) == B.shape(0));
  std::array<std::size_t, 2> s = {A.shape(0), B.shape(1)};

  xt::xtensor<double, 2> r(s);
  for (std::size_t i = 0; i < s[0]; ++i)
    for (std::size_t j = 0; j < s[1]; ++j)
    {
      r(i, j) = 0;
      for (std::size_t k = 0; k < A.shape(1); ++k)
        r(i, j) += A(i, k) * B(k, j);
    }
  return r;
}
//-----------------------------------------------------------------------------
// TODO: replace this with xt::linalg::dot
xt::xtensor<double, 1> dot21(xt::xtensor<double, 2> A, xt::xtensor<double, 1> B)
{
  assert(A.shape(1) == B.shape(0));
  std::array<std::size_t, 1> s = {A.shape(0)};

  xt::xtensor<double, 1> r(s);
  for (std::size_t i = 0; i < s[0]; ++i)
  {
    r(i) = 0;
    for (std::size_t k = 0; k < A.shape(1); ++k)
      r(i) += A(i, k) * B(k);
  }
  return r;
}
//-----------------------------------------------------------------------------
std::vector<double> identity(const tcb::span<const double>& reference_data,
                             const xt::xtensor<double, 2>& /*J*/,
                             double /*detJ*/,
                             const xt::xtensor<double, 2>& /*K*/)
{
  return std::vector<double>(reference_data.begin(), reference_data.end());
}
//-----------------------------------------------------------------------------
std::vector<double>
covariant_piola(const tcb::span<const double>& reference_data,
                const xt::xtensor<double, 2>& /*J*/, double /*detJ*/,
                const xt::xtensor<double, 2>& K)
{
  std::array<std::size_t, 1> s = {reference_data.size()};
  auto _reference_data = xt::adapt(reference_data.data(), reference_data.size(),
                                   xt::no_ownership(), s);

  auto r = dot21(xt::transpose(K), _reference_data);
  return std::vector<double>(r.begin(), r.end());
}
//-----------------------------------------------------------------------------
std::vector<double>
contravariant_piola(const tcb::span<const double>& reference_data,
                    const xt::xtensor<double, 2>& J, double detJ,
                    const xt::xtensor<double, 2>& /*K*/)
{
  std::array<std::size_t, 1> s = {reference_data.size()};
  auto _reference_data = xt::adapt(reference_data.data(), reference_data.size(),
                                   xt::no_ownership(), s);
  auto r = 1 / detJ * dot21(J, _reference_data);
  return std::vector<double>(r.begin(), r.end());
}
//-----------------------------------------------------------------------------
std::vector<double>
double_covariant_piola(const tcb::span<const double>& reference_data,
                       const xt::xtensor<double, 2>& J, double /*detJ*/,
                       const xt::xtensor<double, 2>& K)
{
  std::array<std::size_t, 2> s = {J.shape(1), J.shape(1)};
  auto data_matrix = xt::adapt(reference_data.data(), reference_data.size(),
                               xt::no_ownership(), s);
  xt::xtensor<double, 2> r = dot22(xt::transpose(K), dot22(data_matrix, K));
  return std::vector<double>(r.begin(), r.end());
}
//-----------------------------------------------------------------------------
std::vector<double>
double_contravariant_piola(const tcb::span<const double>& reference_data,
                           const xt::xtensor<double, 2>& J, double detJ,
                           const xt::xtensor<double, 2>& /*K*/)
{
  std::array<std::size_t, 2> s = {J.shape(1), J.shape(1)};
  auto data_matrix = xt::adapt(reference_data.data(), reference_data.size(),
                               xt::no_ownership(), s);

  xt::xtensor<double, 2> r
      = 1 / (detJ * detJ) * dot22(J, dot22(data_matrix, xt::transpose(J)));
  return std::vector<double>(r.begin(), r.end());
}
//-----------------------------------------------------------------------------
} // namespace

using namespace basix;

//-----------------------------------------------------------------------------
std::function<std::vector<double>(const tcb::span<const double>&,
                                  const xt::xtensor<double, 2>&, double,
                                  const xt::xtensor<double, 2>&)>
mapping::get_forward_map(mapping::type mapping_type)
{
  switch (mapping_type)
  {
  case mapping::type::identity:
    return identity;
  case mapping::type::covariantPiola:
    return covariant_piola;
  case mapping::type::contravariantPiola:
    return contravariant_piola;
  case mapping::type::doubleCovariantPiola:
    return double_covariant_piola;
  case mapping::type::doubleContravariantPiola:
    return double_contravariant_piola;
  default:
    throw std::runtime_error("Mapping not yet implemented");
  }
}
//-----------------------------------------------------------------------------
const std::string& mapping::type_to_str(mapping::type type)
{
  static const std::map<mapping::type, std::string> type_to_name = {
      {mapping::type::identity, "identity"},
      {mapping::type::covariantPiola, "covariant Piola"},
      {mapping::type::contravariantPiola, "contravariant Piola"},
      {mapping::type::doubleCovariantPiola, "double covariant Piola"},
      {mapping::type::doubleContravariantPiola, "double contravariant Piola"}};

  auto it = type_to_name.find(type);
  if (it == type_to_name.end())
    throw std::runtime_error("Can't find type");

  return it->second;
}
//-----------------------------------------------------------------------------
