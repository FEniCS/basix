// Copyright (c) 2020-2024 Chris Richardson, Matthew Scroggs and Garth N. Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "mdspan.hpp"
#include <array>
#include <vector>

namespace basix::impl
{
template <typename T, std::size_t d>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, d>>;
template <typename T, std::size_t d>
using mdarray_t
    = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE::mdarray<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, d>>;

/// Create a container of cmdspan2_t objects from a container of
/// mdarray2_t objects
template <typename T>
std::array<std::vector<mdspan_t<const T, 2>>, 4>
to_mdspan(std::array<std::vector<mdarray_t<T, 2>>, 4>& x)
{
  std::array<std::vector<mdspan_t<const T, 2>>, 4> x1;
  for (std::size_t i = 0; i < x.size(); ++i)
    for (std::size_t j = 0; j < x[i].size(); ++j)
      x1[i].emplace_back(x[i][j].data(), x[i][j].extents());

  return x1;
}

/// Create a container of cmdspan4_t objects from a container of
/// mdarray4_t objects
template <typename T>
std::array<std::vector<mdspan_t<const T, 4>>, 4>
to_mdspan(std::array<std::vector<mdarray_t<T, 4>>, 4>& M)
{
  std::array<std::vector<mdspan_t<const T, 4>>, 4> M1;
  for (std::size_t i = 0; i < M.size(); ++i)
    for (std::size_t j = 0; j < M[i].size(); ++j)
      M1[i].emplace_back(M[i][j].data(), M[i][j].extents());

  return M1;
}

/// Create a container of cmdspan2_t objects from containers holding
/// data buffers and shapes
template <typename T>
std::array<std::vector<mdspan_t<const T, 2>>, 4>
to_mdspan(const std::array<std::vector<std::vector<T>>, 4>& x,
          const std::array<std::vector<std::array<std::size_t, 2>>, 4>& shape)
{
  std::array<std::vector<mdspan_t<const T, 2>>, 4> x1;
  for (std::size_t i = 0; i < x.size(); ++i)
    for (std::size_t j = 0; j < x[i].size(); ++j)
      x1[i].push_back(mdspan_t<const T, 2>(x[i][j].data(), shape[i][j]));

  return x1;
}

/// Create a container of cmdspan4_t objects from containers holding
/// data buffers and shapes
template <typename T>
std::array<std::vector<mdspan_t<const T, 4>>, 4>
to_mdspan(const std::array<std::vector<std::vector<T>>, 4>& M,
          const std::array<std::vector<std::array<std::size_t, 4>>, 4>& shape)
{
  std::array<std::vector<mdspan_t<const T, 4>>, 4> M1;
  for (std::size_t i = 0; i < M.size(); ++i)
    for (std::size_t j = 0; j < M[i].size(); ++j)
      M1[i].push_back(mdspan_t<const T, 4>(M[i][j].data(), shape[i][j]));

  return M1;
}

} // namespace basix::impl
