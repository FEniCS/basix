// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "precompute.h"
#include "math.h"
#include <numeric>

using namespace basix;

namespace stdex = std::experimental;
using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
using mdarray2_t = stdex::mdarray<double, stdex::dextents<std::size_t, 2>>;

//-----------------------------------------------------------------------------
void precompute::prepare_permutation(const std::span<std::size_t>& perm)
{
  for (std::size_t row = 0; row < perm.size(); ++row)
  {
    while (perm[row] < row)
      perm[row] = perm[perm[row]];
  }
}
//-----------------------------------------------------------------------------
std::vector<std::size_t> precompute::prepare_matrix(
    std::pair<std::vector<double>, std::array<std::size_t, 2>>& A)
{
  return math::transpose_lu(A);
}
//-----------------------------------------------------------------------------
