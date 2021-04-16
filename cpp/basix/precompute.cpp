// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "precompute.h"

using namespace basix;

std::vector<std::size_t>
precompute::prepare_permutation(const std::vector<std::size_t>& perm)
{
  std::vector<std::size_t> f_perm(perm.size());
  for (std::size_t row = 0; row < perm.size(); ++row)
  {
    std::size_t row2 = perm[row];
    while (row2 < row)
      row2 = perm[row2];
    f_perm[row] = row2;
  }
  return f_perm;
}
