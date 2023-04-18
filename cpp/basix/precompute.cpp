// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "precompute.h"

using namespace basix;

//-----------------------------------------------------------------------------
void precompute::prepare_permutation(std::span<std::size_t> perm)
{
  for (std::size_t row = 0; row < perm.size(); ++row)
  {
    while (perm[row] < row)
      perm[row] = perm[perm[row]];
  }
}
//-----------------------------------------------------------------------------
