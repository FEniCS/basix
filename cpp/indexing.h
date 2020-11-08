// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

namespace libtab
{
/// Compute trivial indexing in a 1D array (for completeness)
/// @param p Index in x
/// @return 1D Index
static inline int idx(int p) { return p; }

/// Compute indexing in a 2D triangular array compressed into a 1D array
/// @param p Index in x
/// @param q Index in y
/// @return 1D Index
static inline int idx(int p, int q) { return (p + q + 1) * (p + q) / 2 + q; }

/// Compute indexing in a 3D tetrahedral array compressed into a 1D array
/// @param p Index in x
/// @param q Index in y
/// @param r Index in z
/// @return 1D Index
static inline int idx(int p, int q, int r)
{
  return (p + q + r) * (p + q + r + 1) * (p + q + r + 2) / 6
         + (q + r) * (q + r + 1) / 2 + r;
};
}
