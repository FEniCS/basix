// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

/// Interfaces for creating finite elements
namespace basix::element
{
/// Variants of a Lagrange space that can be created
enum class lagrange_variant
{
  unset = 0,
  equispaced = 1,
  gll_warped = 2,
  gll_isaac = 3,
  gll_centroid = 4,
  chebyshev_warped = 5,
  chebyshev_isaac = 6,
  chebyshev_centroid = 7,
  gl_warped = 8,
  gl_isaac = 9,
  gl_centroid = 10,
  legendre = 11,
  bernstein = 12,
};

/// Variants of a DPC (discontinuous polynomial cubical) space that can
/// be created. DPC spaces span the same set of polynomials as Lagrange
/// spaces on simplices but are defined on tensor product cells.
enum class dpc_variant
{
  unset = 0,
  simplex_equispaced = 1,
  simplex_gll = 2,
  horizontal_equispaced = 3,
  horizontal_gll = 4,
  diagonal_equispaced = 5,
  diagonal_gll = 6,
  legendre = 7,
};

/// Available element families
enum class family
{
  custom = 0,
  P = 1,
  RT = 2,
  N1E = 3,
  BDM = 4,
  N2E = 5,
  CR = 6,
  Regge = 7,
  DPC = 8,
  bubble = 9,
  serendipity = 10,
  HHJ = 11,
  Hermite = 12,
  iso = 13,
};
} // namespace basix::element
