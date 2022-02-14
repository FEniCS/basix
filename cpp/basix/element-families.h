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
  unset = -1,
  equispaced = 0,
  gll_warped = 1,
  gll_isaac = 2,
  gll_centroid = 3,
  chebyshev_warped = 4,
  chebyshev_isaac = 5,
  chebyshev_centroid = 6,
  gl_warped = 7,
  gl_isaac = 8,
  gl_centroid = 9,
  legendre = 10,
  vtk = 20,
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
  serendipity = 10
};

} // namespace basix::element
