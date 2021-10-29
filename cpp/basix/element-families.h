// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

namespace basix
{

/// Finite elements

/// The functions in this namespace can be used to create finite elements.
namespace element
{
/// An enum defining the variants of a Lagrange space that can be created
enum class lagrange_variant
{
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
};

/// Enum of available element families
enum class family
{
  custom = 0,
  p = 1,
  rt = 2,
  n1e = 3,
  bdm = 4,
  n2e = 5,
  cr = 6,
  regge = 7,
  dpc = 8,
  bubble = 9,
  serendipity = 10
};

} // namespace element

} // namespace basix
