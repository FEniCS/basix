// Copyright (c) 2022 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

/// Information about Sobolev spaces
namespace basix::sobolev
{

/// Sobolev space type
enum class space
{
  L2 = 0,
  H1 = 1,
  H2 = 2,
  H3 = 3,
  HInf = 8,
  HDiv = 10,
  HCurl = 11,
  HEin = 12,
  HDivDiv = 13,
};

/// Get the intersection of two Sobolev spaces.
/// @param[in] space1 First space
/// @param[in] space2 Second space
/// @return Intersection of the spaces
space space_intersection(space space1, space space2);

} // namespace basix::sobolev
