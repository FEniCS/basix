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

/// Get the intersection of two Sobolev spaces
/// @param[in] space1 The first space
/// @param[in] space2 The second space
/// @return The intersection
basix::sobolev::space space_intersection(basix::sobolev::space space1,
                                         basix::sobolev::space space2);

} // namespace basix::sobolev
