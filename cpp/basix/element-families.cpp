// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "element-families.h"
#include <stdexcept>

/// Interfaces for creating finite elements
namespace basix
{
//-----------------------------------------------------------------------------
std::string element::to_string(lagrange_variant type)
{
  switch (type)
  {
  case lagrange_variant::unset:
    return "unset";
  case lagrange_variant::equispaced:
    return "equispaced";
  case lagrange_variant::gll_warped:
    return "gll_warped";
  case lagrange_variant::gll_isaac:
    return "gll_isaac";
  case lagrange_variant::gll_centroid:
    return "gll_centroid";
  case lagrange_variant::chebyshev_warped:
    return "chebyshev_warped";
  case lagrange_variant::chebyshev_isaac:
    return "chebyshev_isaac";
  case lagrange_variant::chebyshev_centroid:
    return "chebyshev_centroid";
  case lagrange_variant::gl_warped:
    return "gl_warped";
  case lagrange_variant::gl_isaac:
    return "gl_isaac";
  case lagrange_variant::gl_centroid:
    return "gl_centroid";
  case lagrange_variant::legendre:
    return "legendre";
  case lagrange_variant::bernstein:
    return "bernstein";
  default:
    throw std::runtime_error("Unknown lagrange variant.");
  }
}

//-----------------------------------------------------------------------------
element::lagrange_variant element::to_type(const std::string& variant)
{
  if (variant == "unset")
    return lagrange_variant::unset;
  else if (variant == "equispaced")
    return lagrange_variant::equispaced;
  else if (variant == "gll_warped")
    return lagrange_variant::gll_warped;
  else if (variant == "gll_isaac")
    return lagrange_variant::gll_isaac;
  else if (variant == "gll_centroid")
    return lagrange_variant::gll_centroid;
  else if (variant == "chebyshev_warped")
    return lagrange_variant::chebyshev_warped;
  else if (variant == "chebyshev_isaac")
    return lagrange_variant::chebyshev_isaac;
  else if (variant == "chebyshev_centroid")
    return lagrange_variant::chebyshev_centroid;
  else if (variant == "gl_warped")
    return lagrange_variant::gl_warped;
  else if (variant == "gl_isaac")
    return lagrange_variant::gl_isaac;
  else if (variant == "gl_centroid")
    return lagrange_variant::gl_centroid;
  else if (variant == "legendre")
    return lagrange_variant::legendre;
  else if (variant == "bernstein")
    return lagrange_variant::bernstein;
  else
    throw std::runtime_error("Unknown lagrange variant (" + variant + ")");
}

} // namespace basix::element
