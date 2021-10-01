// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "finite-element.h"
#include <xtensor/xtensor.hpp>

namespace basix
{

/// Computes a matrix that represents the interpolation between two elements.
///
/// If the two elements have the same value size, this function returns the
/// interpolation between them. If one of the elements has value size 1, and the
/// other has value size > 1, this function returns the interpolation matrix
/// that maps the components of the element with a value size to/from the other
/// element. If both element have different value sizes greater than 1, then
/// this function throws a runtime error.
///
/// @param[in] element_from The element to interpolate from
/// @param[in] element_to The element to interpolate to
xt::xtensor<double, 2>
compute_interpolation_between_elements(const FiniteElement& element_from,
                                       const FiniteElement& element_to);

} // namespace basix
