// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "finite-element.h"

namespace basix
{

/// Computes a matrix that represents the interpolation between two elements.
/// @param[in] element_from The element to interpolatio from
/// @param[in] element_to The element to interpolatio to
xt::xtensor<double, 2>
compute_interpolation_between_elements(const FiniteElement element_from,
                                       const FiniteElement element_to);

} // namespace basix
