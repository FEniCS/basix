// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include <xtensor/xtensor.hpp>

namespace basix
{
class FiniteElement;

/// Computes a matrix that represents the interpolation between two
/// elements.
///
/// If the two elements have the same value size, this function returns
/// the interpolation between them.
///
/// If element_from has value size 1 and element_to has value size > 1, then
/// this function returns a matrix to interpolate from a blocked element_from
/// (ie multiple copies of element_from) into element_to.
///
/// If element_to has value size 1 and element_from has value size > 1, then
/// this function returns a matrix that interpolates the components of
/// element_from into copies of element_to.
///
/// @note If the elements have different value sizes and both are
/// greater than 1, this function throws a runtime error
///
/// @param[in] element_from The element to interpolate from
/// @param[in] element_to The element to interpolate to
/// @return Matrix operator that maps the 'from' degrees-of-freedom to
/// the 'to' degrees-of-freedom
xt::xtensor<double, 2>
compute_interpolation_operator(const FiniteElement& element_from,
                               const FiniteElement& element_to);

} // namespace basix
