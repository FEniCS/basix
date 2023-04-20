// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include <array>
#include <concepts>
#include <utility>
#include <vector>

namespace basix
{
template <std::floating_point T>
class FiniteElement;

/// @brief Compute a matrix that represents the interpolation between
/// two elements.
///
/// If the two elements have the same value size, this function returns
/// the interpolation between them.
///
/// If element_from has value size 1 and element_to has value size > 1,
/// then this function returns a matrix to interpolate from a blocked
/// element_from (ie multiple copies of element_from) into element_to.
///
/// If element_to has value size 1 and element_from has value size > 1,
/// then this function returns a matrix that interpolates the components
/// of element_from into copies of element_to.
///
/// @note If the elements have different value sizes and both are
/// greater than 1, this function throws a runtime error
///
/// In order to interpolate functions between finite element spaces on
/// arbitrary cells, the functions must be pulled back to the reference
/// element (this pull back includes applying DOF transformations). The
/// matrix that this function returns can then be applied, then the
/// result pushed forward to the cell. If element_from and element_to
/// have the same map type, then only the DOF transformations need to be
/// applied, as the pull back and push forward cancel each other out.
///
/// @param[in] element_from The element to interpolate from
/// @param[in] element_to The element to interpolate to
/// @return Matrix operator that maps the 'from' degrees-of-freedom to
/// the 'to' degrees-of-freedom. Shape is (ndofs(element_to),
/// ndofs(element_from))
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
compute_interpolation_operator(const FiniteElement<T>& element_from,
                               const FiniteElement<T>& element_to);

} // namespace basix
