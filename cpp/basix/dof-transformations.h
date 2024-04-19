// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "maps.h"
#include "mdspan.hpp"
#include "polyset.h"
#include <array>
#include <concepts>
#include <map>
#include <utility>
#include <vector>

///
/// @brief Functions to transform DOFs in high degree Lagrange spaces.
/// The functions in this namespace calculate the permutations that can
/// be used to rotate and reflect DOF points in Lagrange spaces.
namespace basix::doftransforms
{
/// @brief Compute the entity DOF transformations for an element.
///
/// @param[in] cell_type The cell type
/// @param[in] x Interpolation points for the element. Indices are
/// (tdim, entity index, point index, dim)
/// @param[in] M Interpolation matrix for the element. Indices are
/// (tdim, entity index, dof, vs, point_index, derivative)
/// @param[in] coeffs The coefficients that define the basis functions
/// of the element in terms of the orthonormal basis. Shape is
/// (dim(Legendre polynomials), dim(finite element polyset))
/// @param[in] degree The degree of the element
/// @param[in] vs The value size of the element
/// @param[in] map_type The map type used by the element
/// @param[in] ptype The polyset type used by the element
/// @return Entity transformations. For each cell, the shape is
/// (ntransformation, ndofs, ndofs)
template <std::floating_point T>
std::map<cell::type, std::pair<std::vector<T>, std::array<std::size_t, 3>>>
compute_entity_transformations(
    cell::type cell_type,
    const std::array<
        std::vector<MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
            const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>>,
        4>& x,
    const std::array<
        std::vector<MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
            const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>>,
        4>& M,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        coeffs,
    int degree, std::size_t vs, maps::type map_type, polyset::type ptype);

} // namespace basix::doftransforms
