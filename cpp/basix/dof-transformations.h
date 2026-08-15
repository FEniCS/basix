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
/// @brief Functions to compute the DOF transformations (rotations and
/// reflections of entity DOFs) needed by a finite element under a
/// change of entity orientation.
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
/// (number of basis functions of the element, dim(Legendre
/// polynomials))
/// @param[in] degree The embedded superdegree of the element (the
/// lowest degree `n` such that the element's polynomial set is a
/// subspace of a Lagrange element of degree `n`)
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
        std::vector<md::mdspan<const T, md::dextents<std::size_t, 2>>>, 4>& x,
    std::array<std::vector<md::mdspan<const T, md::dextents<std::size_t, 4>>>,
               4>
        M,
    md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs, int degree,
    std::size_t vs, maps::type map_type, polyset::type ptype);

} // namespace basix::doftransforms
