// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "element-families.h"
#include "finite-element.h"
#include <concepts>

namespace basix::element
{
/// Create a serendipity element on cell with given degree
/// @param[in] celltype The cell type
/// @param[in] degree The degree of the element
/// @param[in] lvariant The variant of the Lagrange element to be used for
/// integral moments on the edges of the cell
/// @param[in] dvariant The variant of the DPC element to be used for
/// integral moments on the interior of the cell (for quads and hexes). For
/// elements on an interval element::dpc_variant::unset can be passed in
/// @param[in] discontinuous Controls whether the element is continuous or
/// discontinuous
/// @return A finite element
template <std::floating_point T>
FiniteElement<T> create_serendipity(cell::type celltype, int degree,
                                    element::lagrange_variant lvariant,
                                    element::dpc_variant dvariant,
                                    bool discontinuous);

/// Create a DPC (discontinuous polynomial cubical) element on cell with given
/// degree.
/// @note DPC elements must be discontinuous
/// @param[in] celltype The cell type
/// @param[in] degree The degree of the element
/// @param[in] variant The variant of the element to be created
/// @param[in] discontinuous Controls whether the element is continuous or
/// discontinuous
/// @return A finite element
template <std::floating_point T>
FiniteElement<T> create_dpc(cell::type celltype, int degree,
                            element::dpc_variant variant, bool discontinuous);

/// Create a serendipity H(div) element on cell with given degree
/// @param[in] celltype The cell type
/// @param[in] degree The degree of the element
/// @param[in] lvariant The variant of the Lagrange element to be used for
/// integral moments
/// @param[in] dvariant The variant of the DPC element to be used for
/// integral moments
/// @param[in] discontinuous Controls whether the element is continuous or
/// discontinuous
/// @return A finite element
template <std::floating_point T>
FiniteElement<T> create_serendipity_div(cell::type celltype, int degree,
                                        element::lagrange_variant lvariant,
                                        element::dpc_variant dvariant,
                                        bool discontinuous);

/// Create a serendipity H(curl) element on cell with given degree
/// @param[in] celltype The cell type
/// @param[in] degree The degree of the element
/// @param[in] lvariant The variant of the Lagrange element to be used for
/// integral moments
/// @param[in] dvariant The variant of the DPC element to be used for
/// integral moments
/// @param[in] discontinuous Controls whether the element is continuous or
/// discontinuous
/// @return A finite element
template <std::floating_point T>
FiniteElement<T> create_serendipity_curl(cell::type celltype, int degree,
                                         element::lagrange_variant lvariant,
                                         element::dpc_variant dvariant,
                                         bool discontinuous);
} // namespace basix::element
