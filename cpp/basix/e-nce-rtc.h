// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "finite-element.h"

namespace basix::element
{
/// Create RTC H(div) element
/// @param[in] celltype The cell type
/// @param[in] degree The degree of the element
/// @param[in] lvariant The Lagrange variant to use when defining the element to
/// take integral moments against
/// @param[in] discontinuous Controls whether the element is continuous or
/// discontinuous
/// @return A finite element
FiniteElement create_rtc(cell::type celltype, int degree,
                         element::lagrange_variant lvariant,
                         bool discontinuous);

/// Create NC H(curl) element
/// @param[in] celltype The cell type
/// @param[in] degree The degree of the element
/// @param[in] lvariant The Lagrange variant to use when defining the element to
/// take integral moments against
/// @param[in] discontinuous Controls whether the element is continuous or
/// discontinuous
/// @return A finite element
FiniteElement create_nce(cell::type celltype, int degree,
                         element::lagrange_variant lvariant,
                         bool discontinuous);

} // namespace basix::element
