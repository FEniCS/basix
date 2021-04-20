// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "finite-element.h"

namespace basix
{
/// Create RTC H(div) element
/// @param celltype
/// @param degree
/// @param variant
FiniteElement create_rtc(cell::type celltype, int degree,
                         element::variant variant);

/// Create NC H(curl) element
/// @param celltype
/// @param degree
/// @param variant
FiniteElement create_nce(cell::type celltype, int degree,
                         element::variant variant);

} // namespace basix
