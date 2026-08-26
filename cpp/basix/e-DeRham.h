// Copyright (c) 2024 Johannes M. Vogels
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "element-families.h"
#include "finite-element.h"
#include <concepts>

namespace basix::element
{
/// Create a finite element on cell with given degree
/// @param[in] celltype The cell type
/// @param[in] degree The degree of the element
/// @param[in] form -1,0,1,2,3 : de Rahm complex form
/// @param[in] trimmed : P- instead of P based
/// @param[in] tensor : instead of serendipity
/// @param[in] tiniesttensor : instead of full
/// @param[in] rotated2D : Curl Div instead of Grad Curl complex for 2D
/// @return A finite element
template <std::floating_point T>
FiniteElement<T> create_DeRham_element(cell::type celltype,
                                       int degree,
                                       int form,
                                       bool trimmed,
                                       bool tensor,
                                       bool tiniesttensor,
                                       bool rotated2D
                                       );
