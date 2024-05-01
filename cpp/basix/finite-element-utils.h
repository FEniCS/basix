// Copyright (c) 2020-2024 Chris Richardson, Matthew Scroggs and Garth . Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "element-families.h"
#include "finite-element.h"
#include "maps.h"
#include "mdspan.hpp"
#include "polyset.h"
#include "precompute.h"
#include "sobolev-spaces.h"
#include <array>
#include <concepts>
#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

/// Basix: FEniCS runtime basis evaluation library
namespace basix
{

namespace element
{
/// Typedef for mdspan
template <typename T, std::size_t d>
using mdspan_t = impl::mdspan_t<T, d>;

/// Create a version of the interpolation points, interpolation
/// matrices and entity transformation that represent a discontinuous
/// version of the element. This discontinuous version will have the
/// same DOFs but they will all be associated with the interior of the
/// reference cell.
/// @param[in] x Interpolation points. Indices are (tdim, entity index,
/// point index, dim)
/// @param[in] M The interpolation matrices. Indices are (tdim, entity
/// index, dof, vs, point_index, derivative)
/// @param[in] tdim The topological dimension of the cell the element is
/// defined on
/// @param[in] value_size The value size of the element
/// @return (xdata, xshape, Mdata, Mshape), where the x and M data are
/// for  a discontinuous version of the element (with the same shapes as
/// x and M)
template <std::floating_point T>
std::tuple<std::array<std::vector<std::vector<T>>, 4>,
           std::array<std::vector<std::array<std::size_t, 2>>, 4>,
           std::array<std::vector<std::vector<T>>, 4>,
           std::array<std::vector<std::array<std::size_t, 4>>, 4>>
make_discontinuous(const std::array<std::vector<mdspan_t<const T, 2>>, 4>& x,
                   const std::array<std::vector<mdspan_t<const T, 4>>, 4>& M,
                   std::size_t tdim, std::size_t value_size);

} // namespace element

} // namespace basix
