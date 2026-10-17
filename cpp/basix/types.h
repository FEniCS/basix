// Copyright (C) 2025 Paul T. Kühner
//
// This file is part of Basix (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "mdspan.hpp"

namespace basix
{
/// @private mdspan namespace
namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

/// @private mdarray namespace
namespace mdex = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;
} // namespace basix
