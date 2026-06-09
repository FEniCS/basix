# Copyright (C) 2023-2024 Matthew Scroggs and Garth N. Wells
#
# This file is part of Basix (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT
"""Utility funcitons."""

from basix._basixcpp import index as _index


def index(p: int, q: int | None = None, r: int | None = None) -> int:
    """Compute the indexing in a 1D, 2D or 3D simplex.

    Args:
        p: Index in x.
        q: Index in y.
        r: Index in z.

    Returns:
        Index in a flattened 1D array.
    """
    if q is None:
        assert r is None
        return _index(p)
    elif r is None:
        return _index(p, q)
    else:
        return _index(p, q, r)
