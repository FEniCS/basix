# Copyright (C) 2023-2024 Matthew Scroggs and Garth N. Wells
#
# This file is part of Basix (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT
"""Maps."""

from basix._basixcpp import MapType

__all__ = ["string_to_type"]


def string_to_type(mapname: str) -> MapType:
    """Convert a string to a Basix MapType.

    Args:
        mapname: Name of the map as a string.

    Returns:
        The map type.
    """
    return MapType[mapname]
