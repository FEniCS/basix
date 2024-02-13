# Copyright (C) 2023-2024 Matthew Scroggs and Garth N. Wells
#
# This file is part of Basix (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT
"""Maps."""

from basix._basixcpp import MapType as _MT
from basix.utils import Enum

__all__ = ["string_to_type"]


class MapType(Enum):
    """Map type."""

    identity = _MT.identity
    L2Piola = _MT.L2Piola
    covariantPiola = _MT.covariantPiola
    contravariantPiola = _MT.contravariantPiola
    doubleCovariantPiola = _MT.doubleCovariantPiola
    doubleContravariantPiola = _MT.doubleContravariantPiola


def string_to_type(mapname: str) -> MapType:
    """Convert a string to a Basix MapType.

    Args:
        mapname: Name of the map as a string.

    Returns:
        The map type.
    """
    if not hasattr(MapType, mapname):
        raise ValueError(f"Unknown map: {mapname}")
    return getattr(MapType, mapname)
