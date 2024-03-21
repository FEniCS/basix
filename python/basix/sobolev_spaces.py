# Copyright (C) 2023-2024 Matthew Scroggs and Garth N. Wells
#
# This file is part of Basix (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT
"""Functions for handling Sobolev spaces."""

from basix._basixcpp import SobolevSpace as _SS
from basix._basixcpp import sobolev_space_intersection as _ssi
from basix.utils import Enum

__all__ = ["intersection", "string_to_sobolev_space"]


class SobolevSpace(Enum):
    """Sobolev space."""

    L2 = _SS.L2
    H1 = _SS.H1
    H2 = _SS.H2
    H3 = _SS.H3
    HInf = _SS.HInf
    HDiv = _SS.HDiv
    HCurl = _SS.HCurl
    HEin = _SS.HEin
    HDivDiv = _SS.HDivDiv


def intersection(spaces: list[SobolevSpace]) -> SobolevSpace:
    """Compute the intersection of a list of Sobolev spaces.

    Args:
        spaces: List of Sobolev spaces.

    Returns:
        Intersection of the Sobolev spaces.
    """
    space = spaces[0].value
    for s in spaces[1:]:
        space = _ssi(space, s.value)
    return getattr(SobolevSpace, space.name)


def string_to_sobolev_space(space: str) -> SobolevSpace:
    """Convert a string to a Basix SobolevSpace.

    Args:
        space: Name of the space.

    Returns:
        Cell type.
    """
    if not hasattr(SobolevSpace, space):
        raise ValueError(f"Unknown Sobolev space: {space}")
    return getattr(SobolevSpace, space)
