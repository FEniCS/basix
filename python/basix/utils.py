# Copyright (C) 2023-2024 Matthew Scroggs and Garth N. Wells
#
# This file is part of Basix (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT
"""Utility funcitons."""

import typing
from enum import Enum as _Enum

from basix._basixcpp import index as _index

__all__ = ["Enum"]


class Enum(_Enum):
    """An enum with comparisons implemented."""

    def __lt__(self, other) -> bool:
        """Less than."""
        if self.__class__ != other.__class__:
            return NotImplemented
        else:
            return self.value < other.value

    def __le__(self, other) -> bool:
        """Less than or equal."""
        if self.__class__ != other.__class__:
            return NotImplemented
        else:
            return self.value <= other.value

    def __gt__(self, other) -> bool:
        """Greater than."""
        if self.__class__ != other.__class__:
            return NotImplemented
        else:
            return self.value > other.value

    def __ge__(self, other) -> bool:
        """Greater than or equal."""
        if self.__class__ != other.__class__:
            return NotImplemented
        else:
            return self.value >= other.value

    def __int__(self) -> int:
        """Convert to int."""
        return int(self.value)


def index(p: int, q: typing.Optional[int] = None, r: typing.Optional[int] = None) -> int:
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
