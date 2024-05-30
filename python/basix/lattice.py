# Copyright (C) 2023-2024 Matthew Scroggs and Garth N. Wells
#
# This file is part of Basix (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT
"""Functions to manipulate lattice types."""

import numpy.typing as npt

from basix._basixcpp import LatticeSimplexMethod, LatticeType
from basix._basixcpp import create_lattice as _create_lattice
from basix.cell import CellType

__all__ = ["string_to_type", "string_to_simplex_method"]


def string_to_type(lattice: str) -> LatticeType:
    """Convert a string to a Basix LatticeType enum.

    Args:
        lattice: Lattice type as a string.

    Returns:
        Lattice type.
    """
    return LatticeType[lattice]


def string_to_simplex_method(method: str) -> LatticeSimplexMethod:
    """Convert a string to a Basix LatticeSimplexMethod enum.

    Args:
        method: Simplex method as a string.

    Returns:
        Simplex method.
    """
    return LatticeSimplexMethod[method]


def create_lattice(
    celltype: CellType,
    n: int,
    ltype: LatticeType,
    exterior: bool,
    method: LatticeSimplexMethod = LatticeSimplexMethod.none,
) -> npt.NDArray:
    """Create a lattice of points on a reference cell.

    Args:
        celltype: Cell type.
        n: The size in each direction. There will be ``n+1`` points
            along each edge of the cell.
        ltype: Lattice type.
        exterior: If ``True``, the points on the edges will be included.
        method: The simplex method used to generate points on simplices.

    Returns:
        Lattice points
    """
    return _create_lattice(celltype, n, ltype.value, exterior, method.value)
