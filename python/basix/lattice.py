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

__all__ = ["create_lattice"]


def create_lattice(
    celltype: CellType,
    n: int,
    ltype: LatticeType,
    exterior: bool,
    method: LatticeSimplexMethod = LatticeSimplexMethod.none,
) -> npt.ArrayLike:
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
    return _create_lattice(celltype, n, ltype, exterior, method)
