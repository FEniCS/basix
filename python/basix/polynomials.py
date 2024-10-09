# Copyright (C) 2023-2024 Matthew Scroggs and Garth N. Wells
#
# This file is part of Basix (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT
"""Functions for working with polynomials."""

import numpy as np
import numpy.typing as npt

from basix._basixcpp import PolynomialType, PolysetType
from basix._basixcpp import polynomials_dim as _pd
from basix._basixcpp import restriction as _restriction
from basix._basixcpp import superset as _superset
from basix._basixcpp import tabulate_polynomial_set as _tps
from basix._basixcpp import tabulate_polynomials as _tabulate_polynomials
from basix.cell import CellType
from basix.utils import index

__all__ = ["reshape_coefficients", "dim", "tabulate_polynomial_set"]


def reshape_coefficients(
    poly_type: PolynomialType,
    cell_type: CellType,
    coefficients: npt.NDArray[np.float64],
    value_size: int,
    input_degree: int,
    output_degree: int,
) -> npt.NDArray[np.float64]:
    """Reshape the coefficients.

    Args:
        poly_type: The polynomial type.
        cell_type: The cell type
        coefficients: The coefficients
        value_size: The value size of the polynomials associated with
            the coefficients.
        input_degree: The maximum degree of polynomials associated with
            the input coefficients.
        output_degree: The maximum degree of polynomials associated with
            the output coefficients.

    Returns:
        Coefficients representing the same coefficients as the input in
        the set of polynomials of the output degree.

    """
    if poly_type != PolynomialType.legendre:
        raise NotImplementedError()
    if output_degree < input_degree:
        raise ValueError("Output degree must be greater than or equal to input degree")

    if output_degree == input_degree:
        return coefficients

    pdim = dim(poly_type, cell_type, output_degree)
    out = np.zeros((coefficients.shape[0], pdim * value_size))

    indices: list[tuple[int, ...]] = []

    if cell_type == CellType.interval:
        indices = [(i,) for i in range(input_degree + 1)]

        def idx(d, i):
            return index(i[0])

    elif cell_type == CellType.triangle:
        indices = [(i, j) for i in range(input_degree + 1) for j in range(input_degree + 1 - i)]

        def idx(d, i):
            return index(i[1], i[0])

    elif cell_type == CellType.tetrahedron:
        indices = [
            (i, j, k)
            for i in range(input_degree + 1)
            for j in range(input_degree + 1 - i)
            for k in range(input_degree + 1 - i - j)
        ]

        def idx(d, i):
            return index(i[2], i[1], i[0])

    elif cell_type == CellType.quadrilateral:
        indices = [(i, j) for i in range(input_degree + 1) for j in range(input_degree + 1)]

        def idx(d, i):
            return (d + 1) * i[0] + i[1]

    elif cell_type == CellType.hexahedron:
        indices = [
            (i, j, k)
            for i in range(input_degree + 1)
            for j in range(input_degree + 1)
            for k in range(input_degree + 1)
        ]

        def idx(d, i):
            return (d + 1) ** 2 * i[0] + (d + 1) * i[1] + i[2]

    elif cell_type == CellType.pyramid:
        indices = [
            (i, j, k)
            for k in range(input_degree + 1)
            for i in range(input_degree + 1 - k)
            for j in range(input_degree + 1 - k)
        ]

        def idx(d, i):
            rv = d - i[2] + 1
            r0 = i[2] * (d + 1) * (d - i[2] + 2) + (2 * i[2] - 1) * (i[2] - 1) * i[2] // 6
            return r0 + i[0] * rv + i[1]

    elif cell_type == CellType.prism:
        indices = [
            (i, j, k)
            for i in range(input_degree + 1)
            for j in range(input_degree + 1 - i)
            for k in range(input_degree + 1)
        ]

        def idx(d, i):
            return (d + 1) * index(i[1], i[0]) + i[2]

    else:
        raise ValueError("Unsupported cell type")

    pdim_in = dim(poly_type, cell_type, input_degree)
    for v in range(value_size):
        for i in indices:
            out[:, v * pdim + idx(output_degree, i)] = coefficients[
                :, v * pdim_in + idx(input_degree, i)
            ]

    return out


def dim(ptype: PolynomialType, celltype: CellType, degree: int) -> int:
    """Dimension of a polynomial space.

    Args:
        ptype: The polynomial type
        celltype: The cell type
        degree: The polynomial degree

    Returns:
        The dimension of the polynomial space
    """
    return _pd(ptype, celltype, degree)


def tabulate_polynomials(
    ptype: PolynomialType, celltype: CellType, degree: int, pts: npt.NDArray
) -> npt.ArrayLike:
    """Tabulate a set of polynomials on a reference cell.

    Args:
        ptype: The polynomial type
        celltype: The cell type
        degree: The polynomial degree
        pts: The points

    Returns:
        Tabulated polynomials
    """
    return _tabulate_polynomials(ptype, celltype, degree, pts)


def restriction(ptype: PolysetType, cell: CellType, restriction_cell: CellType) -> PolysetType:
    """Get the polyset type that represents the restrictions of a type on a subentity.

    Args:
        ptype: The polynomial type
        cell: The cell type
        restriction_cell: The cell type if the subentity

    Returns:
        The restricted polyset type
    """
    return _restriction(ptype, cell, restriction_cell)


def superset(cell: CellType, type1: PolysetType, type2: PolysetType) -> PolysetType:
    """Get the polyset type that is a superset of two types on the given cell.

    Args:
        cell: The cell type
        type1: The first polyset type
        type2: The second polyset type

    Returns:
        The superset type
    """
    return _superset(cell, type1, type2)


def tabulate_polynomial_set(
    celltype: CellType, ptype: PolysetType, degree: int, nderiv: int, pts: npt.NDArray
) -> npt.ArrayLike:
    """Tabulate a polynomial set.

    Args:
        celltype: The cell type
        ptype: The polyset type
        degree: The polynomial degree
        nderiv: The number of derivatives
        pts: The points to tabulat at

    Returns:
        Tabulated polynomial set
    """
    return _tps(celltype, ptype, degree, nderiv, pts)
