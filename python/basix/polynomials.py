"""Functions for working with polynomials."""

import typing as _typing

import numpy as _np
import numpy.typing as npt

from basix._basixcpp import PolynomialType as _PT
from basix._basixcpp import PolysetType as _PST
from basix._basixcpp import restriction as _restriction
from basix._basixcpp import superset as _superset
from basix._basixcpp import index as _index
from basix._basixcpp import tabulate_polynomial_set as _tps
from basix._basixcpp import polynomials_dim as _pd
from basix.cell import CellType
from basix.enums import Enum
from basix._basixcpp import tabulate_polynomials as _tabulate_polynomials

__all__ = ["reshape_coefficients", "dim"]


class PolynomialType(Enum):
    """Polynomail type."""
    legendre = _PT.legendre
    bernstein = _PT.bernstein


class PolysetType(Enum):
    """Polyset type."""
    standard = _PST.standard
    macroedge = _PST.macroedge


def reshape_coefficients(poly_type: PolynomialType, cell_type: CellType, coefficients: npt.NDArray[_np.float64],
                         value_size: int, input_degree: int, output_degree: int) -> npt.NDArray[_np.float64]:
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
    out = _np.zeros((coefficients.shape[0], pdim * value_size))

    indices: _typing.List[_typing.Tuple[int, ...]] = []

    if cell_type == CellType.interval:
        indices = [(i, ) for i in range(input_degree + 1)]

        def idx(d, i):
            return _index(i[0])

    elif cell_type == CellType.triangle:
        indices = [(i, j) for i in range(input_degree + 1) for j in range(input_degree + 1 - i)]

        def idx(d, i):
            return _index(i[1], i[0])

    elif cell_type == CellType.tetrahedron:
        indices = [(i, j, k) for i in range(input_degree + 1) for j in range(input_degree + 1 - i)
                   for k in range(input_degree + 1 - i - j)]

        def idx(d, i):
            return _index(i[2], i[1], i[0])

    elif cell_type == CellType.quadrilateral:
        indices = [(i, j) for i in range(input_degree + 1) for j in range(input_degree + 1)]

        def idx(d, i):
            return (d + 1) * i[0] + i[1]

    elif cell_type == CellType.hexahedron:
        indices = [(i, j, k) for i in range(input_degree + 1) for j in range(input_degree + 1)
                   for k in range(input_degree + 1)]

        def idx(d, i):
            return (d + 1) ** 2 * i[0] + (d + 1) * i[1] + i[2]

    elif cell_type == CellType.pyramid:
        indices = [(i, j, k) for k in range(input_degree + 1) for i in range(input_degree + 1 - k)
                   for j in range(input_degree + 1 - k)]

        def idx(d, i):
            rv = d - i[2] + 1
            r0 = i[2] * (d + 1) * (d - i[2] + 2) + (2 * i[2] - 1) * (i[2] - 1) * i[2] // 6
            return r0 + i[0] * rv + i[1]

    elif cell_type == CellType.prism:
        indices = [(i, j, k) for i in range(input_degree + 1) for j in range(input_degree + 1 - i)
                   for k in range(input_degree + 1)]

        def idx(d, i):
            return (d + 1) * _index(i[1], i[0]) + i[2]

    else:
        raise ValueError("Unsupported cell type")

    pdim_in = dim(poly_type, cell_type, input_degree)
    for v in range(value_size):
        for i in indices:
            out[:, v * pdim + idx(output_degree, i)] = coefficients[:, v * pdim_in + idx(input_degree, i)]

    return out


def dim(ptype: PolynomialType, celltype: CellType, degree: int) -> int:
    return _pd(ptype.value, celltype.value, degree)


def tabulate_polynomials(
    ptype: PolynomialType, celltype: CellType, d: int, x: npt.NDArray
) -> npt.NDArray:
    """TODO."""
    return _tabulate_polynomials(ptype.value, celltype.value, d, x)


def restriction(
    ptype: PolysetType, cell: CellType, restriction_cell: CellType
) -> PolysetType:
    """TODO."""
    return _restriction(ptype.value, cell.value, restriction_cell.value)


def superset(cell: CellType, type1: PolysetType, type2: PolysetType) -> PolysetType:
    """TODO."""
    return _superset(cell.value, type1.value, type2.value)


def string_to_polyset_type(pname: str) -> PolysetType:
    """Convert a string to a Basix PolysetType.

    Args:
        pname: Name of the polyset type as a string.

    Returns:
        The polyset type.
    """
    if not hasattr(PolysetType, pname):
        raise ValueError(f"Unknown polyset: {pname}")
    return getattr(PolysetType, pname)


def tabulate_polynomial_set(
    celltype: CellType, ptype: PolysetType, n: int, nderiv: int, pts: npt.NDArray
) -> npt.NDArray:
    return _tps(celltype.value, ptype.value, n, nderiv, pts)

