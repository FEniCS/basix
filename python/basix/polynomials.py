"""Functions for working with polynomials."""

from ._basixcpp import CellType as _CT
from ._basixcpp import PolynomialType as _PT
from ._basixcpp import index as _index
import numpy as _numpy
import numpy.typing as _numpy_typing
import typing as _typing

from ._basixcpp import polynomials_dim as dim  # noqa: F401

_nda_f64 = _numpy_typing.NDArray[_numpy.float64]


def reshape_coefficients(
    poly_type: _PT, cell_type: _CT, coefficients: _nda_f64, value_size: int, input_degree: int, output_degree: int
) -> _nda_f64:
    """Reshape the coefficients.

    Args:
        poly_type: The polynomial type.
        cell_type: The cell type
        coefficients: The coefficients
        value_size: The value size of the polynomials associated with the coefficients
        input_degree: The maximum degree of polynomials associated with the input coefficients
        output_degree: The maximum degree of polynomials associated with the output coefficients

    Returns:
        Coefficients representing the same coefficients as the input in the set of polynomials of the output degree
    """
    if poly_type != _PT.legendre:
        raise NotImplementedError()
    if output_degree < input_degree:
        raise ValueError("Output degree must be greater than or equal to input degree")

    if output_degree == input_degree:
        return coefficients

    pdim = dim(poly_type, cell_type, output_degree)
    out = _numpy.zeros((coefficients.shape[0], pdim * value_size))

    indices: _typing.List[_typing.Tuple[int, ...]] = []

    if cell_type == _CT.interval:
        indices = [(i, ) for i in range(input_degree + 1)]

        def idx(d, i):
            return _index(i[0])

    elif cell_type == _CT.triangle:
        indices = [(i, j) for i in range(input_degree + 1) for j in range(input_degree + 1 - i)]

        def idx(d, i):
            return _index(i[1], i[0])

    elif cell_type == _CT.tetrahedron:
        indices = [(i, j, k) for i in range(input_degree + 1) for j in range(input_degree + 1 - i)
                   for k in range(input_degree + 1 - i - j)]

        def idx(d, i):
            return _index(i[2], i[1], i[0])

    elif cell_type == _CT.quadrilateral:
        indices = [(i, j) for i in range(input_degree + 1) for j in range(input_degree + 1)]

        def idx(d, i):
            return (d + 1) * i[0] + i[1]

    elif cell_type == _CT.hexahedron:
        indices = [(i, j, k) for i in range(input_degree + 1) for j in range(input_degree + 1)
                   for k in range(input_degree + 1)]

        def idx(d, i):
            return (d + 1) ** 2 * i[0] + (d + 1) * i[1] + i[2]

    elif cell_type == _CT.pyramid:
        indices = [(i, j, k) for k in range(input_degree + 1) for i in range(input_degree + 1 - k)
                   for j in range(input_degree + 1 - k)]

        def idx(d, i):
            rv = d - i[2] + 1
            r0 = i[2] * (d + 1) * (d - i[2] + 2) + (2 * i[2] - 1) * (i[2] - 1) * i[2] // 6
            return r0 + i[0] * rv + i[1]

    elif cell_type == _CT.prism:
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
