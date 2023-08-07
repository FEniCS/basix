"""Functions to manipulate quadrature types."""

import typing as _typing
import numpy as _np
import numpy.typing as _npt

from ._basixcpp import QuadratureType as _QT
from ._basixcpp import CellType as _CT
from ._basixcpp import PolysetType as _PT
from ._basixcpp import make_quadrature as _mq


def string_to_type(rule: str) -> _QT:
    """Convert a string to a Basix QuadratureType enum.

    Args:
        rule: The quadrature rule as a string.

    Returns:
        The quadrature type
    """
    if rule == "default":
        return _QT.Default

    if rule in ["Gauss-Lobatto-Legendre", "GLL"]:
        return _QT.gll
    if rule in ["Gauss-Legendre", "GL", "Gauss-Jacobi"]:
        return _QT.gauss_jacobi
    if rule == "Xiao-Gimbutas":
        return _QT.xiao_gimbutas

    if not hasattr(_QT, rule):
        raise ValueError(f"Unknown quadrature rule: {rule}")
    return getattr(_QT, rule)


def type_to_string(quadraturetype: _QT) -> str:
    """Convert a Basix QuadratureType enum to a string.

    Args:
        quadraturetype: The quadrature type

    Returns:
        The quadrature rule as a string.
    """
    return quadraturetype.name


def make_quadrature(
    cell: _CT, degree: int, rule: _QT = _QT.Default, polyset_type: _PT = _PT.standard
) -> _typing.Tuple[_npt.NDArray[_np.float64],_npt.NDArray[_np.float64]]:
    """Create a quadrature rule.

    Args:
        cell: The cell type
        degree: The maximum polynomial degree that will be integrated exactly
        rule: The quadrature rule
        polyset_type: The type of polynomial that will be integrated exactly

    Returns:
        The quadrature points and weights
    """
    return make_quadrature(rule, cell, polyset_type, degree)
