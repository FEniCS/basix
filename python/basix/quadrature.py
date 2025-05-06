# Copyright (C) 2023-2024 Matthew Scroggs and Garth N. Wells
#
# This file is part of Basix (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT
"""Functions to manipulate quadrature types."""

import numpy as _np
import numpy.typing as _npt

from basix._basixcpp import QuadratureType
from basix._basixcpp import make_quadrature as _mq
from basix._basixcpp import gauss_jacobi_rule as _gjr
from basix.cell import CellType
from basix.polynomials import PolysetType

__all__ = ["string_to_type", "make_quadrature"]


def string_to_type(rule: str) -> QuadratureType:
    """Convert a string to a Basix QuadratureType enum.

    Args:
        rule: Qquadrature rule as a string.

    Returns:
        The quadrature type.
    """
    if rule == "default":
        return QuadratureType.default
    elif rule in ["Gauss-Lobatto-Legendre", "GLL"]:
        return QuadratureType.gll
    elif rule in ["Gauss-Legendre", "GL", "Gauss-Jacobi"]:
        return QuadratureType.gauss_jacobi
    elif rule == "Xiao-Gimbutas":
        return QuadratureType.xiao_gimbutas

    return QuadratureType[rule]


def make_quadrature(
    cell: CellType,
    degree: int,
    rule: QuadratureType = QuadratureType.default,
    polyset_type: PolysetType = PolysetType.standard,
) -> tuple[_npt.ArrayLike, _npt.ArrayLike]:
    """Create a quadrature rule.

    Args:
        cell: Cell type.
        degree: Maximum polynomial degree that will be integrated
            exactly.
        rule: Quadrature rule.
        polyset_type: Type of polynomial that will be integrated
            exactly.

    Returns:
        Quadrature points and weights.
    """
    return _mq(rule, cell, polyset_type, degree)


def gauss_jacobi_rule(
    alpha: _np.floating,
    npoints: int,
) -> tuple[_npt.ArrayLike, _npt.ArrayLike]:
    """Create a Gauss-Jacobi quadrature rule for integrating f(x)*(1-x)**alpha
    on the interval [0, 1].

    Args:
        alpha: The exponent alpha
        npoints: Number of points

    Returns:
        Quadrature points and weights.
    """
    return _gjr(_np.float64(alpha), npoints)
