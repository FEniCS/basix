# Copyright (C) 2023-2024 Matthew Scroggs and Garth N. Wells
#
# This file is part of Basix (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT
"""Interpolation."""

import numpy.typing as npt

from basix._basixcpp import compute_interpolation_operator as _compute_interpolation_operator
from basix.finite_element import FiniteElement


def compute_interpolation_operator(
    e0: FiniteElement,
    e1: FiniteElement,
) -> npt.NDArray:
    """Compute a matrix that represents the interpolation between two elements.

    If the two elements have the same value size, this function returns
    the interpolation between them.

    If element_from has value size 1 and element_to has value size > 1,
    then this function returns a matrix to interpolate from a blocked
    element_from (ie multiple copies of element_from) into element_to.

    If element_to has value size 1 and element_from has value size > 1,
    then this function returns a matrix that interpolates the components
    of element_from into copies of element_to.

    Note:
        If the elements have different value sizes and both are
        greater than 1, this function throws a runtime error

    In order to interpolate functions between finite element spaces on
    arbitrary cells, the functions must be pulled back to the reference
    element (this pull back includes applying DOF transformations). The
    matrix that this function returns can then be applied, then the
    result pushed forward to the cell. If element_from and element_to
    have the same map type, then only the DOF transformations need to be
    applied, as the pull back and push forward cancel each other out.

    Args:
        e0: The element to interpolate from
        e1: The element to interpolate to

    Returns:
        Matrix operator that maps the 'from' degrees-of-freedom to
        the 'to' degrees-of-freedom. Shape is (ndofs(element_to),
        ndofs(element_from))
    """
    return _compute_interpolation_operator(e0._e, e1._e)
