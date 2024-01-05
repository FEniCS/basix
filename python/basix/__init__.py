"""Basix is a finite element definition and tabulation runtime library.

The core of the library is written in C++, but the majority of Basix's
functionality can be used via this Python interface.
"""
import typing

from basix import cell, finite_element, lattice, polynomials, quadrature, sobolev_spaces, variants
from basix._basixcpp import (CellType, DPCVariant, ElementFamily, LagrangeVariant, LatticeSimplexMethod, LatticeType,
                             MapType, PolynomialType, PolysetType, QuadratureType, SobolevSpace, __version__)
from basix._basixcpp import compute_interpolation_operator as _compute_interpolation_operator
from basix._basixcpp import create_lattice as _create_lattice
from basix._basixcpp import geometry as _geometry
from basix._basixcpp import index as _index
from basix._basixcpp import restriction as _restriction
from basix._basixcpp import superset as _superset
from basix._basixcpp import tabulate_polynomials as _tabulate_polynomials
from basix._basixcpp import topology as _topology
from basix.finite_element import create_custom_element, create_element
from basix.quadrature import make_quadrature
import numpy.typing as npt

__all__ = ["cell", "finite_element", "lattice", "polynomials", "quadrature", "sobolev_spaces", "variants",
           "CellType", "DPCVariant", "ElementFamily", "LagrangeVariant", "LatticeSimplexMethod", "LatticeType",
           "MapType", "PolynomialType", "PolysetType", "QuadratureType", "SobolevSpace", "__version__",
           "create_lattice", "geometry", "index", "polyset_restriction", "polyset_superset",
           "tabulate_polynomials", "topology", "create_custom_element", "create_element",
           "make_quadrature", "compute_interpolation_operator"]


def compute_interpolation_operator(
    e0: finite_element.FiniteElement, e1: finite_element.FiniteElement,
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


def create_lattice(
    celltype: CellType, n: int, ltype: LatticeType, exterior: bool,
    method: LatticeSimplexMethod = LatticeSimplexMethod.none
) -> npt.NDArray:
    """TODO."""
    return _create_lattice(celltype, n, ltype, exterior, method)


def polyset_restriction(
    ptype: PolysetType, cell: CellType, restriction_cell: CellType
) -> PolysetType:
    """TODO."""
    return _restriction(ptype, cell, restriction_cell)


def polyset_superset(cell: CellType, type1: PolysetType, type2: PolysetType) -> PolysetType:
    """TODO."""
    return _superset(cell, type1, type2)


def geometry(celltype: CellType) -> npt.NDArray:
    """TODO."""
    return _geometry(celltype)


def topology(celltype: CellType) -> typing.List[typing.List[typing.List[int]]]:
    """TODO."""
    return _topology(celltype)


def index(p: int, q: typing.Optional[int] = None, r: typing.Optional[int] = None) -> int:
    """TODO."""
    if q is None:
        assert r is None
        return _index(p)
    elif r is None:
        return _index(p, q)
    else:
        return _index(p, q, r)


def tabulate_polynomials(
    ptype: PolynomialType, celltype: CellType, d: int
) -> npt.NDArray:
    """TODO."""
    return _tabulate_polynomials(ptype, celltype, d)
