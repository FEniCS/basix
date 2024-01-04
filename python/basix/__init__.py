"""Basix is a finite element definition and tabulation runtime library.

The core of the library is written in C++, but the majority of Basix's
functionality can be used via this Python interface.
"""
import typing
import numpy as _np
import numpy.typing as npt
from basix import cell, finite_element, lattice, polynomials, quadrature, sobolev_spaces, variants
from basix._basixcpp import (CellType, DPCVariant, ElementFamily, LagrangeVariant, LatticeSimplexMethod, LatticeType,
                             MapType, PolynomialType, PolysetType, QuadratureType, SobolevSpace, __version__,
                             compute_interpolation_operator, create_custom_element, create_lattice,
                             geometry, index)
from basix._basixcpp import create_element as _create_element
from basix._basixcpp import restriction as polyset_restriction
from basix._basixcpp import superset as polyset_superset
from basix._basixcpp import tabulate_polynomials, topology
from basix.quadrature import make_quadrature


def create_element(family_name: ElementFamily, cell_name: CellType, degree: int,
                   lvariant: typing.Optional[LagrangeVariant] = LagrangeVariant.unset,
                   dvariant: typing.Optional[DPCVariant] = DPCVariant.unset,
                   discontinuous: typing.Optional[bool] = False,
                   dof_ordering:  typing.Optional[list[int]] = [],
                   dtype: typing.Optional[npt.DTypeLike] = _np.float64):
    """Create a finite element.

    Args:
        family_name: Finite element family.
        cell_name: Cell shape.
        degree: Polynomial degree.
        lvariant: Lagrange variant type.
        dvariant: DPC variant type/
        discontinuous: If `True`, make element discontinuous
        dof_ordering:
        dtype: Element scalar type.

    Returns:
        A finite element.
    """
    return _create_element(family_name, cell_name, degree, lvariant, dvariant,
                           discontinuous, dof_ordering, _np.dtype(dtype).char)
