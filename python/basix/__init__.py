# Copyright (C) 2023-2024 Matthew Scroggs and Garth N. Wells
#
# This file is part of Basix (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT
"""Basix is a finite element definition and tabulation library.

The core of the library is written in C++, but the majority of Basix's
functionality can be used via this Python interface.
"""

from basix import cell, finite_element, lattice, polynomials, quadrature, sobolev_spaces
from basix._basixcpp import __version__
from basix.cell import CellType, geometry, topology
from basix.finite_element import (
    DPCVariant,
    ElementFamily,
    LagrangeVariant,
    create_custom_element,
    create_element,
    create_tp_element,
)
from basix.interpolation import compute_interpolation_operator
from basix.lattice import LatticeSimplexMethod, LatticeType, create_lattice
from basix.maps import MapType
from basix.polynomials import PolynomialType, PolysetType, tabulate_polynomials
from basix.polynomials import restriction as polyset_restriction
from basix.polynomials import superset as polyset_superset
from basix.quadrature import QuadratureType, make_quadrature
from basix.sobolev_spaces import SobolevSpace
from basix.utils import index

__all__ = [
    "cell",
    "finite_element",
    "lattice",
    "polynomials",
    "quadrature",
    "sobolev_spaces",
    "CellType",
    "DPCVariant",
    "ElementFamily",
    "LagrangeVariant",
    "LatticeSimplexMethod",
    "LatticeType",
    "MapType",
    "PolynomialType",
    "PolysetType",
    "QuadratureType",
    "SobolevSpace",
    "__version__",
    "create_lattice",
    "geometry",
    "index",
    "polyset_restriction",
    "polyset_superset",
    "tabulate_polynomials",
    "topology",
    "create_custom_element",
    "create_element",
    "create_tp_element",
    "make_quadrature",
    "compute_interpolation_operator",
]
