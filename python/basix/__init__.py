# Copyright (C) 2023-2024 Matthew Scroggs and Garth N. Wells
#
# This file is part of Basix (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT
"""Basix is a finite element definition and tabulation library.

The core of the library is written in C++, but the majority of Basix's
functionality can be used via this Python interface.
"""

# Template placeholder for injecting Windows dll directories in CI
# WINDOWSDLL

from importlib.metadata import metadata

from basix import cell, finite_element, lattice, polynomials, quadrature, sobolev_spaces
from basix._basixcpp import MapType
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
from basix.polynomials import PolynomialType, PolysetType, tabulate_polynomials
from basix.polynomials import restriction as polyset_restriction
from basix.polynomials import superset as polyset_superset
from basix.quadrature import QuadratureType, make_quadrature
from basix.sobolev_spaces import SobolevSpace
from basix.utils import index

__version__ = metadata("fenics-basix")["Version"]

__all__ = [
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
    "cell",
    "compute_interpolation_operator",
    "create_custom_element",
    "create_element",
    "create_lattice",
    "create_tp_element",
    "finite_element",
    "geometry",
    "index",
    "lattice",
    "make_quadrature",
    "polynomials",
    "polyset_restriction",
    "polyset_superset",
    "quadrature",
    "sobolev_spaces",
    "tabulate_polynomials",
    "topology",
]
