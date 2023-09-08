"""Basix is a finite element definition and tabulation runtime library.

The core of the library is written in C++, but the majority of Basix's
functionality can be used via this Python interface.
"""

from . import cell, finite_element, lattice, polynomials, quadrature, sobolev_spaces, variants
from ._basixcpp import (CellType, DPCVariant, ElementFamily, LagrangeVariant, LatticeSimplexMethod, LatticeType,
                        MapType, PolynomialType, PolysetType, QuadratureType, SobolevSpace, __version__,
                        compute_interpolation_operator, create_custom_element, create_element, create_lattice, geometry,
                        index)
from ._basixcpp import restriction as polyset_restriction
from ._basixcpp import superset as polyset_superset
from ._basixcpp import tabulate_polynomials, topology
from .quadrature import make_quadrature
