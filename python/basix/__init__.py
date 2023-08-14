"""Basix is a finite element definition and tabulation runtime library.

The core of the library is written in C++, but the majority of Basix's
functionality can be used via this Python interface.
"""

from ._basixcpp import __version__
from . import cell, finite_element, lattice, polynomials, quadrature, sobolev_spaces, variants
from ._basixcpp import (CellType, LatticeType, LatticeSimplexMethod, ElementFamily, LagrangeVariant,
                        DPCVariant, QuadratureType, PolynomialType, MapType, SobolevSpace,
                        PolysetType)
from ._basixcpp import (create_lattice, create_element, compute_interpolation_operator, topology,
                        geometry, index, tabulate_polynomials, create_custom_element)

from ._basixcpp import superset as polyset_superset
from ._basixcpp import restriction as polyset_restriction
from .quadrature import make_quadrature
