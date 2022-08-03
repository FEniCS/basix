"""Basix is a finite element definition and tabulation runtime library.

The core of the library is written in C++, but the majority of Basix's
functionality can be used via this Python interface.
"""

from ._basixcpp import __version__
from . import cell, finite_element, lattice, polynomials, quadrature, variants
from ._basixcpp import (CellType, LatticeType, LatticeSimplexMethod, ElementFamily, LagrangeVariant,
                        DPCVariant, QuadratureType, PolynomialType, MapType)
from ._basixcpp import (create_lattice, create_element, compute_interpolation_operator, topology,
                        geometry, make_quadrature, index, tabulate_polynomials, create_custom_element)
