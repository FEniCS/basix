"""
Basix is a finite element definition and tabulation runtime library.

The core of the library is written in C++, but the majority of Basix's
functionality can be used via this Python interface.
"""

# Public interface
from ._basixcpp import __version__
from . import cell, finite_element, lattice, variants, interpolation
from ._basixcpp import (CellType, LatticeType, LatticeSimplexMethod, create_lattice, ElementFamily,
                        create_element, LagrangeVariant)

# To possibly be removed
from ._basixcpp import (topology, geometry, tabulate_polynomial_set,
                        index, make_quadrature, compute_jacobi_deriv)
