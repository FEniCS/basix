"""
Basix is a finite element definition and tabulation runtime library.

The core of the library is written in C++, but the majority of Basix's
functionality can be used via this Python interface.
"""

# Public interface
from ._basixcpp import __version__
from ._basixcpp import create_element, CellType, cell_to_str, mapping_to_str, family_to_str, MappingType
from . import cell

# To possibly be removed
from ._basixcpp import (topology, geometry, tabulate_polynomial_set,
                        create_lattice, LatticeType, index,
                        make_quadrature, compute_jacobi_deriv, ElementFamily)
