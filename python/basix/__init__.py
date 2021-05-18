# Public interface
from ._basixcpp import __version__, __doc__
from ._basixcpp import create_element, CellType, cell_to_str, mapping_to_str, family_to_str, MappingType
from . import cell

# To possibly be removed
from ._basixcpp import (topology, geometry, tabulate_polynomial_set,
                        create_lattice, LatticeType, index,
                        make_quadrature, compute_jacobi_deriv)
