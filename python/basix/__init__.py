import os

# Public interface
from ._basixcpp import __version__
from ._basixcpp import (create_element, CellType, MappingType, mapping_to_str,
                        ElementFamily, family_to_str, ElementVariant, variant_to_str)


# To possibly be removed
from ._basixcpp import (topology, geometry, tabulate_polynomial_set,
                        #  create_new_element,
                        create_lattice, LatticeType, index,
                        make_quadrature, compute_jacobi_deriv)

_prefix_dir = os.path.dirname(os.path.abspath(__file__))


def get_include_path():
    return os.path.join(_prefix_dir, "include")


def get_prefix_dir():
    return _prefix_dir
