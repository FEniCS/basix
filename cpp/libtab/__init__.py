import os

# Public interface
from ._libtabcpp import __version__
from ._libtabcpp import create_element, CellType


# To possibly be removed
from ._libtabcpp import (topology, geometry, tabulate_polynomial_set,
                         create_new_element, create_lattice, LatticeType, index,
                         make_quadrature, compute_jacobi_deriv,
                         gauss_lobatto_legendre_line_rule)

# To be removed
from ._libtabcpp import (Nedelec, NedelecSecondKind, Lagrange,
                         DiscontinuousLagrange, CrouzeixRaviart, RaviartThomas,
                         Regge)

_prefix_dir = os.path.dirname(os.path.abspath(__file__))


def get_include_path():
    return os.path.join(_prefix_dir, "include")


def get_prefix_dir():
    return _prefix_dir
