"""Functions to create lattices and manipulate lattice types."""

from ._basixcpp import LatticeType as _LT
from ._basixcpp import LatticeSimplexMethod as _LSM


def string_to_type(lattice: str):
    """Convert a string to a Basix LatticeType enum."""
    if not hasattr(_LT, lattice):
        raise ValueError(f"Unknown lattice: {lattice}")
    return getattr(_LT, lattice)


def type_to_string(latticetype: _LT):
    """Convert a Basix LatticeType enum to a string."""
    return latticetype.name


def string_to_simplex_method(method: str):
    """Convert a string to a Basix LatticeSimplexMethod enum."""
    if not hasattr(_LSM, method):
        raise ValueError(f"Unknown simplex method: {method}")
    return getattr(_LSM, method)


def simplex_method_to_string(simplex_method: _LSM):
    """Convert a Basix LatticeSimplexMethod enum to a string."""
    return simplex_method.name
