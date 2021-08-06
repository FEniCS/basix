"""Functions to create lattices and manipulate lattice types."""

from ._basixcpp import LatticeType


def string_to_type(lattice: str):
    """Convert a string to a Basix LatticeType enum."""
    if not hasattr(LatticeType, lattice):
        raise ValueError(f"Unknown lattice: {lattice}")
    return getattr(LatticeType, lattice)


def type_to_string(latticetype: LatticeType):
    """Convert a Basix LatticeType enum to a string."""
    return latticetype.name
