"""Functions to create lattices and manipulate lattice types."""

from ._basixcpp import LatticeType as _LT


def string_to_type(lattice: str):
    """Convert a string to a Basix LatticeType enum."""
    if lattice == "gll":
        return _LT.gll_warped

    if not hasattr(_LT, lattice):
        raise ValueError(f"Unknown lattice: {lattice}")
    return getattr(_LT, lattice)


def type_to_string(latticetype: _LT):
    """Convert a Basix LatticeType enum to a string."""
    return latticetype.name
