"""Functions to manipulate lattice types."""

from ._basixcpp import LatticeType as _LT
from ._basixcpp import LatticeSimplexMethod as _LSM


def string_to_type(lattice: str) -> _LT:
    """
    Convert a string to a Basix LatticeType enum.

    Parameters
    ----------
    lattice : str
        The lattice type as a string.

    Returns
    -------
    basix.LatticeType
        The lattice type
    """
    if not hasattr(_LT, lattice):
        raise ValueError(f"Unknown lattice: {lattice}")
    return getattr(_LT, lattice)


def type_to_string(latticetype: _LT) -> str:
    """
    Convert a Basix LatticeType enum to a string.

    Parameters
    ----------
    latticetype : basix.LatticeType
        The lattice type

    Returns
    -------
    str
        The lattice type as a string.
    """
    return latticetype.name


def string_to_simplex_method(method: str) -> _LSM:
    """
    Convert a string to a Basix LatticeSimplexMethod enum.

    Parameters
    ----------
    method : str
        The simplex method as a string.

    Returns
    -------
    basix.LatticeSimplexMethod
        The simplex method
    """
    if not hasattr(_LSM, method):
        raise ValueError(f"Unknown simplex method: {method}")
    return getattr(_LSM, method)


def simplex_method_to_string(simplex_method: _LSM) -> str:
    """
    Convert a Basix LatticeSimplexMethod enum to a string.

    Parameters
    ----------
    simplex_method : basix.LatticeSimplexMethod
        The simplex method

    Returns
    -------
    str
        The simplex method as a string.
    """
    return simplex_method.name
