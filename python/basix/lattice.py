"""Functions to manipulate lattice types."""

import numpy.typing as npt

from basix._basixcpp import LatticeSimplexMethod as _LSM
from basix._basixcpp import LatticeType as _LT
from basix._basixcpp import create_lattice as _create_lattice
from basix.cell import CellType
from basix.enums import Enum

__all__ = ["string_to_type", "string_to_simplex_method"]


class LatticeType(Enum):
    """Lattice type."""
    equispaced = _LT.equispaced
    gll = _LT.gll
    chebyshev = _LT.chebyshev
    gl = _LT.gl


class LatticeSimplexMethod(Enum):
    """Lattice simplex method."""
    none = _LSM.none
    warp = _LSM.warp
    isaac = _LSM.isaac
    centroid = _LSM.centroid


def string_to_type(lattice: str) -> LatticeType:
    """Convert a string to a Basix LatticeType enum.

    Args:
        lattice: Lattice type as a string.

    Returns:
        The lattice type.

    """
    if not hasattr(LatticeType, lattice):
        raise ValueError(f"Unknown lattice: {lattice}")
    return getattr(LatticeType, lattice)


def string_to_simplex_method(method: str) -> LatticeSimplexMethod:
    """Convert a string to a Basix LatticeSimplexMethod enum.

    Args:
        method: The simplex method as a string.

    Returns:
        The simplex method.

    """
    if not hasattr(LatticeSimplexMethod, method):
        raise ValueError(f"Unknown simplex method: {method}")
    return getattr(LatticeSimplexMethod, method)


def create_lattice(
    celltype: CellType, n: int, ltype: LatticeType, exterior: bool,
    method: LatticeSimplexMethod = LatticeSimplexMethod.none
) -> npt.NDArray:
    """TODO."""
    return _create_lattice(celltype.value, n, ltype.value, exterior, method.value)
