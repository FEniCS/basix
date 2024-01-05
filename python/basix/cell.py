"""Functions to get cell geometry information and manipulate cell types."""

from enum import Enum
import typing
import numpy.typing as npy

from basix._basixcpp import CellType as _CT
from basix._basixcpp import cell_facet_jacobians as _cfj
from basix._basixcpp import cell_facet_normals as _cfn
from basix._basixcpp import cell_facet_orientations as _fo
from basix._basixcpp import cell_facet_outward_normals as _fon
from basix._basixcpp import cell_facet_reference_volumes as _frv
from basix._basixcpp import cell_volume as _v
from basix._basixcpp import sub_entity_connectivity as _sec

__all__ = ["string_to_type", "type_to_string", "sub_entity_connectivity", "volume",
           "facet_jacobians", "facet_normals", "facet_orientations", "facet_outward_normals",
           "facet_reference_volumes"]


class CellType(Enum):
    """TODO."""
    point = _CT.point
    interval = _CT.interval
    triangle = _CT.triangle
    tetrahedron = _CT.tetrahedron
    quadrilateral = _CT.quadrilateral
    hexahedron = _CT.hexahedron
    prism = _CT.prism
    pyramid = _CT.pyramid


def string_to_type(cell: str) -> CellType:
    """Convert a string to a Basix CellType.

    Args:
        cell: Name of the cell as a string.

    Returns:
        The cell type.
    """
    if not hasattr(CellType, cell):
        raise ValueError(f"Unknown cell: {cell}")
    return getattr(CellType, cell)


def type_to_string(celltype: CellType) -> str:
    """Convert a Basix CellType to a string.

    Args:
        celltype: Cell type.

    Returns:
        The name of the cell as a string.
    """
    return celltype.name


def sub_entity_connectivity(celltype: CellType) -> typing.List[typing.List[typing.List[int]]]:
    """TODO."""
    return _sec(celltype.value)


def volume(celltype: CellType) -> float:
    """TODO."""
    return _v(celltype.value)


def facet_jacobians(celltype: CellType) -> npt.NDArray:
    """TODO."""
    return _fj(celltype.value)


def facet_normals(celltype: CellType) -> npt.NDArray:
    """TODO."""
    return _fn(celltype.value)


def facet_orientations(celltype: CellType) -> typing.List[int]:
    """TODO."""
    return _fo(celltype.value)


def facet_outward_normals(celltype: CellType) -> npt.NDArray:
    """TODO."""
    return _fon(celltype.value)


def facet_reference_volumes(celltype: CellType) -> npt.NDArray:
    """TODO."""
    return _frv(celltype.value)
