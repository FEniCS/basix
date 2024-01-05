"""Functions to get cell geometry information and manipulate cell types."""

from enum import Enum

from basix._basixcpp import CellType as _CT
from basix._basixcpp import cell_facet_jacobians as facet_jacobians  # noqa: F401
from basix._basixcpp import cell_facet_normals as facet_normals  # noqa: F401
from basix._basixcpp import cell_facet_orientations as facet_orientations  # noqa: F401
from basix._basixcpp import cell_facet_outward_normals as facet_outward_normals  # noqa: F401
from basix._basixcpp import cell_facet_reference_volumes as facet_reference_volumes  # noqa: F401
from basix._basixcpp import cell_volume as volume  # noqa: F401
from basix._basixcpp import sub_entity_connectivity  # noqa: F401

__all__ = ["string_to_type", "type_to_string"]


class CellType(Enum):
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
