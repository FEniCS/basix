"""Functions to get cell geometry information and manipulate cell types."""

from ._basixcpp import cell_volume as volume
from ._basixcpp import cell_facet_reference_volumes as facet_reference_volumes
from ._basixcpp import cell_facet_normals as facet_normals
from ._basixcpp import cell_facet_outward_normals as facet_outward_normals
from ._basixcpp import cell_facet_orientations as facet_orientations
from ._basixcpp import cell_facet_jacobians as facet_jacobians
from ._basixcpp import sub_entity_connectivity
from ._basixcpp import CellType as _CT


def string_to_type(cell: str) -> _CT:
    """Convert a string to a Basix CellType."""
    if not hasattr(_CT, cell):
        raise ValueError(f"Unknown cell: {cell}")
    return getattr(_CT, cell)


def type_to_string(celltype: _CT) -> str:
    """Convert a Basix CellType to a string."""
    return celltype.name
