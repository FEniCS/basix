"""Functions to get cell geometry information and manipulate cell types."""

from ._basixcpp import cell_volume as volume  # noqa: F401
from ._basixcpp import cell_facet_reference_volumes as facet_reference_volumes  # noqa: F401
from ._basixcpp import cell_facet_normals as facet_normals  # noqa: F401
from ._basixcpp import cell_facet_outward_normals as facet_outward_normals  # noqa: F401
from ._basixcpp import cell_facet_orientations as facet_orientations  # noqa: F401
from ._basixcpp import cell_facet_jacobians as facet_jacobians  # noqa: F401
from ._basixcpp import sub_entity_connectivity  # noqa: F401
from ._basixcpp import CellType as _CT


def string_to_type(cell: str) -> _CT:
    """Convert a string to a Basix CellType.

    Args:
        cell: The name of the cell as a string.

    Returns:
        The cell type
    """
    if not hasattr(_CT, cell):
        raise ValueError(f"Unknown cell: {cell}")
    return getattr(_CT, cell)


def type_to_string(celltype: _CT) -> str:
    """Convert a Basix CellType to a string.

    Args:
        celltype: The cell type.

    Returns:
        The name of the cell as a string.
    """
    return celltype.name
