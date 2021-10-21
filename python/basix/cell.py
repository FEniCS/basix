"""Functions to get cell geometry information and manipulate cell types."""

from . import _basixcpp
import numpy as _np


def string_to_type(cell: str) -> _basixcpp.CellType:
    """Convert a string to a Basix CellType."""
    if not hasattr(_basixcpp.CellType, cell):
        raise ValueError(f"Unknown cell: {cell}")
    return getattr(_basixcpp.CellType, cell)


def type_to_string(celltype: _basixcpp.CellType) -> str:
    """Convert a Basix CellType to a string."""
    return celltype.name


def volume(celltype: _basixcpp.CellType) -> float:
    """Get the volume of a reference cell."""
    return _basixcpp.cell_volume(celltype)


def facet_reference_volumes(celltype: _basixcpp.CellType) -> _np.array:
    """Get the reference volumes of the facets of a reference cell."""
    return _basixcpp.cell_facet_reference_volumes(celltype)


def facet_normals(celltype: _basixcpp.CellType) -> _np.ndarray:
    """Get the normals to the facets of a reference cell."""
    return _basixcpp.cell_facet_normals(celltype)


def facet_outward_normals(celltype: _basixcpp.CellType) -> _np.ndarray:
    """Get the outward pointing normals to the facets of a reference cell."""
    return _basixcpp.cell_facet_outward_normals(celltype)


def facet_orientations(celltype: _basixcpp.CellType) -> list:
    """Get the orientation of the facets of a reference cell."""
    return _basixcpp.cell_facet_orientations(celltype)


def facet_jacobians(celltype: _basixcpp.CellType) -> _np.ndarray:
    """Get the Jacobians of the facets of a reference cell."""
    return _basixcpp.cell_facet_jacobians(celltype)


def sub_entity_connectivity(celltype: _basixcpp.CellType) -> list:
    """Get the connectivity between subentities of a refernce cell."""
    return _basixcpp.sub_entity_connectivity(celltype)
