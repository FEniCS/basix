"""Functions to get cell geometry information and manipulate cell types."""

import typing

import numpy.typing as npt

from basix._basixcpp import CellType as _CT
from basix._basixcpp import cell_facet_jacobians as _fj
from basix._basixcpp import cell_facet_normals as _fn
from basix._basixcpp import cell_facet_orientations as _fo
from basix._basixcpp import cell_facet_outward_normals as _fon
from basix._basixcpp import cell_facet_reference_volumes as _frv
from basix._basixcpp import cell_volume as _v
from basix._basixcpp import geometry as _geometry
from basix._basixcpp import sub_entity_connectivity as _sec
from basix._basixcpp import topology as _topology
from basix.utils import Enum


class CellType(Enum):
    """Cell type."""
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


def sub_entity_connectivity(celltype: CellType) -> typing.List[typing.List[typing.List[typing.List[int]]]]:
    """Get the numbers of entities connected to each subentity of the cell.

    Args:
        celltype: The cell type

    Returns:
        List of topology (vertex indices) for each dimension (0..tdim)
    """
    return _sec(celltype.value)


def volume(celltype: CellType) -> float:
    """Get the volume of a reference cell.

    Args:
        celltype: The cell type

    Returns:
        The volume of the reference cell
    """
    return _v(celltype.value)


def facet_jacobians(celltype: CellType) -> npt.NDArray:
    """Get the jacobians of the facets of a reference cell.

    Args:
        celltype: The cell type

    Returns:
        The jacobians of the facets
    """
    return _fj(celltype.value)


def facet_normals(celltype: CellType) -> npt.NDArray:
    """Get the normals to the facets of a reference cell.

    These normals will be oriented using the low-to-high ordering of the facet.

    Args:
        celltype: The cell type

    Returns:
        The normals to the facets
    """
    return _fn(celltype.value)


def facet_orientations(celltype: CellType) -> typing.List[bool]:
    """Get the orientations of the facets of a reference cell.

    This returns a list of bools that are True if the facet normal points outwards
    and False otherwise.

    Args:
        celltype: The cell type

    Returns:
        The facet orientations
    """
    return _fo(celltype.value)


def facet_outward_normals(celltype: CellType) -> npt.NDArray:
    """Get the normals to the facets of a reference cell.

    These normals will be oriented to be pointing outwards.

    Args:
        celltype: The cell type

    Returns:
        The normals to the facets
    """
    return _fon(celltype.value)


def facet_reference_volumes(celltype: CellType) -> npt.NDArray:
    """Get the reference volumes of the facets of a reference cell.

    Args:
        celltype: The cell type

    Returns:
        The reference volumes
    """
    return _frv(celltype.value)


def geometry(celltype: CellType) -> npt.NDArray:
    """Get the cell geometry.

    Args:
        celltype: The cell type

    Returns:
        The vertices of the cell
    """
    return _geometry(celltype.value)


def topology(celltype: CellType) -> typing.List[typing.List[typing.List[int]]]:
    """Get the cell topology.

    Args:
        celltype: The cell type

    Returns:
        The list of vertex indices for each sub-entity of the cell
    """
    return _topology(celltype.value)
