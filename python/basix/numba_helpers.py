"""Helper functions for writing DOLFINx custom kernels using Numba."""

try:
    import numba as _numba
except ImportError:
    raise RuntimeError("You must have Numba installed to use the Numba helper functions.")

import numpy as _np
import typing as _typing
from typing import List as _ListT
from typing import Dict as _Dict
if _typing.TYPE_CHECKING:
    import numpy.typing as _npt
    _nda = _npt.NDArray
    _nda_i32 = _npt.NDArray[_np.int32]
    _nda_f64 = _npt.NDArray[_np.float64]
else:
    _nda = None
    _nda_i32 = None
    _nda_f64 = None


@_numba.jit(nopython=True)
def apply_dof_transformation(
    tdim: int, edge_count: int, face_count: int, entity_transformations: _Dict[str, _nda],
    entity_dofs: _ListT[_ListT[int]], data: _nda, cell_info: int, face_types: _ListT[str]
):
    """Apply dof transformations to some data.

    Args:
        tdim: The topological dimension of the cell.
        edge_count: The number of edges the cell has.
        face_count: The number of faces the cell has.
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the subentities of the cell.
        face_types: A list of strings giving the shapes of the faces of the cell.
    """
    if tdim >= 2:
        if tdim == 3:
            face_start = 3 * face_count
        else:
            face_start = 0

        dofstart = 0
        for i in entity_dofs[0]:
            dofstart += i
        # NOTE: Copy array to make numba compilation faster (contiguous array assumption)
        edge_reflection = entity_transformations["interval"][0].copy()
        for e in range(edge_count):
            edofs = entity_dofs[1][e]
            if edofs == 0:
                continue
            if cell_info >> (face_start + e) & 1:
                data[dofstart:dofstart+edofs] = _np.dot(edge_reflection, data[dofstart:dofstart+edofs])
            dofstart += edofs

        if tdim == 3:
            for f in range(face_count):
                face_rotation = entity_transformations[face_types[f]][0].copy()
                face_reflection = entity_transformations[face_types[f]][1].copy()
                fdofs = entity_dofs[2][f]
                if fdofs == 0:
                    continue
                if cell_info >> (3 * f) & 1:
                    data[dofstart:dofstart+fdofs] = _np.dot(face_reflection, data[dofstart:dofstart+fdofs])
                for _ in range(cell_info >> (3 * f + 1) & 3):
                    data[dofstart:dofstart+fdofs] = _np.dot(face_rotation, data[dofstart:dofstart+fdofs])
                dofstart += fdofs


@_numba.jit(nopython=True)
def apply_dof_transformation_interval(
    entity_transformations: _Dict[str, _nda_f64],
    entity_dofs: _Dict[str, _nda_i32],
    data: _nda, cell_info: int
):
    """Apply dof transformations to some data on an interval.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the subentities of the cell.
    """
    return


@_numba.jit(nopython=True)
def apply_dof_transformation_triangle(
    entity_transformations: _Dict[str, _nda_f64],
    entity_dofs: _Dict[str, _nda_i32],
    data: _nda, cell_info: int
):
    """Apply dof transformations to some data on a triangle.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the subentities of the cell.
    """
    apply_dof_transformation(2, 3, 1, entity_transformations, entity_dofs,
                             data, cell_info, _numba.typed.List.empty_list(_numba.core.types.string))


@_numba.jit(nopython=True)
def apply_dof_transformation_quadrilateral(
    entity_transformations: _Dict[str, _nda_f64],
    entity_dofs: _Dict[str, _nda_i32],
    data: _nda, cell_info: int
):
    """Apply dof transformations to some data on an quadrilateral.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the subentities of the cell.
    """
    apply_dof_transformation(2, 4, 1, entity_transformations, entity_dofs,
                             data, cell_info, _numba.typed.List.empty_list(_numba.core.types.string))


@_numba.jit(nopython=True)
def apply_dof_transformation_tetrahedron(
    entity_transformations: _Dict[str, _nda_f64],
    entity_dofs: _Dict[str, _nda_i32],
    data: _nda, cell_info: int
):
    """Apply dof transformations to some data on a tetrahedron.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the subentities of the cell.
    """
    apply_dof_transformation(3, 6, 4, entity_transformations, entity_dofs,
                             data, cell_info, _numba.typed.List(["triangle"] * 4))


@_numba.jit(nopython=True)
def apply_dof_transformation_hexahedron(
    entity_transformations: _Dict[str, _nda_f64],
    entity_dofs: _Dict[str, _nda_i32],
    data: _nda, cell_info: int
):
    """Apply dof transformations to some data on a hexahedron.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the subentities of the cell.
    """
    apply_dof_transformation(3, 12, 6, entity_transformations, entity_dofs,
                             data, cell_info, _numba.typed.List(["quadrilateral"] * 6))


@_numba.jit(nopython=True)
def apply_dof_transformation_prism(
    entity_transformations: _Dict[str, _nda_f64],
    entity_dofs: _Dict[str, _nda_i32],
    data: _nda, cell_info: int
):
    """Apply dof transformations to some data on an prism.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the subentities of the cell.
    """
    apply_dof_transformation(3, 9, 5, entity_transformations, entity_dofs,
                             data, cell_info, _numba.typed.List(["triangle"] + ["quadrilateral"] * 4 + ["triangle"]))


@_numba.jit(nopython=True)
def apply_dof_transformation_pyramid(
    entity_transformations: _Dict[str, _nda_f64],
    entity_dofs: _Dict[str, _nda_i32],
    data: _nda, cell_info: int
):
    """Apply dof transformations to some data on an prism.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the subentities of the cell.
    """
    apply_dof_transformation(3, 8, 5, entity_transformations, entity_dofs,
                             data, cell_info, _numba.typed.List(["quadrilateral"] + ["triangle"] * 4))


@_numba.jit(nopython=True)
def apply_dof_transformation_to_transpose(
    tdim: int, edge_count: int, face_count: int, entity_transformations: _ListT[int], entity_dofs: _ListT[int],
    data: _nda, cell_info: int, face_types: _ListT[str]
):
    """Apply dof transformations to some transposed data.

    Args:
        tdim: The topological dimension of the cell.
        edge_count: The number of edges the cell has.
        face_count: The number of faces the cell has.
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the subentities of the cell.
        face_types: A list of strings giving the shapes of the faces of the cell.
    """
    transposed_data = data.transpose().copy()
    apply_dof_transformation(tdim, edge_count, face_count, entity_transformations, entity_dofs,
                             transposed_data, cell_info, face_types)
    data[:] = transposed_data.transpose()


@_numba.jit(nopython=True)
def apply_dof_transformation_to_transpose_interval(
    entity_transformations: _Dict[str, _nda_f64],
    entity_dofs: _Dict[str, _nda_i32],
    data: _nda, cell_info: int
):
    """Apply dof transformations to some transposed data on an interval.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the subentities of the cell.
    """
    return


@_numba.jit(nopython=True)
def apply_dof_transformation_to_transpose_triangle(
    entity_transformations: _Dict[str, _nda_f64],
    entity_dofs: _Dict[str, _nda_i32],
    data: _nda, cell_info: int
):
    """Apply dof transformations to some transposed data on a triangle.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the subentities of the cell.
    """
    apply_dof_transformation_to_transpose(2, 3, 1, entity_transformations, entity_dofs,
                                          data, cell_info, _numba.typed.List.empty_list(_numba.core.types.string))


@_numba.jit(nopython=True)
def apply_dof_transformation_to_transpose_quadrilateral(
    entity_transformations: _Dict[str, _nda_f64],
    entity_dofs: _Dict[str, _nda_i32],
    data: _nda, cell_info: int
):
    """Apply dof transformations to some transposed data on an quadrilateral.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the subentities of the cell.
    """
    apply_dof_transformation_to_transpose(2, 4, 1, entity_transformations, entity_dofs,
                                          data, cell_info, _numba.typed.List.empty_list(_numba.core.types.string))


@_numba.jit(nopython=True)
def apply_dof_transformation_to_transpose_tetrahedron(
    entity_transformations: _Dict[str, _nda_f64],
    entity_dofs: _Dict[str, _nda_i32],
    data: _nda, cell_info: int
):
    """Apply dof transformations to some transposed data on a tetrahedron.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the subentities of the cell.
    """
    apply_dof_transformation_to_transpose(3, 6, 4, entity_transformations, entity_dofs,
                                          data, cell_info, _numba.typed.List(["triangle"] * 4))


@_numba.jit(nopython=True)
def apply_dof_transformation_to_transpose_hexahedron(
    entity_transformations: _Dict[str, _nda_f64],
    entity_dofs: _Dict[str, _nda_i32],
    data: _nda, cell_info: int
):
    """Apply dof transformations to some transposed data on a hexahedron.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the subentities of the cell.
    """
    apply_dof_transformation_to_transpose(3, 12, 6, entity_transformations, entity_dofs,
                                          data, cell_info, _numba.typed.List(["quadrilateral"] * 6))


@_numba.jit(nopython=True)
def apply_dof_transformation_to_transpose_prism(
    entity_transformations: _Dict[str, _nda_f64],
    entity_dofs: _Dict[str, _nda_i32],
    data: _nda, cell_info: int
):
    """Apply dof transformations to some transposed data on an prism.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the subentities of the cell.
    """
    apply_dof_transformation_to_transpose(
        3, 9, 5, entity_transformations, entity_dofs,
        data, cell_info, _numba.typed.List(["triangle"] + ["quadrilateral"] * 4 + ["triangle"]))


@_numba.jit(nopython=True)
def apply_dof_transformation_to_transpose_pyramid(
    entity_transformations: _Dict[str, _nda_f64],
    entity_dofs: _Dict[str, _nda_i32],
    data: _nda, cell_info: int
):
    """Apply dof transformations to some transposed data on an prism.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the subentities of the cell.
    """
    apply_dof_transformation_to_transpose(3, 8, 5, entity_transformations, entity_dofs,
                                          data, cell_info, _numba.typed.List(["quadrilateral"] + ["triangle"] * 4))
