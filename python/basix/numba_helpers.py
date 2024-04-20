# Copyright (C) 2023-2024 Matthew Scroggs and Garth N. Wells
#
# This file is part of Basix (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT
"""Helper functions for writing DOLFINx custom kernels using Numba."""

try:
    import numba as _numba
except ImportError:
    print("You must have Numba installed to use the Numba helper functions.")
    raise

import numpy as np
import numpy.typing as npt

__all__ = [
    "T_apply",
    "T_apply_interval",
    "T_apply_triangle",
    "T_apply_quadrilateral",
    "T_apply_tetrahedron",
    "T_apply_hexahedron",
    "T_apply_prism",
    "T_apply_pyramid",
    "Tt_apply_right",
    "Tt_apply_right_interval",
    "Tt_apply_right_triangle",
    "Tt_apply_right_quadrilateral",
    "Tt_apply_right_tetrahedron",
    "Tt_apply_right_hexahedron",
    "Tt_apply_right_prism",
    "Tt_apply_right_pyramid",
]


@_numba.jit(nopython=True)
def T_apply(
    tdim: int,
    edge_count: int,
    face_count: int,
    entity_transformations: dict[str, npt.NDArray],
    entity_dofs: list[list[int]],
    data: npt.NDArray,
    cell_info: int,
    face_types: list[str],
):
    """Apply dof transformations to some data.

    Args:
        tdim: The topological dimension of the cell.
        edge_count: The number of edges the cell has.
        face_count: The number of faces the cell has.
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the
            subentities of the cell.
        face_types: A list of strings giving the shapes of the faces of
            the cell.

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
                data[dofstart : dofstart + edofs] = np.dot(
                    edge_reflection, data[dofstart : dofstart + edofs]
                )
            dofstart += edofs

        if tdim == 3:
            for f in range(face_count):
                face_rotation = entity_transformations[face_types[f]][0].copy()
                face_reflection = entity_transformations[face_types[f]][1].copy()
                fdofs = entity_dofs[2][f]
                if fdofs == 0:
                    continue
                if cell_info >> (3 * f) & 1:
                    data[dofstart : dofstart + fdofs] = np.dot(
                        face_reflection, data[dofstart : dofstart + fdofs]
                    )
                for _ in range(cell_info >> (3 * f + 1) & 3):
                    data[dofstart : dofstart + fdofs] = np.dot(
                        face_rotation, data[dofstart : dofstart + fdofs]
                    )
                dofstart += fdofs


@_numba.jit(nopython=True)
def T_apply_interval(
    entity_transformations: dict[str, npt.NDArray[np.float64]],
    entity_dofs: dict[str, npt.NDArray[np.int32]],
    data: npt.NDArray,
    cell_info: int,
):
    """Apply dof transformations to some data on an interval.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the
            subentities of the cell.

    """
    return


@_numba.jit(nopython=True)
def T_apply_triangle(
    entity_transformations: dict[str, npt.NDArray[np.float64]],
    entity_dofs: dict[str, npt.NDArray[np.int32]],
    data: npt.NDArray,
    cell_info: int,
):
    """Apply dof transformations to some data on a triangle.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the
            sub-entities of the cell.

    """
    T_apply(
        2,
        3,
        1,
        entity_transformations,
        entity_dofs,
        data,
        cell_info,
        _numba.typed.List.empty_list(_numba.core.types.string),
    )


@_numba.jit(nopython=True)
def T_apply_quadrilateral(
    entity_transformations: dict[str, npt.NDArray[np.float64]],
    entity_dofs: dict[str, npt.NDArray[np.int32]],
    data: npt.NDArray,
    cell_info: int,
):
    """Apply dof transformations to some data on an quadrilateral.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the
            sub-entities of the cell.

    """
    T_apply(
        2,
        4,
        1,
        entity_transformations,
        entity_dofs,
        data,
        cell_info,
        _numba.typed.List.empty_list(_numba.core.types.string),
    )


@_numba.jit(nopython=True)
def T_apply_tetrahedron(
    entity_transformations: dict[str, npt.NDArray[np.float64]],
    entity_dofs: dict[str, npt.NDArray[np.int32]],
    data: npt.NDArray,
    cell_info: int,
):
    """Apply dof transformations to some data on a tetrahedron.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the
            subentities of the cell.

    """
    T_apply(
        3,
        6,
        4,
        entity_transformations,
        entity_dofs,
        data,
        cell_info,
        _numba.typed.List(["triangle"] * 4),
    )


@_numba.jit(nopython=True)
def T_apply_hexahedron(
    entity_transformations: dict[str, npt.NDArray[np.float64]],
    entity_dofs: dict[str, npt.NDArray[np.int32]],
    data: npt.NDArray,
    cell_info: int,
):
    """Pre-apply dof transformations to some data on a hexahedron.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the
            subentities of the cell.

    """
    T_apply(
        3,
        12,
        6,
        entity_transformations,
        entity_dofs,
        data,
        cell_info,
        _numba.typed.List(["quadrilateral"] * 6),
    )


@_numba.jit(nopython=True)
def T_apply_prism(
    entity_transformations: dict[str, npt.NDArray[np.float64]],
    entity_dofs: dict[str, npt.NDArray[np.int32]],
    data: npt.NDArray,
    cell_info: int,
):
    """Apply dof transformations to some data on an prism.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the
            subentities of the cell.

    """
    T_apply(
        3,
        9,
        5,
        entity_transformations,
        entity_dofs,
        data,
        cell_info,
        _numba.typed.List(["triangle"] + ["quadrilateral"] * 4 + ["triangle"]),
    )


@_numba.jit(nopython=True)
def T_apply_pyramid(
    entity_transformations: dict[str, npt.NDArray[np.float64]],
    entity_dofs: dict[str, npt.NDArray[np.int32]],
    data: npt.NDArray,
    cell_info: int,
):
    """Apply dof transformations to some data on an prism.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the
            subentities of the cell.

    """
    T_apply(
        3,
        8,
        5,
        entity_transformations,
        entity_dofs,
        data,
        cell_info,
        _numba.typed.List(["quadrilateral"] + ["triangle"] * 4),
    )


@_numba.jit(nopython=True)
def Tt_apply_right(
    tdim: int,
    edge_count: int,
    face_count: int,
    entity_transformations: list[int],
    entity_dofs: list[int],
    data: npt.NDArray,
    cell_info: int,
    face_types: list[str],
):
    """Right(post)-apply dof transformations to some transposed data.

    Args:
        tdim: The topological dimension of the cell.
        edge_count: The number of edges the cell has.
        face_count: The number of faces the cell has.
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the
            subentities of the cell.
        face_types: A list of strings giving the shapes of the faces of
            the cell.

    """
    transposed_data = data.transpose().copy()
    T_apply(
        tdim,
        edge_count,
        face_count,
        entity_transformations,
        entity_dofs,
        transposed_data,
        cell_info,
        face_types,
    )
    data[:] = transposed_data.transpose()


@_numba.jit(nopython=True)
def Tt_apply_right_interval(
    entity_transformations: dict[str, npt.NDArray[np.float64]],
    entity_dofs: dict[str, npt.NDArray[np.int32]],
    data: npt.NDArray,
    cell_info: int,
):
    """Right(post)-apply dof transformations to some transposed data on an interval.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the
            subentities of the cell.

    """
    return


@_numba.jit(nopython=True)
def Tt_apply_right_triangle(
    entity_transformations: dict[str, npt.NDArray[np.float64]],
    entity_dofs: dict[str, npt.NDArray[np.int32]],
    data: npt.NDArray,
    cell_info: int,
):
    """Right(post)-apply dof transformations to some transposed data on a triangle.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the
            subentities of the cell.

    """
    Tt_apply_right(
        2,
        3,
        1,
        entity_transformations,
        entity_dofs,
        data,
        cell_info,
        _numba.typed.List.empty_list(_numba.core.types.string),
    )


@_numba.jit(nopython=True)
def Tt_apply_right_quadrilateral(
    entity_transformations: dict[str, npt.NDArray[np.float64]],
    entity_dofs: dict[str, npt.NDArray[np.int32]],
    data: npt.NDArray,
    cell_info: int,
):
    """Right(post)-apply dof transformations to some transposed data on an quadrilateral.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the
            subentities of the cell.

    """
    Tt_apply_right(
        2,
        4,
        1,
        entity_transformations,
        entity_dofs,
        data,
        cell_info,
        _numba.typed.List.empty_list(_numba.core.types.string),
    )


@_numba.jit(nopython=True)
def Tt_apply_right_tetrahedron(
    entity_transformations: dict[str, npt.NDArray[np.float64]],
    entity_dofs: dict[str, npt.NDArray[np.int32]],
    data: npt.NDArray,
    cell_info: int,
):
    """Right(post)-apply dof transformations to some transposed data on a tetrahedron.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the
        subentities of the cell.

    """
    Tt_apply_right(
        3,
        6,
        4,
        entity_transformations,
        entity_dofs,
        data,
        cell_info,
        _numba.typed.List(["triangle"] * 4),
    )


@_numba.jit(nopython=True)
def Tt_apply_right_hexahedron(
    entity_transformations: dict[str, npt.NDArray[np.float64]],
    entity_dofs: dict[str, npt.NDArray[np.int32]],
    data: npt.NDArray,
    cell_info: int,
):
    """Right(post)-apply dof transformations to some transposed data on a hexahedron.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the
            subentities of the cell.

    """
    Tt_apply_right(
        3,
        12,
        6,
        entity_transformations,
        entity_dofs,
        data,
        cell_info,
        _numba.typed.List(["quadrilateral"] * 6),
    )


@_numba.jit(nopython=True)
def Tt_apply_right_prism(
    entity_transformations: dict[str, npt.NDArray[np.float64]],
    entity_dofs: dict[str, npt.NDArray[np.int32]],
    data: npt.NDArray,
    cell_info: int,
):
    """Right(post)-apply dof transformations to some transposed data on an prism.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the
            subentities of the cell.

    """
    Tt_apply_right(
        3,
        9,
        5,
        entity_transformations,
        entity_dofs,
        data,
        cell_info,
        _numba.typed.List(["triangle"] + ["quadrilateral"] * 4 + ["triangle"]),
    )


@_numba.jit(nopython=True)
def Tt_apply_right_pyramid(
    entity_transformations: dict[str, npt.NDArray[np.float64]],
    entity_dofs: dict[str, npt.NDArray[np.int32]],
    data: npt.NDArray,
    cell_info: int,
):
    """Right(post)-apply dof transformations to some transposed data on an prism.

    Args:
        entity_transformations: The DOF transformations for each entity.
        entity_dofs: The number of DOFs on each entity.
        data: The data. This will be changed by this function.
        cell_info: An integer representing the orientations of the
            subentities of the cell.

    """
    Tt_apply_right(
        3,
        8,
        5,
        entity_transformations,
        entity_dofs,
        data,
        cell_info,
        _numba.typed.List(["quadrilateral"] + ["triangle"] * 4),
    )
