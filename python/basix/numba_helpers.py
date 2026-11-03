# Copyright (C) 2023-2024 Matthew Scroggs and Garth N. Wells
#
# This file is part of Basix (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT
"""Helper functions for calling Basix from Numba-compiled kernels."""

try:
    import numba as _numba
    from llvmlite import binding as _llvm
    from numba.core import types as _types
except ImportError:
    print("You must have Numba installed to use the Numba helper functions.")
    raise

import numpy as np
import numpy.typing as npt

import basix._basixcpp as _basixcpp
from basix.finite_element import FiniteElement

__all__ = [
    "ElementHandle",
    "T_apply",
    "T_apply_hexahedron",
    "T_apply_interval",
    "T_apply_prism",
    "T_apply_pyramid",
    "T_apply_quadrilateral",
    "T_apply_tetrahedron",
    "T_apply_triangle",
    "Tt_apply_right",
    "Tt_apply_right_hexahedron",
    "Tt_apply_right_interval",
    "Tt_apply_right_prism",
    "Tt_apply_right_pyramid",
    "Tt_apply_right_quadrilateral",
    "Tt_apply_right_tetrahedron",
    "Tt_apply_right_triangle",
    "map_fn",
    "tabulate_fn",
]


# Numba signatures for the native Basix entry points. All pointers are
# passed as address-sized unsigned integers, which is the type that
# ``ndarray.ctypes.data`` has inside a Numba kernel.
_TABULATE_SIGNATURE = _types.int32(
    _types.uintp, _types.int32, _types.uintp, _types.uintp, _types.uintp
)
_MAP_SIGNATURE = _types.int32(
    _types.uintp,
    _types.int32,
    _types.uintp,
    _types.uintp,
    _types.uintp,
    _types.uintp,
    _types.uintp,
    _types.uintp,
    _types.uintp,
    _types.uintp,
)

_SCALAR_TYPES = {np.dtype(np.float32): "float32", np.dtype(np.float64): "float64"}


def _external_function(name: str, address: int, signature) -> _types.ExternalFunction:
    """Make a native function callable from Numba kernels.

    The address is registered with LLVM under ``name``, so that kernels
    that call it refer to it by symbol rather than by a literal address.
    Numba can therefore cache the compiled kernels.

    Args:
        name: Symbol name to register the function under.
        address: Address of the function.
        signature: Numba signature of the function.

    Returns:
        A function that can be called from a Numba kernel.
    """
    _llvm.add_symbol(name, address)
    return _types.ExternalFunction(name, signature)


_tabulate_fns = {
    dtype: _external_function(
        f"basix_numba_tabulate_{name}",
        getattr(_basixcpp, f"_numba_tabulate_address_{name}")(),
        _TABULATE_SIGNATURE,
    )
    for dtype, name in _SCALAR_TYPES.items()
}

_map_fns = {
    dtype: _external_function(
        f"basix_numba_map_{name}",
        getattr(_basixcpp, f"_numba_map_address_{name}")(),
        _MAP_SIGNATURE,
    )
    for dtype, name in _SCALAR_TYPES.items()
}


def _check_dtype(dtype: npt.DTypeLike) -> np.dtype:
    """Check that a scalar type is supported by the Numba interface."""
    _dtype = np.dtype(dtype)
    if _dtype not in _SCALAR_TYPES:
        raise ValueError(f"Unsupported dtype for Numba kernels: {dtype}")
    return _dtype


class ElementHandle:
    """A finite element in a form that can be passed to a Numba kernel.

    Numba kernels cannot hold a reference to a Basix element, so an
    element is passed to them as the address of the underlying C++
    object. This class holds that address alongside a reference to the
    element, so that the element cannot be garbage collected while the
    address is in use.

    Note:
        The address is valid only while this object is alive. Keep a
        reference to it for as long as any kernel may call
        :func:`tabulate_fn` or :func:`map_fn` with its address.

    Example:
        Tabulate an element at one point per cell::

            element = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 2)
            handle = basix.numba_helpers.ElementHandle(element)
            tabulate = basix.numba_helpers.tabulate_fn(element.dtype)


            @numba.njit
            def kernel(address, x, basis):
                for i in range(x.shape[0]):
                    if tabulate(address, 0, x[i].ctypes.data, 1, basis.ctypes.data) != 0:
                        raise RuntimeError("Basix tabulation failed.")


            kernel(handle.address, x, basis)
    """

    _element: FiniteElement
    address: int
    """Address of the underlying C++ finite element object."""

    dtype: np.dtype
    """Scalar type of the element."""

    def __init__(self, element: FiniteElement):
        """Create a handle to a finite element.

        Args:
            element: The finite element.
        """
        self._element = element
        self.address = element._e._address
        self.dtype = _check_dtype(element.dtype)


def tabulate_fn(dtype: npt.DTypeLike) -> _types.ExternalFunction:
    """Get a tabulation function that can be called from a Numba kernel.

    The returned function must be referenced as a global (or a closure
    variable) by the kernel that calls it. It has the signature::

        tabulate(address, nd, x, num_points, basis) -> int

    where

    - ``address`` is :attr:`ElementHandle.address` for the element to
      tabulate,
    - ``nd`` is the number of derivatives to compute,
    - ``x`` is the address of the points at which to tabulate, which
      must be a C-contiguous array of shape ``(num_points, tdim)``,
    - ``num_points`` is the number of points, and
    - ``basis`` is the address of the output buffer, which must be a
      C-contiguous array of shape
      ``element.tabulate_shape(nd, num_points)``.

    Array addresses are obtained inside a kernel with
    ``array.ctypes.data``. The return value is 0 on success and -1 if
    the call failed, for instance because an argument was invalid.

    Note:
        No checks are made on the sizes of ``x`` and ``basis``. Passing
        buffers that are too small corrupts memory.

    Note:
        The function does not use the Python API, so kernels that call
        it can be compiled with ``nogil=True``.

    Args:
        dtype: Scalar type of the element to tabulate.

    Returns:
        A tabulation function callable from a Numba kernel.
    """
    return _tabulate_fns[_check_dtype(dtype)]


def map_fn(dtype: npt.DTypeLike) -> _types.ExternalFunction:
    """Get a map function that can be called from a Numba kernel.

    The returned function maps values to or from a physical cell, in
    groups of points that share a Jacobian. It must be referenced as a
    global (or a closure variable) by the kernel that calls it, and has
    the signature::

        map(address, inverse, num_jacobians, num_points, gdim,
            values, J, detJ, K, out) -> int

    where

    - ``address`` is :attr:`ElementHandle.address` for the element whose
      map should be applied,
    - ``inverse`` is 0 to push ``values`` forward to the physical cell
      and 1 to pull them back to the reference cell,
    - ``num_jacobians`` is the number of groups of points that share a
      Jacobian,
    - ``num_points`` is the number of points in each group,
    - ``gdim`` is the geometric dimension of the physical cell,
    - ``values`` is the address of the values to map, shape
      ``(num_jacobians, num_points, vs)``,
    - ``J`` is the address of the Jacobians, shape
      ``(num_jacobians, gdim, tdim)``,
    - ``detJ`` is the address of the Jacobian determinants, shape
      ``(num_jacobians,)``,
    - ``K`` is the address of the inverses of the Jacobians, shape
      ``(num_jacobians, tdim, gdim)``, and
    - ``out`` is the address of the output buffer, shape
      ``(num_jacobians, num_points, vs)``.

    The value size ``vs`` is the reference value size on the reference
    cell and the physical value size on the physical cell. These differ
    when ``gdim`` is not equal to ``tdim``: for a contravariant Piola
    mapped element, for instance, the reference value size is ``tdim``
    and the physical value size is ``gdim``. All arrays must be
    C-contiguous, and their addresses are obtained inside a kernel with
    ``array.ctypes.data``. The return value is 0 on success and -1 if
    the call failed, for instance because the element uses a map that is
    not supported.

    Note:
        No checks are made on the sizes of the arrays. Passing buffers
        that are too small corrupts memory.

    Note:
        The function does not use the Python API, so kernels that call
        it can be compiled with ``nogil=True``.

    Args:
        dtype: Scalar type of the element whose map is applied.

    Returns:
        A map function callable from a Numba kernel.
    """
    return _map_fns[_check_dtype(dtype)]


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
    entity_dofs: list[list[int]],
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
    entity_dofs: list[list[int]],
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
    entity_dofs: list[list[int]],
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
    entity_dofs: list[list[int]],
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
    entity_dofs: list[list[int]],
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
    entity_dofs: list[list[int]],
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
    entity_dofs: list[list[int]],
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
    entity_transformations: dict[str, npt.NDArray],
    entity_dofs: list[list[int]],
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
    entity_dofs: list[list[int]],
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
    entity_dofs: list[list[int]],
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
    entity_dofs: list[list[int]],
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
    entity_dofs: list[list[int]],
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
    entity_dofs: list[list[int]],
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
    entity_dofs: list[list[int]],
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
    entity_dofs: list[list[int]],
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
