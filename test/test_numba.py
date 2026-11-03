# Copyright (c) 2021 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import random

import numpy as np
import pytest

import basix
from basix import CellType

pytest.importorskip("numba")

from numba.core import types
from numba.typed import Dict

from basix import numba_helpers

from .utils import cached_create_element


@pytest.mark.parametrize(
    "cell",
    [
        CellType.triangle,
        CellType.tetrahedron,
        CellType.quadrilateral,
        CellType.hexahedron,
    ],
)
@pytest.mark.parametrize(
    "element, degree, element_args",
    [
        (basix.ElementFamily.P, 1, [basix.LagrangeVariant.gll_warped]),
        (basix.ElementFamily.P, 3, [basix.LagrangeVariant.gll_warped]),
        (basix.ElementFamily.N1E, 3, []),
    ],
)
@pytest.mark.parametrize("block_size", [1, 2, 4])
def test_dof_transformations(cell, element, degree, element_args, block_size):

    transform_functions = {
        CellType.triangle: numba_helpers.T_apply_triangle,
        CellType.quadrilateral: numba_helpers.T_apply_quadrilateral,
        CellType.tetrahedron: numba_helpers.T_apply_tetrahedron,
        CellType.hexahedron: numba_helpers.T_apply_hexahedron,
        CellType.prism: numba_helpers.T_apply_prism,
        CellType.pyramid: numba_helpers.T_apply_pyramid,
    }

    random.seed(1337)

    e = cached_create_element(element, cell, degree, tuple(element_args))
    data = np.array(list(range(e.dim * block_size)), dtype=np.double)

    for i in range(10):
        cell_info = random.randrange(2**30)

        data1 = data.copy()
        e.T_apply(data1, block_size, cell_info)
        # Numba function does not use blocked data
        data2 = data.copy().reshape(e.dim, block_size)
        # Mapping lists to numba dictionaries
        entity_transformations = Dict.empty(
            key_type=types.string, value_type=types.float64[:, :, :]
        )
        for i, transformation in e.entity_transformations().items():
            entity_transformations[i] = transformation

        entity_dofs = Dict.empty(key_type=types.int64, value_type=types.int32[:])
        for i, e_dofs in enumerate(e.num_entity_dofs):
            entity_dofs[i] = np.asarray(e_dofs, dtype=np.int32)
        transform_functions[cell](entity_transformations, entity_dofs, data2, cell_info)
        # Reshape numba output for comparison
        data2 = data2.reshape(-1)
        assert np.allclose(data1, data2)


@pytest.mark.parametrize(
    "cell",
    [
        CellType.triangle,
        CellType.tetrahedron,
        CellType.quadrilateral,
        CellType.hexahedron,
    ],
)
@pytest.mark.parametrize(
    "element, degree, element_args",
    [
        (basix.ElementFamily.P, 1, [basix.LagrangeVariant.gll_warped]),
        (basix.ElementFamily.P, 3, [basix.LagrangeVariant.gll_warped]),
        (basix.ElementFamily.N1E, 3, []),
    ],
)
@pytest.mark.parametrize("block_size", [1, 2, 4])
def test_dof_transformations_to_transpose(cell, element, degree, block_size, element_args):
    transform_functions = {
        CellType.triangle: numba_helpers.Tt_apply_right_triangle,
        CellType.quadrilateral: numba_helpers.Tt_apply_right_quadrilateral,
        CellType.tetrahedron: numba_helpers.Tt_apply_right_tetrahedron,
        CellType.hexahedron: numba_helpers.Tt_apply_right_hexahedron,
        CellType.prism: numba_helpers.Tt_apply_right_prism,
        CellType.pyramid: numba_helpers.Tt_apply_right_pyramid,
    }

    random.seed(1337)

    e = cached_create_element(element, cell, degree, tuple(element_args))
    data = np.array(list(range(e.dim * block_size)), dtype=np.double)

    for i in range(10):
        cell_info = random.randrange(2**30)

        data1 = data.copy()
        e.Tt_apply_right(data1, block_size, cell_info)
        # Numba function does not use blocked data
        data2 = data.copy().reshape(block_size, e.dim)
        # Mapping lists to numba dictionaries
        entity_transformations = Dict.empty(
            key_type=types.string, value_type=types.float64[:, :, :]
        )
        for i, transformation in e.entity_transformations().items():
            entity_transformations[i] = transformation

        entity_dofs = Dict.empty(key_type=types.int64, value_type=types.int32[:])
        for i, e_dofs in enumerate(e.num_entity_dofs):
            entity_dofs[i] = np.asarray(e_dofs, dtype=np.int32)
        transform_functions[cell](entity_transformations, entity_dofs, data2, cell_info)
        # Reshape numba output for comparison
        data2 = data2.reshape(-1)
        assert np.allclose(data1, data2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "cell, family, degree, args",
    [
        (CellType.interval, basix.ElementFamily.P, 2, [basix.LagrangeVariant.gll_isaac]),
        (CellType.triangle, basix.ElementFamily.P, 3, [basix.LagrangeVariant.gll_isaac]),
        (CellType.triangle, basix.ElementFamily.RT, 2, []),
        (CellType.tetrahedron, basix.ElementFamily.N1E, 2, []),
        (CellType.quadrilateral, basix.ElementFamily.P, 2, [basix.LagrangeVariant.gll_isaac]),
    ],
)
@pytest.mark.parametrize("nd", [0, 1, 2])
def test_numba_tabulate(cell, family, degree, args, nd, dtype):
    import numba

    e = basix.create_element(family, cell, degree, *args, dtype=dtype)
    handle = numba_helpers.ElementHandle(e)
    tabulate = numba_helpers.tabulate_fn(dtype)

    @numba.njit
    def kernel(address, nd, x, basis):
        return tabulate(address, nd, x.ctypes.data, x.shape[0], basis.ctypes.data)

    tdim = len(basix.topology(cell)) - 1
    rng = np.random.default_rng(42)
    x = np.ascontiguousarray(rng.random((5, tdim)) / (tdim + 1), dtype=dtype)

    basis = np.zeros(e.tabulate_shape(nd, x.shape[0]), dtype=dtype)
    assert kernel(handle.address, nd, x, basis) == 0
    assert np.allclose(basis, e.tabulate(nd, x))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "cell, family, degree, gdim",
    [
        (CellType.triangle, basix.ElementFamily.P, 2, 2),
        (CellType.triangle, basix.ElementFamily.RT, 1, 2),
        # Reference and physical value sizes differ when gdim != tdim
        (CellType.triangle, basix.ElementFamily.RT, 2, 3),
        (CellType.triangle, basix.ElementFamily.N1E, 1, 3),
        (CellType.tetrahedron, basix.ElementFamily.RT, 1, 3),
    ],
)
def test_numba_map(cell, family, degree, gdim, dtype):
    import numba

    e = basix.create_element(family, cell, degree, dtype=dtype)
    handle = numba_helpers.ElementHandle(e)
    apply_map = numba_helpers.map_fn(dtype)

    @numba.njit
    def kernel(address, inverse, num_jacobians, num_points, gdim, values, J, detJ, K, out):
        return apply_map(
            address,
            inverse,
            num_jacobians,
            num_points,
            gdim,
            values.ctypes.data,
            J.ctypes.data,
            detJ.ctypes.data,
            K.ctypes.data,
            out.ctypes.data,
        )

    tdim = len(basix.topology(cell)) - 1
    num_jacobians, num_points = 3, 4

    rng = np.random.default_rng(13)
    J = np.ascontiguousarray(rng.random((num_jacobians, gdim, tdim)) + np.eye(gdim, tdim))
    K = np.ascontiguousarray([np.linalg.pinv(J[i]) for i in range(num_jacobians)])
    detJ = np.array([np.sqrt(np.linalg.det(J[i].T @ J[i])) for i in range(num_jacobians)])
    J, K, detJ = (np.asarray(a, dtype=dtype) for a in (J, K, detJ))

    reference_vs = max(np.prod(e.value_shape, dtype=int), 1)
    U = np.ascontiguousarray(rng.random((num_jacobians, num_points, reference_vs)), dtype=dtype)

    u = e.push_forward(U, J, detJ, K)
    out = np.zeros_like(u)
    assert kernel(handle.address, 0, num_jacobians, num_points, gdim, U, J, detJ, K, out) == 0
    assert np.allclose(out, u)

    back = np.zeros_like(U)
    assert kernel(handle.address, 1, num_jacobians, num_points, gdim, u, J, detJ, K, back) == 0
    assert np.allclose(back, e.pull_back(u, J, detJ, K))


def test_numba_kernels_are_cacheable(tmp_path):
    """A kernel calling Basix must not be excluded from Numba's cache."""
    import warnings

    import numba

    e = basix.create_element(basix.ElementFamily.P, CellType.triangle, 1)
    handle = numba_helpers.ElementHandle(e)
    tabulate = numba_helpers.tabulate_fn(np.float64)

    @numba.njit(cache=True)
    def kernel(address, x, basis):
        return tabulate(address, 0, x.ctypes.data, x.shape[0], basis.ctypes.data)

    x = np.array([[0.25, 0.25]])
    basis = np.zeros(e.tabulate_shape(0, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error", numba.NumbaWarning)
        assert kernel(handle.address, x, basis) == 0


def test_numba_element_handle():
    e = basix.create_element(basix.ElementFamily.P, CellType.triangle, 1)
    handle = numba_helpers.ElementHandle(e)
    assert handle.address != 0
    assert handle.dtype == np.dtype(np.float64)

    # The handle must keep the element alive
    del e
    assert handle._element is not None

    with pytest.raises(ValueError):
        numba_helpers.tabulate_fn(np.complex128)
    with pytest.raises(ValueError):
        numba_helpers.map_fn(np.int32)


def test_numba_map_reports_errors():
    """A map that Basix does not support must be reported, not raised.

    A C++ exception must never escape into JIT-compiled code, so the
    kernel entry points report failure through their return value.
    """
    import numba

    # A custom element with an L2 Piola map, which push_forward and
    # pull_back do not support
    wcoeffs = np.eye(3)
    z = np.zeros((0, 2))
    x = [
        [np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])],
        [z, z, z],
        [z],
        [],
    ]
    z = np.zeros((0, 1, 0, 1))
    M = [[np.array([[[[1.0]]]]), np.array([[[[1.0]]]]), np.array([[[[1.0]]]])], [z, z, z], [z], []]
    e = basix.create_custom_element(
        CellType.triangle,
        (),
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.L2Piola,
        basix.SobolevSpace.L2,
        False,
        1,
        1,
        basix.PolysetType.standard,
    )

    handle = numba_helpers.ElementHandle(e)
    apply_map = numba_helpers.map_fn(np.float64)

    @numba.njit
    def kernel(address, values, J, detJ, K, out):
        return apply_map(
            address,
            0,
            1,
            1,
            2,
            values.ctypes.data,
            J.ctypes.data,
            detJ.ctypes.data,
            K.ctypes.data,
            out.ctypes.data,
        )

    J = np.eye(2)
    values = np.ones((1, 1, 1))
    out = np.zeros((1, 1, 1))
    assert kernel(handle.address, values, J, np.ones(1), J, out) == -1

    with pytest.raises(RuntimeError):
        e.push_forward(values, J.reshape(1, 2, 2), np.ones(1), J.reshape(1, 2, 2))
