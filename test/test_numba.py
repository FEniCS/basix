# Copyright (c) 2021 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import random

import basix
import numpy as np
import pytest
from basix import CellType


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
    try:
        import numba  # noqa: F401
    except ImportError:
        pytest.skip("Numba must be installed to run this test.")

    from basix import numba_helpers
    from numba.core import types
    from numba.typed import Dict

    transform_functions = {
        CellType.triangle: numba_helpers.T_apply_triangle,
        CellType.quadrilateral: numba_helpers.T_apply_quadrilateral,
        CellType.tetrahedron: numba_helpers.T_apply_tetrahedron,
        CellType.hexahedron: numba_helpers.T_apply_hexahedron,
        CellType.prism: numba_helpers.T_apply_prism,
        CellType.pyramid: numba_helpers.T_apply_pyramid,
    }

    random.seed(1337)

    e = basix.create_element(element, cell, degree, *element_args)
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
    try:
        import numba  # noqa: F401
    except ImportError:
        pytest.skip("Numba must be installed to run this test.")

    from basix import numba_helpers
    from numba.core import types
    from numba.typed import Dict

    transform_functions = {
        CellType.triangle: numba_helpers.Tt_apply_right_triangle,
        CellType.quadrilateral: numba_helpers.Tt_apply_right_quadrilateral,
        CellType.tetrahedron: numba_helpers.Tt_apply_right_tetrahedron,
        CellType.hexahedron: numba_helpers.Tt_apply_right_hexahedron,
        CellType.prism: numba_helpers.Tt_apply_right_prism,
        CellType.pyramid: numba_helpers.Tt_apply_right_pyramid,
    }

    random.seed(1337)

    e = basix.create_element(element, cell, degree, *element_args)
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
