import random

import basix
import numpy as np
import pytest
from basix import numba_helpers
from numba.core import types
from numba.typed import Dict


@pytest.mark.parametrize("cell", ["triangle", "tetrahedron", "quadrilateral", "hexahedron"])
@pytest.mark.parametrize("element, degree", [
    ("Lagrange", 1), ("Lagrange", 3), ("Nedelec 1st kind H(curl)", 3)
])
@pytest.mark.parametrize("block_size", [1, 2, 4])
def test_dof_transformations(cell, element, degree, block_size):

    transform_functions = {
        "triangle": numba_helpers.apply_dof_transformation_triangle,
        "quadrilateral": numba_helpers.apply_dof_transformation_quadrilateral,
        "tetrahedron": numba_helpers.apply_dof_transformation_tetrahedron,
        "hexahedron": numba_helpers.apply_dof_transformation_hexahedron
    }

    random.seed(1337)

    e = basix.create_element(element, cell, degree)
    data = np.array(range(e.dim * block_size), dtype=np.double)

    for i in range(10):
        cell_info = random.randrange(2 ** 30)

        data1 = data.copy()
        data1 = e.apply_dof_transformation(data1, block_size, cell_info)
        # Numba function does not use blocked data
        data2 = data.copy().reshape(e.dim, block_size)
        # Mapping lists to numba dictionaries
        entity_transformations = Dict.empty(key_type=types.int64, value_type=types.float64[:, :])
        for i, transformation in enumerate(e.entity_transformations()):
            entity_transformations[i] = transformation

        entity_dofs = Dict.empty(key_type=types.int64, value_type=types.int32[:])
        for i, e_dofs in enumerate(e.entity_dofs):
            entity_dofs[i] = np.asarray(e_dofs, dtype=np.int32)
        transform_functions[cell](entity_transformations, entity_dofs, data2, cell_info)
        # Reshape numba output for comparison
        data2 = data2.reshape(-1)
        assert np.allclose(data1, data2)
