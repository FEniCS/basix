import basix
import numpy as np
import pytest
import random


@pytest.mark.parametrize("cell", ["triangle", "tetrahedron", "quadrilateral", "hexahedron"])
@pytest.mark.parametrize("element, degree", [
    ("Lagrange", 1), ("Lagrange", 3), ("Nedelec 1st kind H(curl)", 3)
])
@pytest.mark.parametrize("block_size", [1, 2, 4])
def test_dof_transformations(cell, element, degree, block_size):
    from basix import numba_helpers
    from numba.typed import List

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

        data2 = data.copy()

        transform_functions[cell](
            e.entity_transformations(), List(e.entity_dofs), data2, block_size, cell_info)

        assert np.allclose(data1, data2)
