import basix
import numpy as np
import pytest
import random

RED = "\033[31m"
GREEN = "\033[32m"
DEFAULT = "\033[0m"


@pytest.mark.parametrize("cell", ["triangle", "tetrahedron", "quadrilateral", "hexahedron"])
@pytest.mark.parametrize("element, degree", [
    ("Lagrange", 1), ("Lagrange", 3), ("Nedelec 1st kind H(curl)", 3)
])
@pytest.mark.parametrize("block_size", [1, 2, 4])
def test_dof_transformations(cell, element, degree, block_size):
    from basix.numba_helpers import apply_dof_transformation
    from numba.typed import List

    random.seed(1337)

    e = basix.create_element(element, cell, degree)
    data = np.array(range(e.dim * block_size), dtype=np.double)

    for i in range(10):
        cell_info = random.randrange(2 ** 30)

        data1 = data.copy()
        data1 = e.apply_dof_transformation(data1, block_size, cell_info)

        data2 = data.copy()
        for i in (e.entity_transformations(), basix.cell_to_str(e.cell_type),
                                 e.entity_dofs, data2, block_size, cell_info):
            print(type(i))

        apply_dof_transformation(e.entity_transformations(), basix.cell_to_str(e.cell_type),
                                 List(e.entity_dofs), data2, block_size, cell_info)

        assert np.allclose(data1, data2)
