# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import numpy as np
import pytest


@pytest.mark.parametrize("cell_type", [
    basix.CellType.interval,
    basix.CellType.triangle,
    basix.CellType.quadrilateral,
    basix.CellType.tetrahedron,
    basix.CellType.hexahedron,
    basix.CellType.prism,
])
@pytest.mark.parametrize("order", range(8))
def test_cell(cell_type, order):
    Qpts, Qwts = basix.make_quadrature(cell_type, 2*order)
    basis = basix._basixcpp.tabulate_polynomial_set(cell_type, order, 0, Qpts)[0]

    ndofs = basis.shape[1]
    mat = np.zeros((ndofs, ndofs))
    for i in range(ndofs):
        for j in range(ndofs):
            mat[i, j] = sum(basis[:, i] * basis[:, j] * Qwts)

    assert np.allclose(mat, np.eye(ndofs))
