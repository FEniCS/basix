# Copyright (c) 2025 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

from functools import cmp_to_key

import numpy as np
import pytest

import basix


@pytest.mark.parametrize(
    "cell_type",
    [
        basix.CellType.interval,
        basix.CellType.triangle,
        basix.CellType.quadrilateral,
        basix.CellType.tetrahedron,
        basix.CellType.hexahedron,
        basix.CellType.prism,
        basix.CellType.pyramid,
    ],
)
@pytest.mark.parametrize("degree", range(1, 10))
def test_dof_ordering(cell_type, degree):
    element = basix.finite_element.create_element(
        basix.ElementFamily.P, cell_type, degree, basix.LagrangeVariant.equispaced
    )

    def cmp(x, y):
        for i, j in zip(x[1][::-1], y[1][::-1]):
            if not np.isclose(i, j):
                if i > j:
                    return 1
                else:
                    return -1
        return 0

    dof_points = list(enumerate(element.points))
    dof_points.sort(key=cmp_to_key(cmp))

    lex_order = [i[0] for i in dof_points]

    ordering = basix.finite_element.lex_dof_ordering(
        basix.ElementFamily.P, cell_type, degree, basix.LagrangeVariant.equispaced
    )
    for i, p in enumerate(lex_order):
        assert ordering[p] == i
