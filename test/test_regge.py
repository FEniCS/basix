# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import pytest
import numpy as np


@pytest.mark.parametrize("degree", range(1, 4))
@pytest.mark.parametrize("cell", [basix.CellType.triangle, basix.CellType.tetrahedron])
def test_discontinuous_regge(degree, cell):
    e = basix.create_element(basix.ElementFamily.Regge, cell, degree)
    d_e = basix.create_element(basix.ElementFamily.Regge, cell, degree, True)

    pts = basix.create_lattice(cell, 5, basix.LatticeType.equispaced, True)

    assert np.allclose(e.tabulate(1, pts), d_e.tabulate(1, pts))

    for dofs in d_e.num_entity_dofs[:-1]:
        for d in dofs:
            assert d == 0
