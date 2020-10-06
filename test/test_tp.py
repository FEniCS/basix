# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import fiatx
import pytest
import numpy as np


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("celltype", [fiatx.CellType.quadrilateral, fiatx.CellType.hexahedron])
def test_tp(order, celltype):
    np.set_printoptions(suppress=True)
    tp = fiatx.TensorProduct(celltype, order)
    pts = fiatx.create_lattice(celltype, 6, True)
    w = tp.tabulate_basis(pts)
    assert(np.isclose(np.sum(w, axis=1), 1.0).all())
