# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import fiatx
import pytest
import numpy as np


@pytest.mark.parametrize("order", [1, 2, 3])
def test_tp(order):
    np.set_printoptions(suppress=True)
    tp = fiatx.TensorProduct(fiatx.CellType.quadrilateral, order)
    pts = [[0, 0], [1, 0], [0, 1], [1, 1]]
    x = tp.tabulate_basis(pts)
    print(x)

    tp = fiatx.TensorProduct(fiatx.CellType.hexahedron, order)
    pts = [[0, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1]]
    x = tp.tabulate_basis(pts)
    print(x)
