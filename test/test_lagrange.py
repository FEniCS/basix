# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import fiatx
import pytest

@pytest.mark.parametrize("celltype", [fiatx.CellType.interval, fiatx.CellType.triangle, fiatx.CellType.tetrahedron])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_lagrange(celltype, order):
    lagrange = fiatx.Lagrange(celltype, order)
    print(lagrange)
