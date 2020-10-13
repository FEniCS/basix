# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import fiatx
import pytest


@pytest.mark.parametrize("order", [1, 2, 3])
def test_nedelec2d(order):
    ned2 = fiatx.Nedelec(fiatx.CellType.triangle, order)
    pts = fiatx.create_lattice(fiatx.CellType.triangle, 2, True)
    w = ned2.tabulate(0, pts)[0]
    print(w.shape)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_nedelec3d(order):
    ned3 = fiatx.Nedelec(fiatx.CellType.tetrahedron, order)
    pts = fiatx.create_lattice(fiatx.CellType.tetrahedron, 2, True)
    w = ned3.tabulate(0, pts)[0]
    print(w.shape)
