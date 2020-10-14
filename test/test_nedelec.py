# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import libtab
import pytest


@pytest.mark.parametrize("order", [1, 2, 3])
def test_nedelec2d(order):
    ned2 = libtab.Nedelec(libtab.CellType.triangle, order)
    pts = libtab.create_lattice(libtab.CellType.triangle, 2, True)
    w = ned2.tabulate(0, pts)[0]
    print(w.shape)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_nedelec3d(order):
    ned3 = libtab.Nedelec(libtab.CellType.tetrahedron, order)
    pts = libtab.create_lattice(libtab.CellType.tetrahedron, 2, True)
    w = ned3.tabulate(0, pts)[0]
    print(w.shape)
