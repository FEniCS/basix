# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import libtab
import numpy
import pytest


@pytest.mark.parametrize("celltype", [libtab.CellType.interval,
                                      libtab.CellType.triangle,
                                      libtab.CellType.tetrahedron])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_lagrange(celltype, order):
    lagrange = libtab.Lagrange(celltype, order)

    pts = libtab.create_lattice(celltype, 6, True)
    w = lagrange.tabulate(0, pts)[0]
    assert(numpy.isclose(numpy.sum(w, axis=1), 1.0).all())
