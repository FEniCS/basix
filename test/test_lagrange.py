# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import fiatx
import numpy
import pytest


@pytest.mark.parametrize("celltype", [fiatx.CellType.interval,
                                      fiatx.CellType.triangle,
                                      fiatx.CellType.tetrahedron])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_lagrange(celltype, order):
    lagrange = fiatx.Lagrange(celltype, order)

    pts = fiatx.create_lattice(celltype, 6, True)
    w = lagrange.tabulate_basis(pts)
    assert(numpy.isclose(numpy.sum(w, axis=1), 1.0).all())
