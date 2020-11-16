# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import libtab
import pytest
import numpy as np


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("celltype", [libtab.CellType.quadrilateral,
                                      libtab.CellType.hexahedron,
                                      libtab.CellType.pyramid,
                                      libtab.CellType.prism])
def test_tp(order, celltype):
    np.set_printoptions(suppress=True)
    tp = libtab.TensorProduct(celltype, order)
    pts = libtab.create_lattice(celltype, 5,
                                libtab.LatticeType.equispaced, True)
    w = tp.tabulate(0, pts)[0]
    assert(np.isclose(np.sum(w, axis=1), 1.0).all())
