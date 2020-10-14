# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import libtab
import pytest


@pytest.mark.parametrize("celltype", [libtab.CellType.triangle,
                                      libtab.CellType.tetrahedron])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_rt(celltype, order):
    rt = libtab.RaviartThomas(celltype, order)
    print(rt)
