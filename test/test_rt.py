# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import fiatx
import pytest


@pytest.mark.parametrize("celltype", [fiatx.CellType.triangle,
                                      fiatx.CellType.tetrahedron])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_rt(celltype, order):
    rt = fiatx.RaviartThomas(celltype, order)
    print(rt)
