# Copyright (c) 2020 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import libtab
import pytest


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("celltype", ["interval", "triangle", "tetrahedron"])
@pytest.mark.parametrize("element_type", ["Lagrange"])
def test_interpolation(celltype, n, element_type):
    element = libtab.create_element(element_type, celltype, n)
    assert element.interpolation_info[0].shape[0] == element.dim
    assert element.interpolation_info[0].shape[1] == element.interpolation_info[1].shape[0]
    assert element.interpolation_info[1].shape[1] == len(libtab.topology(element.cell_type)) - 1
