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
    assert len(element.interpolation_info) == element.dim
