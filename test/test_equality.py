# Copyright (c) 2022 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix


def test_element_equality():
    p1 = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 1)
    p1_again = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 1)
    rt1 = basix.create_element(basix.ElementFamily.RT, basix.CellType.triangle, 1)
    p1_quad = basix.create_element(basix.ElementFamily.P, basix.CellType.quadrilateral, 1)
    p4_gll = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 4, basix.LagrangeVariant.gll_warped)
    p4_equi = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 4, basix.LagrangeVariant.equispaced)

    assert p1 == p1
    assert p1 == p1_again
    assert p1 != p4_gll
    assert p1 != p4_equi
    assert p4_gll != p4_equi
    assert p1 != p1_quad
    assert p1 != rt1
