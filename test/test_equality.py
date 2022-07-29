# Copyright (c) 2022 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import numpy as np


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


def test_custom_element_equality():
    p1 = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 1)
    cr = basix.create_element(basix.ElementFamily.CR, basix.CellType.triangle, 1)

    wcoeffs = np.eye(3)
    z = np.zeros((0, 2))
    x = [[np.array([[0., 0.]]), np.array([[1., 0.]]), np.array([[0., 1.]])], [z, z, z], [z], []]
    z = np.zeros((0, 1, 0, 1))
    M = [[np.array([[[[1.]]]]), np.array([[[[1.]]]]), np.array([[[[1.]]]])], [z, z, z], [z], []]

    p1_custom = basix.create_custom_element(
        basix.CellType.triangle, [], wcoeffs,
        x, M, 0, basix.MapType.identity, False, 1, 1)
    p1_custom_again = basix.create_custom_element(
        basix.CellType.triangle, [], wcoeffs,
        x, M, 0, basix.MapType.identity, False, 1, 1)

    wcoeffs = np.eye(3)
    z = np.zeros((0, 2))
    x = [[z, z, z], [np.array([[.5, .5]]), np.array([[0., .5]]), np.array([[.5, 0.]])], [z], []]
    z = np.zeros((0, 1, 0, 1))
    M = [[z, z, z], [np.array([[[[1.]]]]), np.array([[[[1.]]]]), np.array([[[[1.]]]])], [z], []]

    cr_custom = basix.create_custom_element(
        basix.CellType.triangle, [], wcoeffs,
        x, M, 0, basix.MapType.identity, False, 1, 1)

    assert p1_custom == p1_custom_again
    assert p1_custom != cr_custom
    assert p1_custom != cr
    assert p1_custom != p1
    assert cr_custom != cr
