# Copyright (c) 2022 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import numpy as np
import pytest


def create_custom_p1():
    wcoeffs = np.eye(3)
    z = np.zeros((0, 2))
    x = [
        [np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])],
        [z, z, z],
        [z],
        [],
    ]
    z = np.zeros((0, 1, 0, 1))
    M = [[np.array([[[[1.0]]]]), np.array([[[[1.0]]]]), np.array([[[[1.0]]]])], [z, z, z], [z], []]

    return basix.create_custom_element(
        basix.CellType.triangle,
        [],
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.identity,
        basix.SobolevSpace.H1,
        False,
        1,
        1,
        basix.PolysetType.standard,
    )


@pytest.fixture
def p1_custom():
    return create_custom_p1()


@pytest.fixture
def p1_custom_again():
    return create_custom_p1()


@pytest.fixture
def cr_custom():
    wcoeffs = np.eye(3)
    z = np.zeros((0, 2))
    x = [
        [z, z, z],
        [np.array([[0.5, 0.5]]), np.array([[0.0, 0.5]]), np.array([[0.5, 0.0]])],
        [z],
        [],
    ]
    z = np.zeros((0, 1, 0, 1))
    M = [[z, z, z], [np.array([[[[1.0]]]]), np.array([[[[1.0]]]]), np.array([[[[1.0]]]])], [z], []]

    return basix.create_custom_element(
        basix.CellType.triangle,
        [],
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.identity,
        basix.SobolevSpace.L2,
        False,
        1,
        1,
        basix.PolysetType.standard,
    )


@pytest.fixture
def cr():
    return basix.create_element(basix.ElementFamily.CR, basix.CellType.triangle, 1)


@pytest.fixture
def p1():
    return basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 1)


@pytest.fixture
def p1_again():
    return basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 1)


@pytest.fixture
def rt1():
    return basix.create_element(basix.ElementFamily.RT, basix.CellType.triangle, 1)


@pytest.fixture
def p1_quad():
    return basix.create_element(basix.ElementFamily.P, basix.CellType.quadrilateral, 1)


@pytest.fixture
def p4_gll():
    return basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, 4, basix.LagrangeVariant.gll_warped
    )


@pytest.fixture
def p4_equi():
    return basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, 4, basix.LagrangeVariant.equispaced
    )


@pytest.fixture
def p1_f32():
    return basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 1, dtype=np.float32)


@pytest.fixture
def p1_f64():
    return basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 1, dtype=np.float64)


@pytest.fixture
def p1_dofs():
    return basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, 1, dof_ordering=[0, 1, 2]
    )


@pytest.fixture
def p1_dofs_again():
    return basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, 1, dof_ordering=[0, 1, 2]
    )


@pytest.fixture
def p1_reverse_dofs():
    return basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, 1, dof_ordering=[2, 1, 0]
    )


def test_equal_same_element(p1, p1_again):
    assert p1 == p1
    assert p1 == p1_again


def test_nonequal_degree(p1, p4_gll):
    assert p1 != p4_gll


def test_nonequal_degree_equi(p1, p4_equi):
    assert p1 != p4_equi


def test_nonequal_variants(p4_gll, p4_equi):
    assert p4_gll != p4_equi


def test_nonequal_celltype(p1, p1_quad):
    assert p1 != p1_quad


def test_nonequal_family(p1, rt1):
    assert p1 != rt1


def test_nonequal_dtype(p1_f64, p1_f32):
    assert p1_f64 != p1_f32


def test_equal_dof_ordering(p1, p1_dofs, p1_dofs_again):
    assert p1_dofs == p1_dofs_again


def test_nonequal_dof_ordering(p1, p1_reverse_dofs, p1_dofs):
    assert p1 != p1_dofs
    assert p1 != p1_reverse_dofs
    assert p1_dofs != p1_reverse_dofs


def test_equal_custom(p1_custom, p1_custom_again):
    assert p1_custom == p1_custom_again


def test_nonequal_custom(p1_custom, cr_custom, cr, p1):
    assert p1_custom != cr_custom
    assert p1_custom != cr
    assert p1_custom != p1
    assert cr_custom != cr
