# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import pytest
import numpy as np


def test_regge_tri():
    # Simplest element
    regge = basix.create_element(basix.ElementFamily.Regge, basix.CellType.triangle, 1)

    # tabulate at origin
    pts = [[0.0, 0.0]]
    w = regge.tabulate(0, pts)[0].reshape(-1, 2, 2)

    ref = np.array([[[-0.,  0.5], [0.5, -0.]],
                    [[0.,  0.5], [0.5, -0.]],
                    [[-0.,  1.], [1.,  2.]],
                    [[-0., -0.5], [-0.5, -1.]],
                    [[2.,  1.], [1., -0.]],
                    [[-1., -0.5], [-0.5,  0.]],
                    [[-0.,  0.], [0.,  0.]],
                    [[0., -0.], [-0., -0.]],
                    [[-0., -1.5], [-1.5,  0.]]])

    assert np.allclose(ref, w)


def test_regge_tri2():
    # Second order
    regge = basix.create_element(basix.ElementFamily.Regge, basix.CellType.triangle, 2)
    # tabulate at origin
    pts = [[0.0, 0.0]]
    w = regge.tabulate(0, pts)[0].reshape(-1, 2, 2)

    ref = np.array([[[0., -0.5], [-0.5,  0.]],
                    [[0., -0.5], [-0.5, -0.]],
                    [[-0., -0.5], [-0.5,  0.]],
                    [[-0.,  1.5], [1.5,  3.]],
                    [[0., -1.5], [-1.5, -3.]],
                    [[-0.,  0.5], [0.5,  1.]],
                    [[3.,  1.5], [1.5, -0.]],
                    [[-3., -1.5], [-1.5,  0.]],
                    [[1.,  0.5], [0.5, -0.]],
                    [[-0., -0.], [-0., -0.]],
                    [[0., -0.], [-0., -0.]],
                    [[0., -3.], [-3.,  0.]],
                    [[0., -0.], [-0., -0.]],
                    [[-0., -0.], [-0.,  0.]],
                    [[-0.,  2.], [2., -0.]],
                    [[-0.,  0.], [0.,  0.]],
                    [[0.,  0.], [0., -0.]],
                    [[0.,  2.], [2., -0.]]])
    assert(np.isclose(ref, w).all())


def test_regge_tet():
    # Simplest element
    regge = basix.create_element(basix.ElementFamily.Regge, basix.CellType.tetrahedron, 1)
    # tabulate at origin
    pts = [[0.0, 0.0, 0.0]]
    w = regge.tabulate(0, pts)[0].reshape(-1, 3, 3)

    ref = np.array([[[0.,  0.,  0.], [0.,  0.,  0.5], [0.,  0.5, -0.]],
                    [[-0.,  0., -0.], [0., -0.,  0.5], [-0.,  0.5,  0.]],
                    [[0., -0.,  0.5], [-0.,  0.,  0.], [0.5,  0.,  0.]],
                    [[-0.,  0.,  0.5], [0., -0.,  0.], [0.5,  0.,  0.]],
                    [[-0.,  0.5,  0.], [0.5, -0., -0.], [0., -0.,  0.]],
                    [[0.,  0.5,  0.], [0.5, -0.,  0.], [0.,  0.,  0.]],
                    [[0.,  0.,  1.], [0.,  0.,  1.], [1.,  1.,  2.]],
                    [[0., -0., -0.5], [-0.,  0., -0.5], [-0.5, -0.5, -1.]],
                    [[0.,  1., -0.], [1.,  2.,  1.], [-0.,  1., -0.]],
                    [[-0., -0.5, -0.], [-0.5, -1., -0.5], [-0., -0.5, -0.]],
                    [[2.,  1.,  1.], [1., -0.,  0.], [1.,  0.,  0.]],
                    [[-1., -0.5, -0.5], [-0.5, -0.,  0.], [-0.5,  0., -0.]],
                    [[-0.,  0., -0.], [0., -0., -0.], [-0., -0., -0.]],
                    [[-0., -0., -0.], [-0., -0., -0.], [-0., -0., -0.]],
                    [[-0.,  0., -0.], [0.,  0., -0.], [-0., -0., -0.]],
                    [[0., -0.,  0.], [-0., -0., -0.], [0., -0.,  0.]],
                    [[-0., -0., -0.], [-0., -0.,  0.], [-0.,  0.,  0.]],
                    [[-0., -0., -0.], [-0., -0., -1.5], [-0., -1.5, -0.]],
                    [[0., -0.,  0.], [-0.,  0., -0.], [0., -0.,  0.]],
                    [[0., -0., -0.], [-0., -0., -0.], [-0., -0.,  0.]],
                    [[-0.,  0., -1.5], [0., -0., -0.], [-1.5, -0., -0.]],
                    [[-0., -0., -0.], [-0.,  0., -0.], [-0., -0., -0.]],
                    [[0., -0., -0.], [-0., -0., -0.], [-0., -0.,  0.]],
                    [[-0., -1.5, -0.], [-1.5,  0., -0.], [-0., -0., -0.]]])

    assert(np.isclose(ref, w).all())


@pytest.mark.parametrize("degree", range(1, 4))
@pytest.mark.parametrize("cell", [basix.CellType.triangle, basix.CellType.tetrahedron])
def test_discontinuous_regge(degree, cell):
    e = basix.create_element(basix.ElementFamily.Regge, cell, degree)
    d_e = basix.create_element(basix.ElementFamily.Regge, cell, degree, True)

    pts = basix.create_lattice(cell, 5, basix.LatticeType.equispaced, True)

    assert np.allclose(e.tabulate(1, pts), d_e.tabulate(1, pts))

    for dofs in d_e.num_entity_dofs[:-1]:
        for d in dofs:
            assert d == 0
