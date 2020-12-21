# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import pytest
import numpy as np


@pytest.mark.parametrize("order", [1, 2, 3])
def test_quad(order):
    pts = basix.create_lattice(basix.CellType.interval, 1, basix.LatticeType.equispaced, True)
    Lpts, Lwts = basix.make_quadrature(pts, order + 2)
    Qwts = []
    Qpts = []
    for p, u in zip(Lpts, Lwts):
        for q, v in zip(Lpts, Lwts):
            Qpts.append([p[0], q[0]])
            Qwts.append(u * v)
    basis = basix.tabulate_polynomial_set(basix.CellType.quadrilateral,
                                          order, 0, Qpts)[0]
    ndofs = basis.shape[1]

    mat = np.zeros((ndofs, ndofs))
    for i in range(ndofs):
        for j in range(ndofs):
            mat[i, j] = sum(basis[:, i] * basis[:, j] * Qwts)

    np.set_printoptions(suppress=True)
    print(mat, np.eye(mat.shape[0]))
    assert(np.isclose(mat * 4.0, np.eye(mat.shape[0])).all())


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_pyramid(order):
    pts = basix.create_lattice(basix.CellType.interval, 1, basix.LatticeType.equispaced, True)
    Lpts, Lwts = basix.make_quadrature(pts, order + 4)
    Qwts = []
    Qpts = []
    for p, u in zip(Lpts, Lwts):
        for q, v in zip(Lpts, Lwts):
            for r, w in zip(Lpts, Lwts):
                sc = (1.0 - r[0])
                Qpts.append([p[0] * sc, q[0] * sc, r[0]])
                Qwts.append(u * v * sc * sc * w)
    basis = basix.tabulate_polynomial_set(basix.CellType.pyramid,
                                          order, 0, Qpts)[0]
    ndofs = basis.shape[1]

    mat = np.zeros((ndofs, ndofs))
    for i in range(ndofs):
        for j in range(ndofs):
            mat[i, j] = sum(basis[:, i] * basis[:, j] * Qwts)

    np.set_printoptions(suppress=True, linewidth=220)
    print(mat)
    assert(np.isclose(mat * 8.0, np.eye(mat.shape[0])).all())


@pytest.mark.parametrize("order", [1, 2, 3])
def test_hex(order):
    pts = basix.create_lattice(basix.CellType.interval, 1, basix.LatticeType.equispaced, True)
    Lpts, Lwts = basix.make_quadrature(pts, order + 2)
    Qwts = []
    Qpts = []
    for p, u in zip(Lpts, Lwts):
        for q, v in zip(Lpts, Lwts):
            for r, w in zip(Lpts, Lwts):
                Qpts.append([p[0], q[0], r[0]])
                Qwts.append(u * v * w)
    basis = basix.tabulate_polynomial_set(basix.CellType.hexahedron,
                                          order, 0, Qpts)[0]
    ndofs = basis.shape[1]

    mat = np.zeros((ndofs, ndofs))
    for i in range(ndofs):
        for j in range(ndofs):
            mat[i, j] = sum(basis[:, i] * basis[:, j] * Qwts)

    np.set_printoptions(suppress=True)
    print(mat)
    assert(np.isclose(mat * 8.0, np.eye(mat.shape[0])).all())


@pytest.mark.parametrize("order", [1, 2, 3])
def test_prism(order):
    pts = basix.create_lattice(basix.CellType.triangle, 1, basix.LatticeType.equispaced, True)
    Tpts, Twts = basix.make_quadrature(pts, order + 2)
    pts = basix.create_lattice(basix.CellType.interval, 1, basix.LatticeType.equispaced, True)
    Lpts, Lwts = basix.make_quadrature(pts, order + 2)
    Qwts = []
    Qpts = []
    for p, u in zip(Tpts, Twts):
        for q, v in zip(Lpts, Lwts):
            Qpts.append([p[0], p[1], q[0]])
            Qwts.append(u * v)
    basis = basix.tabulate_polynomial_set(basix.CellType.prism,
                                          order, 0, Qpts)[0]
    ndofs = basis.shape[1]

    mat = np.zeros((ndofs, ndofs))
    for i in range(ndofs):
        for j in range(ndofs):
            mat[i, j] = sum(basis[:, i] * basis[:, j] * Qwts)

    np.set_printoptions(suppress=True)
    print(mat)
    assert(np.isclose(mat * 8.0, np.eye(mat.shape[0])).all())


@pytest.mark.parametrize("cell_type", [basix.CellType.interval,
                                       basix.CellType.triangle,
                                       basix.CellType.tetrahedron])
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_cell(cell_type, order):

    pts = basix.create_lattice(cell_type, 1, basix.LatticeType.equispaced, True)
    Qpts, Qwts = basix.make_quadrature(pts, order + 2)
    basis = basix.tabulate_polynomial_set(cell_type, order, 0, Qpts)[0]
    ndofs = basis.shape[1]
    mat = np.zeros((ndofs, ndofs))
    for i in range(ndofs):
        for j in range(ndofs):
            mat[i, j] = sum(basis[:, i] * basis[:, j] * Qwts)

    np.set_printoptions(suppress=True)
    print(mat)
    fac = 2 ** pts.shape[0] / 2
    assert(np.isclose(mat * fac, np.eye(mat.shape[0])).all())
