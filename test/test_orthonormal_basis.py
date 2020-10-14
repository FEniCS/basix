# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import libtab
import pytest
import numpy as np


@pytest.mark.parametrize("order", [1, 2, 3])
def test_quad(order):
    pts = libtab.create_lattice(libtab.CellType.interval, 1, True)
    Lpts, Lwts = libtab.make_quadrature(pts, order + 2)
    Qwts = []
    Qpts = []
    for p, u in zip(Lpts, Lwts):
        for q, v in zip(Lpts, Lwts):
            Qpts.append([p[0], q[0]])
            Qwts.append(u*v)
    basis = libtab.tabulate_polynomial_set(libtab.CellType.quadrilateral,
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
    pts = libtab.create_lattice(libtab.CellType.interval, 1, True)
    Lpts, Lwts = libtab.make_quadrature(pts, order + 4)
    Qwts = []
    Qpts = []
    for p, u in zip(Lpts, Lwts):
        for q, v in zip(Lpts, Lwts):
            for r, w in zip(Lpts, Lwts):
                sc = (1.0 - r[0])
                Qpts.append([p[0]*sc, q[0]*sc, r[0]])
                Qwts.append(u*v*sc*sc*w)
    basis = libtab.tabulate_polynomial_set(libtab.CellType.pyramid,
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
    pts = libtab.create_lattice(libtab.CellType.interval, 1, True)
    Lpts, Lwts = libtab.make_quadrature(pts, order + 2)
    Qwts = []
    Qpts = []
    for p, u in zip(Lpts, Lwts):
        for q, v in zip(Lpts, Lwts):
            for r, w in zip(Lpts, Lwts):
                Qpts.append([p[0], q[0], r[0]])
                Qwts.append(u*v*w)
    basis = libtab.tabulate_polynomial_set(libtab.CellType.hexahedron,
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
    pts = libtab.create_lattice(libtab.CellType.triangle, 1, True)
    Tpts, Twts = libtab.make_quadrature(pts, order + 2)
    pts = libtab.create_lattice(libtab.CellType.interval, 1, True)
    Lpts, Lwts = libtab.make_quadrature(pts, order + 2)
    Qwts = []
    Qpts = []
    for p, u in zip(Tpts, Twts):
        for q, v in zip(Lpts, Lwts):
            Qpts.append([p[0], p[1], q[0]])
            Qwts.append(u*v)
    basis = libtab.tabulate_polynomial_set(libtab.CellType.prism, order, 0, Qpts)[0]
    ndofs = basis.shape[1]

    mat = np.zeros((ndofs, ndofs))
    for i in range(ndofs):
        for j in range(ndofs):
            mat[i, j] = sum(basis[:, i] * basis[:, j] * Qwts)

    np.set_printoptions(suppress=True)
    print(mat)
    assert(np.isclose(mat * 8.0, np.eye(mat.shape[0])).all())


@pytest.mark.parametrize("cell_type", [libtab.CellType.interval,
                                       libtab.CellType.triangle,
                                       libtab.CellType.tetrahedron])
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_cell(cell_type, order):

    pts = libtab.create_lattice(cell_type, 1, True)
    Qpts, Qwts = libtab.make_quadrature(pts, order + 2)
    basis = libtab.tabulate_polynomial_set(cell_type, order, 0, Qpts)[0]
    ndofs = basis.shape[1]
    mat = np.zeros((ndofs, ndofs))
    for i in range(ndofs):
        for j in range(ndofs):
            mat[i, j] = sum(basis[:, i] * basis[:, j] * Qwts)

    np.set_printoptions(suppress=True)
    print(mat)
    fac = 2 ** pts.shape[0] / 2
    assert(np.isclose(mat * fac, np.eye(mat.shape[0])).all())
