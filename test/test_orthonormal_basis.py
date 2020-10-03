# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import fiatx
import pytest
import numpy as np


@pytest.mark.parametrize("order", [1, 2, 3])
def test_quad(order):
    pts = fiatx.create_lattice(fiatx.CellType.interval, 1, True)
    Lpts, Lwts = fiatx.make_quadrature(pts, order + 2)
    Qwts = []
    Qpts = []
    for p, u in zip(Lpts, Lwts):
        for q, v in zip(Lpts, Lwts):
            Qpts.append([p[0], q[0]])
            Qwts.append(u*v)
    basis = fiatx.tabulate_polynomial_set(fiatx.CellType.quadrilateral,
                                          order, Qpts)
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
    pts = fiatx.create_lattice(fiatx.CellType.interval, 1, True)
    Lpts, Lwts = fiatx.make_quadrature(pts, order + 4)
    Qwts = []
    Qpts = []
    for p, u in zip(Lpts, Lwts):
        for q, v in zip(Lpts, Lwts):
            for r, w in zip(Lpts, Lwts):
                sc = (1.0 - r[0])
                Qpts.append([p[0]*sc, q[0]*sc, r[0]])
                Qwts.append(u*v*sc*sc*w)
    basis = fiatx.tabulate_polynomial_set(fiatx.CellType.pyramid,
                                          order, Qpts)
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
    pts = fiatx.create_lattice(fiatx.CellType.interval, 1, True)
    Lpts, Lwts = fiatx.make_quadrature(pts, order + 2)
    Qwts = []
    Qpts = []
    for p, u in zip(Lpts, Lwts):
        for q, v in zip(Lpts, Lwts):
            for r, w in zip(Lpts, Lwts):
                Qpts.append([p[0], q[0], r[0]])
                Qwts.append(u*v*w)
    basis = fiatx.tabulate_polynomial_set(fiatx.CellType.hexahedron,
                                          order, Qpts)
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
    pts = fiatx.create_lattice(fiatx.CellType.triangle, 1, True)
    Tpts, Twts = fiatx.make_quadrature(pts, order + 2)
    pts = fiatx.create_lattice(fiatx.CellType.interval, 1, True)
    Lpts, Lwts = fiatx.make_quadrature(pts, order + 2)
    Qwts = []
    Qpts = []
    for p, u in zip(Tpts, Twts):
        for q, v in zip(Lpts, Lwts):
            Qpts.append([p[0], p[1], q[0]])
            Qwts.append(u*v)
    basis = fiatx.tabulate_polynomial_set(fiatx.CellType.prism, order, Qpts)
    ndofs = basis.shape[1]

    mat = np.zeros((ndofs, ndofs))
    for i in range(ndofs):
        for j in range(ndofs):
            mat[i, j] = sum(basis[:, i] * basis[:, j] * Qwts)

    np.set_printoptions(suppress=True)
    print(mat)
    assert(np.isclose(mat * 8.0, np.eye(mat.shape[0])).all())


@pytest.mark.parametrize("cell_type", [fiatx.CellType.interval,
                                       fiatx.CellType.triangle,
                                       fiatx.CellType.tetrahedron])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_cell(cell_type, order):

    pts = fiatx.create_lattice(cell_type, 1, True)
    Qpts, Qwts = fiatx.make_quadrature(pts, order + 2)
    basis = fiatx.tabulate_polynomial_set(cell_type, order, Qpts)
    ndofs = basis.shape[1]
    mat = np.zeros((ndofs, ndofs))
    for i in range(ndofs):
        for j in range(ndofs):
            mat[i, j] = sum(basis[:, i] * basis[:, j] * Qwts)

    np.set_printoptions(suppress=True)
    print(mat)
    fac = 2 ** pts.shape[0] / 2
    assert(np.isclose(mat * fac, np.eye(mat.shape[0])).all())


def test_derivs_line():
    cell = fiatx.CellType.interval
    pts0 = fiatx.create_lattice(cell, 10, True)
    eps = 1e-6
    pts1 = pts0 - eps
    pts2 = pts0 + eps
    n = 3
    nderiv = 1
    w = fiatx.tabulate_polynomial_set_deriv(cell, n, nderiv, pts0)
    w1 = fiatx.tabulate_polynomial_set_deriv(cell, n, 0, pts1)[0]
    w2 = fiatx.tabulate_polynomial_set_deriv(cell, n, 0, pts2)[0]
    v = (w2 - w1)/2/eps
    assert(np.isclose(w[1], v).all())


def test_derivs_triangle():
    cell = fiatx.CellType.triangle
    pts0 = fiatx.create_lattice(cell, 10, True)
    eps = np.array([1e-6, 0.0])
    pts1 = pts0 - eps
    pts2 = pts0 + eps
    n = 3
    nderiv = 1
    w = fiatx.tabulate_polynomial_set_deriv(cell, n, nderiv, pts0)
    w1 = fiatx.tabulate_polynomial_set_deriv(cell, n, 0, pts1)[0]
    w2 = fiatx.tabulate_polynomial_set_deriv(cell, n, 0, pts2)[0]
    v = (w2 - w1)/2/eps[0]
    assert(np.isclose(w[1], v).all())
    eps = np.array([0.0, 1e-6])
    pts1 = pts0 - eps
    pts2 = pts0 + eps
    w1 = fiatx.tabulate_polynomial_set_deriv(cell, n, 0, pts1)[0]
    w2 = fiatx.tabulate_polynomial_set_deriv(cell, n, 0, pts2)[0]
    v = (w2 - w1)/2/eps[1]
    assert(np.isclose(w[2], v).all())
