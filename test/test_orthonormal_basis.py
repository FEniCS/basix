# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import fiatx
import pytest
import numpy as np


@pytest.mark.parametrize("order", [2, 3, 4, 5])
def test_quad(order):
    basis = fiatx.compute_polynomial_set(fiatx.CellType.quadrilateral, order)
    cell = fiatx.Cell(fiatx.CellType.interval)
    pts = cell.create_lattice(1, True)
    Lpts, Lwts = fiatx.make_quadrature(pts, order + 2)
    Qwts = []
    Qpts = []
    for p, u in zip(Lpts, Lwts):
        for q, v in zip(Lpts, Lwts):
            Qpts.append([p[0], q[0]])
            Qwts.append(u*v)

    mat = np.zeros((len(basis), len(basis)))
    for i, p in enumerate(basis):
        for j, q in enumerate(basis):
            w = p.tabulate(Qpts) * q.tabulate(Qpts)
            s = 0.0
            for val, wt in zip(w, Qwts):
                s += val*wt
            mat[i, j] = s

    np.set_printoptions(suppress=True)
    print(mat, np.eye(mat.shape[0]))
    assert(np.isclose(mat * 4.0, np.eye(mat.shape[0])).all())


@pytest.mark.parametrize("order", [1, 2, 3])
def test_pyramid(order):
    basis = fiatx.compute_polynomial_set(fiatx.CellType.pyramid, order)
    cell = fiatx.Cell(fiatx.CellType.interval)
    pts = cell.create_lattice(1, True)
    Lpts, Lwts = fiatx.make_quadrature(pts, order + 4)
    Qwts = []
    Qpts = []
    for p, u in zip(Lpts, Lwts):
        for q, v in zip(Lpts, Lwts):
            for r, w in zip(Lpts, Lwts):
                sc = (1.0 - r[0])
                Qpts.append([p[0]*sc, q[0]*sc, r[0]])
                Qwts.append(u*v*sc*sc*w)

    mat = np.zeros((len(basis), len(basis)))
    for i, p in enumerate(basis):
        for j, q in enumerate(basis):
            w = p.tabulate(Qpts) * q.tabulate(Qpts)
            s = 0.0
            for val, wt in zip(w, Qwts):
                s += val*wt
            mat[i, j] = s

    np.set_printoptions(suppress=True, linewidth=220)
    print(mat)
    assert(np.isclose(mat * 8.0, np.eye(mat.shape[0])).all())


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_hex(order):
    basis = fiatx.compute_polynomial_set(fiatx.CellType.hexahedron, order)
    cell = fiatx.Cell(fiatx.CellType.interval)
    pts = cell.create_lattice(1, True)
    Lpts, Lwts = fiatx.make_quadrature(pts, order + 2)
    Qwts = []
    Qpts = []
    for p, u in zip(Lpts, Lwts):
        for q, v in zip(Lpts, Lwts):
            for r, w in zip(Lpts, Lwts):
                Qpts.append([p[0], q[0], r[0]])
                Qwts.append(u*v*w)

    mat = np.zeros((len(basis), len(basis)))
    for i, p in enumerate(basis):
        for j, q in enumerate(basis):
            w = p.tabulate(Qpts) * q.tabulate(Qpts)
            s = 0.0
            for val, wt in zip(w, Qwts):
                s += val*wt
            mat[i, j] = s

    np.set_printoptions(suppress=True)
    print(mat)
    assert(np.isclose(mat * 8.0, np.eye(mat.shape[0])).all())


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_prism(order):
    basis = fiatx.compute_polynomial_set(fiatx.CellType.prism, order)
    tri_cell = fiatx.Cell(fiatx.CellType.triangle)
    pts = tri_cell.create_lattice(1, True)
    Tpts, Twts = fiatx.make_quadrature(pts, order + 2)
    line_cell = fiatx.Cell(fiatx.CellType.interval)
    pts = line_cell.create_lattice(1, True)
    Lpts, Lwts = fiatx.make_quadrature(pts, order + 2)
    Qwts = []
    Qpts = []
    for p, u in zip(Tpts, Twts):
        for q, v in zip(Lpts, Lwts):
            Qpts.append([p[0], p[1], q[0]])
            Qwts.append(u*v)

    mat = np.zeros((len(basis), len(basis)))
    for i, p in enumerate(basis):
        for j, q in enumerate(basis):
            w = p.tabulate(Qpts) * q.tabulate(Qpts)
            s = 0.0
            for val, wt in zip(w, Qwts):
                s += val*wt
            mat[i, j] = s

    np.set_printoptions(suppress=True)
    print(mat)
    assert(np.isclose(mat * 8.0, np.eye(mat.shape[0])).all())


@pytest.mark.parametrize("cell_type", [fiatx.CellType.interval, fiatx.CellType.triangle, fiatx.CellType.tetrahedron])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_cell(cell_type, order):
    basis = fiatx.compute_polynomial_set(cell_type, order)
    cell = fiatx.Cell(cell_type)
    pts = cell.create_lattice(1, True)
    Qpts, Qwts = fiatx.make_quadrature(pts, order + 2)
    mat = np.zeros((len(basis), len(basis)))
    for i, p in enumerate(basis):
        for j, q in enumerate(basis):
            w = p.tabulate(Qpts) * q.tabulate(Qpts)
            s = 0.0
            for val, wt in zip(w, Qwts):
                s += val*wt
            mat[i, j] = s

    np.set_printoptions(suppress=True)
    print(mat)
    fac = 2 ** pts.shape[0] / 2
    assert(np.isclose(mat * fac, np.eye(mat.shape[0])).all())
