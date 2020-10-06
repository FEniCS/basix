# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import sympy
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


def test_symbolic_interval():
    n = 7
    nderiv = 4

    from sympy import S
    x = sympy.Symbol("x")
    x0 = x * S(2) - S(1)
    r = [S(1) for i in range(n + 1)]

    for p in range(1, n + 1):
        a = S(1) - sympy.Rational(1, p)
        r[p] = x0 * r[p - 1] * (a + S(1))
        if p > 1:
            r[p] = r[p] - r[p - 2] * a
    for p in range(n + 1):
        r[p] *= sympy.sqrt(p + sympy.Rational(1, 2))

    cell = fiatx.CellType.interval
    pts0 = fiatx.create_lattice(cell, 10, True)
    w = fiatx.tabulate_polynomial_set_deriv(cell, n, nderiv, pts0)

    wsym = np.zeros_like(w[0])

    for i in range(n + 1):
        for j, p in enumerate(pts0):
            wsym[j, i] = r[i].subs(x, p[0])

    assert(np.isclose(w[0], wsym).all())

    for k in range(1, 5):
        for i in range(n+1):
            r[i] = sympy.diff(r[i], x)

        wsym = np.zeros_like(w[0])
        for i in range(n + 1):
            for j, p in enumerate(pts0):
                wsym[j, i] = r[i].subs(x, p[0])

        assert(np.isclose(w[k], wsym).all())


def jrc(a, n):
    an = sympy.Rational((a + 2 * n + 1) * (a + 2 * n + 2),
                        2 * (n + 1) * (a + n + 1))
    bn = sympy.Rational(a * a * (a + 2 * n + 1),
                        2 * (n + 1) * (a + n + 1) * (a + 2 * n))
    cn = sympy.Rational(n * (a + n) * (a + 2 * n + 2),
                        (n + 1) * (a + n + 1) * (a + 2 * n))
    return (an, bn, cn)


def test_symbolic_triangle():
    n = 5
    nderiv = 4

    def idx(p, q):
        return (p + q + 1) * (p + q) // 2 + q

    from sympy import S
    m = (n + 1) * (n + 2) // 2
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    x0 = x * S(2) - S(1)
    y0 = y * S(2) - S(1)
    f3 = (S(1) - y0)**2 / S(4)
    r = [S(1) for i in range(m)]

    np.set_printoptions(linewidth=200)
    for p in range(1, n + 1):
        a = sympy.Rational(2 * p - 1, p)
        r[idx(p, 0)] = (x0 + (y0 + S(1))/S(2)) \
            * r[idx(p - 1, 0)] * a
        if p > 1:
            r[idx(p, 0)] -= f3 * r[idx(p - 2, 0)] * (a - S(1))

    for p in range(n):
        r[idx(p, 1)] = r[idx(p, 0)] * (y0 * sympy.Rational(3 + 2 * p, 2)
                                       + sympy.Rational(1 + 2 * p, 2))
        for q in range(1, n - p):
            a1, a2, a3 = jrc(2 * p + 1, q)
            r[idx(p, q + 1)] = r[idx(p, q)] * (y0 * a1 + a2) \
                - r[idx(p, q - 1)] * a3

    for p in range(n + 1):
        for q in range(n - p + 1):
            r[idx(p, q)] *= sympy.sqrt(sympy.Rational(2*p + 1, 2)
                                       * S(p + q + 1))

    cell = fiatx.CellType.triangle
    pts0 = fiatx.create_lattice(cell, 3, True)
    w = fiatx.tabulate_polynomial_set_deriv(cell, n, nderiv, pts0)

    wsym = np.zeros_like(w[0])

    for i in range(m):
        for j, p in enumerate(pts0):
            wsym[j, i] = r[i].subs([(x, p[0]), (y, p[1])])

    assert(np.isclose(w[0], wsym).all())

    rd = r.copy()
    for kx in range(nderiv):
        for ky in range(0, nderiv - kx):
            for i in range(m):
                rd[i] = sympy.diff(r[i], x, kx, y, ky)

            wsym = np.zeros_like(w[0])
            for i in range(m):
                for j, p in enumerate(pts0):
                    wsym[j, i] = rd[i].subs([(x, p[0]), (y, p[1])])

            assert(np.isclose(w[idx(kx, ky)], wsym).all())


def test_symbolic_tetrahedron():
    n = 4
    nderiv = 4

    def idx(p, q, r):
        return ((p + q + r) * (p + q + r + 1) * (p + q + r + 2) // 6
                + (q + r) * (q + r + 1) // 2 + r)

    from sympy import S
    m = (n + 1) * (n + 2) * (n + 3) // 6
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    x0 = x * S(2) - S(1)
    y0 = y * S(2) - S(1)
    z0 = z * S(2) - S(1)
    f2 = (y0 + z0)**2 / S(4)
    f3 = y0 + (S(1) + z0) / S(2)
    f4 = (S(1) - z0) / S(2)
    f5 = f4 * f4

    w = [S(1) for i in range(m)]

    np.set_printoptions(linewidth=200, suppress=True, precision=3)
    for p in range(1, n + 1):
        a = sympy.Rational(2 * p - 1, p)
        w[idx(p, 0, 0)] = (x0 + S(1) + (y0 + z0)/S(2)) \
            * w[idx(p - 1, 0, 0)] * a
        if p > 1:
            w[idx(p, 0, 0)] -= f2 * w[idx(p - 2, 0, 0)] * (a - S(1))

    for p in range(n):
        w[idx(p, 1, 0)] = w[idx(p, 0, 0)] * \
            ((S(1) + y0)*S(p) +
             (S(2) + y0 * S(3) + z0) / S(2))
        for q in range(1, n - p):
            aq, bq, cq = jrc(2 * p + 1, q)
            w[idx(p, q + 1, 0)] = w[idx(p, q, 0)] * (f3 * aq + f4 * bq) \
                - w[idx(p, q - 1, 0)] * f5 * cq

    for p in range(n):
        for q in range(n - p):
            w[idx(p, q, 1)] = w[idx(p, q, 0)] * \
                (S(1 + p + q) + z0 * S(2 + p + q))

    for p in range(n - 1):
        for q in range(n - p - 1):
            for r in range(n - p - q):
                ar, br, cr = jrc(2 * p + 2 * q + 2, r)
                w[idx(p, q, r + 1)] = w[idx(p, q, r)] * \
                    (z0 * ar + br) - w[idx(p, q, r - 1)] * cr

    for p in range(n + 1):
        for q in range(n - p + 1):
            for r in range(n - p - q + 1):
                w[idx(p, q, r)] *= \
                    sympy.sqrt(sympy.Rational(2 * p + 1, 2)
                               * S(p + q + 1)
                               * sympy.Rational(2 * p + 2 * q + 2 * r + 3, 2))

    cell = fiatx.CellType.tetrahedron
    pts0 = fiatx.create_lattice(cell, 2, True)
    wtab = fiatx.tabulate_polynomial_set_deriv(cell, n, nderiv, pts0)

    wsym = np.zeros_like(wtab[0])

    for i in range(m):
        for j, p in enumerate(pts0):
            wsym[j, i] = w[i].subs([(x, p[0]), (y, p[1]), (z, p[2])])

    assert(np.isclose(wtab[0], wsym).all())

    for k in range(nderiv + 1):
        for q in range(k + 1):
            for kx in range(q + 1):
                ky = q - kx
                kz = k - q
                print((kx, ky, kz))

                wd = []
                wsym = np.zeros_like(wtab[0])
                for i in range(m):
                    wd += [sympy.diff(w[i], x, kx, y, ky, z, kz)]
                    for j, p in enumerate(pts0):
                        wsym[j, i] = wd[i].subs([(x, p[0]),
                                                 (y, p[1]),
                                                 (z, p[2])])

                assert(np.isclose(wtab[idx(kx, ky, kz)], wsym).all())
