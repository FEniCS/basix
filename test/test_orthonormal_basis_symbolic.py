# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import sympy
import basix
import numpy as np


def P_interval(n, x):
    from sympy import S
    r = [sympy.sqrt(p + sympy.Rational(1, 2))
         * sympy.legendre(p, x * S(2) - S(1))
         for p in range(n + 1)]
    return(r)


def test_symbolic_interval():
    n = 7
    nderiv = 7

    x = sympy.Symbol("x")
    w = P_interval(n, x)

    cell = basix.CellType.interval
    pts0 = basix.create_lattice(cell, 10, basix.LatticeType.equispaced, True)
    wtab = basix._basixcpp.tabulate_polynomial_set(cell, n, nderiv, pts0)

    wd = [w[i] for i in range(n + 1)]
    for k in range(nderiv + 1):
        wsym = np.zeros_like(wtab[k])
        for i in range(n + 1):
            for j, p in enumerate(pts0):
                wsym[j, i] = wd[i].subs(x, p[0])
            wd[i] = sympy.diff(wd[i], x)
        assert(np.isclose(wtab[k], wsym).all())


def test_symbolic_quad():
    n = 2
    nderiv = 2

    idx = basix.index

    x = sympy.Symbol("x")
    wx = P_interval(n, x)
    y = sympy.Symbol("y")
    wy = P_interval(n, y)

    w = []
    for i in range(n + 1):
        for j in range(n + 1):
            w += [wx[i] * wy[j]]

    m = (n + 1)**2
    cell = basix.CellType.quadrilateral
    pts0 = basix.create_lattice(cell, 2, basix.LatticeType.equispaced, True)
    wtab = basix._basixcpp.tabulate_polynomial_set(cell, n, nderiv, pts0)

    for kx in range(nderiv):
        for ky in range(0, nderiv - kx):

            wsym = np.zeros_like(wtab[0])
            for i in range(m):
                wd = sympy.diff(w[i], x, kx, y, ky)
                for j, p in enumerate(pts0):
                    wsym[j, i] = wd.subs([(x, p[0]), (y, p[1])])
            assert(np.isclose(wtab[idx(kx, ky)], wsym).all())


def test_symbolic_triangle():
    n = 5
    nderiv = 4

    idx = basix.index

    from sympy import S
    m = (n + 1) * (n + 2) // 2
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    x0 = x * S(2) - S(1)
    y0 = y * S(2) - S(1)
    w = [None] * m

    zeta = (S(2) * x0 + y0 + S(1)) / (S(1) - y0)
    for p in range(n + 1):
        for q in range(n - p + 1):
            w[idx(p, q)] = sympy.sqrt(S(2 * p + 1) * S(p + q + 1) / S(2)) \
                * sympy.cancel(sympy.legendre(p, zeta)
                               * ((S(1) - y0) / S(2))**p) \
                * sympy.jacobi(S(q), S(2 * p + 1), S(0), y0)

    np.set_printoptions(linewidth=200)

    cell = basix.CellType.triangle
    pts0 = basix.create_lattice(cell, 3, basix.LatticeType.equispaced, True)
    wtab = basix._basixcpp.tabulate_polynomial_set(cell, n, nderiv, pts0)

    for kx in range(nderiv):
        for ky in range(0, nderiv - kx):
            wsym = np.zeros_like(wtab[0])
            for i in range(m):
                wd = sympy.diff(w[i], x, kx, y, ky)
                for j, p in enumerate(pts0):
                    wsym[j, i] = wd.subs([(x, p[0]), (y, p[1])])

            assert(np.isclose(wtab[idx(kx, ky)], wsym).all())


def test_symbolic_tetrahedron():
    n = 4
    nderiv = 4

    idx = basix.index

    from sympy import S
    m = (n + 1) * (n + 2) * (n + 3) // 6
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    x0 = x * S(2) - S(1)
    y0 = y * S(2) - S(1)
    z0 = z * S(2) - S(1)

    w = [None] * m

    np.set_printoptions(linewidth=200, suppress=True, precision=3)

    zeta = S(2) * (S(1) + x0) / (y0 + z0) + S(1)
    xi = S(2) * (S(1) + y0) / (S(1) - z0) - S(1)

    for p in range(n + 1):
        for q in range(n - p + 1):
            for r in range(n - p - q + 1):
                w[idx(p, q, r)] = sympy.cancel(
                    sympy.legendre(p, zeta) * ((y0 + z0) / S(2))**p)
                w[idx(p, q, r)] *= sympy.cancel(
                    sympy.jacobi(S(q), S(2 * p + 1), 0, xi)
                    * ((S(1) - z0) / S(2))**q)
                w[idx(p, q, r)] *= sympy.jacobi(S(r),
                                                S(2 * p + 2 * q + 2), 0, z0)
                w[idx(p, q, r)] *= sympy.sqrt(
                    S((2 * p + 1) * (p + q + 1)
                      * (2 * p + 2 * q + 2 * r + 3))) / S(2)

    cell = basix.CellType.tetrahedron
    pts0 = basix.create_lattice(cell, 2, basix.LatticeType.equispaced, True)
    wtab = basix._basixcpp.tabulate_polynomial_set(cell, n, nderiv, pts0)

    for k in range(nderiv + 1):
        for q in range(k + 1):
            for kx in range(q + 1):
                ky = q - kx
                kz = k - q
                wsym = np.zeros_like(wtab[0])
                for i in range(m):
                    wd = sympy.diff(w[i], x, kx, y, ky, z, kz)
                    for j, p in enumerate(pts0):
                        wsym[j, i] = wd.subs([(x, p[0]),
                                              (y, p[1]),
                                              (z, p[2])])

                assert(np.isclose(wtab[idx(kx, ky, kz)], wsym).all())


def test_symbolic_pyramid():
    np.set_printoptions(linewidth=200, suppress=True, precision=2)
    n = 3
    nderiv = 3

    idx = basix.index

    def pyr_idx(p, q, r):
        rv = n - r + 1
        r0 = r * (n + 1) * (n - r + 2) + (2 * r - 1) * (r - 1) * r // 6
        idx = r0 + p * rv + q
        return idx

    from sympy import S
    m = (n + 1) * (n + 2) * (2 * n + 3) // 6
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    x0 = x * S(2) - S(1)
    y0 = y * S(2) - S(1)
    z0 = z * S(2) - S(1)

    w = [None] * m
    zetax = (S(2) * x0 + S(1) + z0) / (S(1) - z0)
    zetay = (S(2) * y0 + S(1) + z0) / (S(1) - z0)
    for r in range(n + 1):
        for p in range(n - r + 1):
            for q in range(n - r + 1):
                w[pyr_idx(p, q, r)] = sympy.cancel(
                    sympy.legendre(p, zetax) * ((S(1) - z0) / S(2))**p)
                w[pyr_idx(p, q, r)] *= sympy.cancel(
                    sympy.legendre(q, zetay) * ((S(1) - z0) / S(2))**q)
                w[pyr_idx(p, q, r)] *= \
                    sympy.jacobi(r, S(2 * p + 2 * q + 2), S(0), z0)
                w[pyr_idx(p, q, r)] *= \
                    sympy.sqrt(S((2 * q + 1) * (2 * p + 1)
                                 * (2 * p + 2 * q + 2 * r + 3)) / S(8))

    cell = basix.CellType.pyramid
    pts0 = basix.create_lattice(cell, 1, basix.LatticeType.equispaced, True)
    wtab = basix._basixcpp.tabulate_polynomial_set(cell, n, nderiv, pts0)

    for k in range(nderiv + 1):
        for q in range(k + 1):
            for kx in range(q + 1):
                ky = q - kx
                kz = k - q
                print((kx, ky, kz))

                wsym = np.zeros_like(wtab[0])
                for i in range(m):
                    wd = sympy.diff(w[i], x, kx, y, ky, z, kz)
                    for j, p in enumerate(pts0):
                        wsym[j, i] = wd.subs([(x, p[0]),
                                              (y, p[1]),
                                              (z, p[2])])
                assert(np.isclose(wtab[idx(kx, ky, kz)], wsym).all())
