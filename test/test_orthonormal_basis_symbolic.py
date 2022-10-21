# Copyright (c) 2020 Chris Richardson & Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import sympy
import basix
import numpy as np


def P_interval(n, x):
    r = []
    for i in range(n + 1):
        p = x ** i
        for j in r:
            p -= (p * j).integrate((x, 0, 1)) * j
        p /= sympy.sqrt((p * p).integrate((x, 0, 1)))
        r.append(p)
    return r


def test_symbolic_interval():
    n = 7
    nderiv = 7

    x = sympy.Symbol("x")
    wd = P_interval(n, x)

    cell = basix.CellType.interval
    pts0 = basix.create_lattice(cell, 10, basix.LatticeType.equispaced, True)
    wtab = basix._basixcpp.tabulate_polynomial_set(cell, n, nderiv, pts0)

    for k in range(nderiv + 1):
        wsym = np.zeros_like(wtab[k])
        for i in range(n + 1):
            for j, p in enumerate(pts0):
                wsym[i, j] = wd[i].subs(x, p[0])
            wd[i] = sympy.diff(wd[i], x)
        assert np.allclose(wtab[k], wsym)


def test_symbolic_quad():
    n = 5
    nderiv = 5

    idx = basix.index

    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    w = [wx * wy for wx in P_interval(n, x) for wy in P_interval(n, y)]

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
                    wsym[i, j] = wd.subs([(x, p[0]), (y, p[1])])
            assert np.allclose(wtab[idx(kx, ky)], wsym)
