# Copyright (c) 2020 Chris Richardson & Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import sympy

import basix


def P_interval(n, x):
    r = []
    for i in range(n + 1):
        p = x**i
        for j in r:
            p -= (p * j).integrate((x, 0, 1)) * j
        p /= sympy.sqrt((p * p).integrate((x, 0, 1)))
        r.append(p)
    return r


@pytest.mark.parametrize("n", range(8))
@pytest.mark.parametrize("nderiv", range(8))
def test_symbolic_interval(n, nderiv):
    x = sympy.Symbol("x")
    wd = P_interval(n, x)

    cell = basix.CellType.interval
    pts0 = basix.create_lattice(cell, 10, basix.LatticeType.equispaced, True)
    wtab = basix.polynomials.tabulate_polynomial_set(
        cell, basix.PolysetType.standard, n, nderiv, pts0
    )

    for k in range(nderiv + 1):
        wsym = np.zeros_like(wtab[k])
        for i in range(n + 1):
            for j, p in enumerate(pts0):
                wsym[i, j] = wd[i].subs(x, p[0])
            wd[i] = sympy.diff(wd[i], x)
        assert np.allclose(wtab[k], wsym)


@pytest.mark.parametrize("n", range(6))
@pytest.mark.parametrize("nderiv", range(6))
def test_symbolic_quad(n, nderiv):
    idx = basix.index

    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    w = [wx * wy for wx in P_interval(n, x) for wy in P_interval(n, y)]

    m = (n + 1) ** 2
    cell = basix.CellType.quadrilateral
    pts0 = basix.create_lattice(cell, 2, basix.LatticeType.equispaced, True)
    wtab = basix.polynomials.tabulate_polynomial_set(
        cell, basix.PolysetType.standard, n, nderiv, pts0
    )

    for kx in range(nderiv):
        for ky in range(0, nderiv - kx):
            wsym = np.zeros_like(wtab[0])
            for i in range(m):
                wd = sympy.diff(w[i], x, kx, y, ky)
                for j, p in enumerate(pts0):
                    wsym[i, j] = wd.subs([(x, p[0]), (y, p[1])])
            assert np.allclose(wtab[idx(kx, ky)], wsym)


def symbolic_pyramid(n):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")

    if n == 0:
        return [sympy.sqrt(3)]
    elif n == 1:
        return [
            sympy.sqrt(3),
            sympy.sqrt(5) * (4 * z - 1),
            3 * (2 * y + z - 1) / (1 - z),
            sympy.sqrt(15) * (4 * z - 1) * (2 * y + z - 1) / (1 - z),
            3 * (2 * x + z - 1) / (1 - z),
            sympy.sqrt(15) * (4 * z - 1) * (2 * x + z - 1) / (1 - z),
            3 * sympy.sqrt(3) * (2 * x + z - 1) * (2 * y + z - 1) / (1 - z) ** 2,
            3 * sympy.sqrt(5) * (4 * z - 1) * (2 * x + z - 1) * (2 * y + z - 1) / (1 - z) ** 2,
        ]
    elif n == 2:
        return [
            sympy.sqrt(3),
            sympy.sqrt(5) * (4 * z - 1),
            sympy.sqrt(7) * ((15 * z / 4 - 25 / 16) * (4 * z - 1) - 9 / 16),
            3 * (2 * y + z - 1) / (1 - z),
            sympy.sqrt(15) * (4 * z - 1) * (2 * y + z - 1) / (1 - z),
            sympy.sqrt(21)
            * ((15 * z / 4 - 25 / 16) * (4 * z - 1) - 9 / 16)
            * (2 * y + z - 1)
            / (1 - z),
            sympy.sqrt(15) * (-1 / 2 + 3 * (2 * y + z - 1) ** 2 / (2 * (1 - z) ** 2)),
            5 * (-1 / 2 + 3 * (2 * y + z - 1) ** 2 / (2 * (1 - z) ** 2)) * (4 * z - 1),
            sympy.sqrt(35)
            * (-1 / 2 + 3 * (2 * y + z - 1) ** 2 / (2 * (1 - z) ** 2))
            * ((15 * z / 4 - 25 / 16) * (4 * z - 1) - 9 / 16),
            3 * (2 * x + z - 1) / (1 - z),
            sympy.sqrt(15) * (4 * z - 1) * (2 * x + z - 1) / (1 - z),
            sympy.sqrt(21)
            * ((15 * z / 4 - 25 / 16) * (4 * z - 1) - 9 / 16)
            * (2 * x + z - 1)
            / (1 - z),
            3 * sympy.sqrt(3) * (2 * x + z - 1) * (2 * y + z - 1) / (1 - z) ** 2,
            3 * sympy.sqrt(5) * (4 * z - 1) * (2 * x + z - 1) * (2 * y + z - 1) / (1 - z) ** 2,
            3
            * sympy.sqrt(7)
            * ((15 * z / 4 - 25 / 16) * (4 * z - 1) - 9 / 16)
            * (2 * x + z - 1)
            * (2 * y + z - 1)
            / (1 - z) ** 2,
            3
            * sympy.sqrt(5)
            * (-1 / 2 + 3 * (2 * y + z - 1) ** 2 / (2 * (1 - z) ** 2))
            * (2 * x + z - 1)
            / (1 - z),
            5
            * sympy.sqrt(3)
            * (-1 / 2 + 3 * (2 * y + z - 1) ** 2 / (2 * (1 - z) ** 2))
            * (4 * z - 1)
            * (2 * x + z - 1)
            / (1 - z),
            sympy.sqrt(105)
            * (-1 / 2 + 3 * (2 * y + z - 1) ** 2 / (2 * (1 - z) ** 2))
            * ((15 * z / 4 - 25 / 16) * (4 * z - 1) - 9 / 16)
            * (2 * x + z - 1)
            / (1 - z),
            sympy.sqrt(15) * (-1 / 2 + 3 * (2 * x + z - 1) ** 2 / (2 * (1 - z) ** 2)),
            5 * (-1 / 2 + 3 * (2 * x + z - 1) ** 2 / (2 * (1 - z) ** 2)) * (4 * z - 1),
            sympy.sqrt(35)
            * (-1 / 2 + 3 * (2 * x + z - 1) ** 2 / (2 * (1 - z) ** 2))
            * ((15 * z / 4 - 25 / 16) * (4 * z - 1) - 9 / 16),
            3
            * sympy.sqrt(5)
            * (-1 / 2 + 3 * (2 * x + z - 1) ** 2 / (2 * (1 - z) ** 2))
            * (2 * y + z - 1)
            / (1 - z),
            5
            * sympy.sqrt(3)
            * (-1 / 2 + 3 * (2 * x + z - 1) ** 2 / (2 * (1 - z) ** 2))
            * (4 * z - 1)
            * (2 * y + z - 1)
            / (1 - z),
            sympy.sqrt(105)
            * (-1 / 2 + 3 * (2 * x + z - 1) ** 2 / (2 * (1 - z) ** 2))
            * ((15 * z / 4 - 25 / 16) * (4 * z - 1) - 9 / 16)
            * (2 * y + z - 1)
            / (1 - z),
            5
            * sympy.sqrt(3)
            * (-1 / 2 + 3 * (2 * x + z - 1) ** 2 / (2 * (1 - z) ** 2))
            * (-1 / 2 + 3 * (2 * y + z - 1) ** 2 / (2 * (1 - z) ** 2)),
            5
            * sympy.sqrt(5)
            * (-1 / 2 + 3 * (2 * x + z - 1) ** 2 / (2 * (1 - z) ** 2))
            * (-1 / 2 + 3 * (2 * y + z - 1) ** 2 / (2 * (1 - z) ** 2))
            * (4 * z - 1),
            5
            * sympy.sqrt(7)
            * (-1 / 2 + 3 * (2 * x + z - 1) ** 2 / (2 * (1 - z) ** 2))
            * (-1 / 2 + 3 * (2 * y + z - 1) ** 2 / (2 * (1 - z) ** 2))
            * ((15 * z / 4 - 25 / 16) * (4 * z - 1) - 9 / 16),
        ]
    else:
        raise NotImplementedError()


@pytest.mark.parametrize("n", range(3))
@pytest.mark.parametrize("nderiv", range(6))
def test_symbolic_pyramid(n, nderiv):
    idx = basix.index

    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")

    w = symbolic_pyramid(n)

    cell = basix.CellType.pyramid
    pts0 = basix.create_lattice(cell, 5, basix.LatticeType.equispaced, False)
    wtab = basix.polynomials.tabulate_polynomial_set(
        cell, basix.PolysetType.standard, n, nderiv, pts0
    )

    for kx in range(nderiv + 1):
        for ky in range(0, nderiv + 1 - kx):
            for kz in range(0, nderiv + 1 - kx - ky):
                wsym = np.zeros_like(wtab[0])
                for i, wi in enumerate(w):
                    wd = sympy.diff(wi, x, kx, y, ky, z, kz)
                    for j, p in enumerate(pts0):
                        wsym[i, j] = wd.subs([(x, p[0]), (y, p[1]), (z, p[2])])
                assert np.allclose(wtab[idx(kx, ky, kz)], wsym)


@pytest.mark.parametrize("n", range(3))
@pytest.mark.parametrize("nderiv", range(6))
def test_symbolic_pyramid_nan(n, nderiv):
    idx = basix.index

    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")

    w = symbolic_pyramid(n)

    cell = basix.CellType.pyramid
    pts0 = np.array([[0.0, 0.0, 1.0]])
    wtab = basix.polynomials.tabulate_polynomial_set(
        cell, basix.PolysetType.standard, n, nderiv, pts0
    )

    for kx in range(nderiv + 1):
        for ky in range(0, nderiv + 1 - kx):
            for kz in range(0, nderiv + 1 - kx - ky):
                wsym = np.zeros_like(wtab[0])
                for i, wi in enumerate(w):
                    wd = sympy.diff(wi, x, kx, y, ky, z, kz)
                    for j, p in enumerate(pts0):
                        value = wd.subs([(x, p[0]), (y, p[1]), (z, p[2])])
                        if value.is_finite:
                            wsym[i, j] = value
                        else:
                            wsym[i, j] = np.nan
                for a, b in zip(wtab[idx(kx, ky, kz)], wsym):
                    assert len(a) == len(b) == 1
                    assert np.isnan(b[0]) or np.isclose(a[0], b[0])
