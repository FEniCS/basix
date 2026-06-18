# Copyright (c) 2020 Chris Richardson & Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import sympy

import basix


def sympy_nedelec(celltype, n):
    # These basis functions were computed using symfem. They can be recomputed
    # by running (eg):
    #    import symfem
    #    e = symfem.create_element("triangle", "N2curl", 2)
    #    print(e.get_basis_functions())
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")

    if celltype == basix.CellType.triangle:
        if n == 1:
            return [
                [-6 * x - 4 * y + 4, -2 * x],
                [6 * x + 2 * y - 2, 4 * x],
                [-2 * y, -4 * x - 6 * y + 4],
                [4 * y, 2 * x + 6 * y - 2],
                [2 * y, 4 * x],
                [-4 * y, -2 * x],
            ]
        if n == 2:
            return [
                [
                    30 * x**2 + 48 * x * y - 36 * x + 18 * y**2 - 27 * y + 9,
                    12 * x**2 + 12 * x * y - 9 * x,
                ],
                [30 * x**2 + 12 * x * y - 24 * x - 3 * y + 3, 18 * x**2 - 9 * x],
                [
                    -15 * x**2 - 15 * x * y + 15 * x + 3 * y**2 / 2 - 3 / 2,
                    -15 * x**2 / 2 - 9 * x * y + 6 * x,
                ],
                [
                    12 * x * y + 12 * y**2 - 9 * y,
                    18 * x**2 + 48 * x * y - 27 * x + 30 * y**2 - 36 * y + 9,
                ],
                [18 * y**2 - 9 * y, 12 * x * y - 3 * x + 30 * y**2 - 24 * y + 3],
                [
                    -9 * x * y - 15 * y**2 / 2 + 6 * y,
                    3 * x**2 / 2 - 15 * x * y - 15 * y**2 + 15 * y - 3 / 2,
                ],
                [12 * x * y - 3 * y, 18 * x**2 - 9 * x],
                [-18 * y**2 + 9 * y, -12 * x * y + 3 * x],
                [-9 * x * y - 3 * y**2 / 2 + 3 * y, 3 * x**2 / 2 + 9 * x * y - 3 * x],
                [-24 * x * y - 12 * y**2 + 12 * y, -36 * x**2 - 48 * x * y + 36 * x],
                [48 * x * y + 36 * y**2 - 36 * y, 12 * x**2 + 24 * x * y - 12 * x],
                [-48 * x * y - 12 * y**2 + 12 * y, -12 * x**2 - 48 * x * y + 12 * x],
            ]
    if celltype == basix.CellType.tetrahedron:
        if n == 1:
            return [
                [-6 * x - 4 * y - 4 * z + 4, -2 * x, -2 * x],
                [6 * x + 2 * y + 2 * z - 2, 4 * x, 4 * x],
                [-2 * y, -4 * x - 6 * y - 4 * z + 4, -2 * y],
                [4 * y, 2 * x + 6 * y + 2 * z - 2, 4 * y],
                [-2 * z, -2 * z, -4 * x - 4 * y - 6 * z + 4],
                [4 * z, 4 * z, 2 * x + 2 * y + 6 * z - 2],
                [2 * y, 4 * x, 0],
                [-4 * y, -2 * x, 0],
                [2 * z, 0, 4 * x],
                [-4 * z, 0, -2 * x],
                [0, 2 * z, 4 * y],
                [0, -4 * z, -2 * y],
            ]
        if n == 2:
            return [
                (
                    30 * x**2
                    + 48 * x * y
                    + 48 * x * z
                    - 36 * x
                    + 18 * y**2
                    + 36 * y * z
                    - 27 * y
                    + 18 * z**2
                    - 27 * z
                    + 9,
                    12 * x**2 + 12 * x * y + 12 * x * z - 9 * x,
                    12 * x**2 + 12 * x * y + 12 * x * z - 9 * x,
                ),
                (
                    30 * x**2 + 12 * x * y + 12 * x * z - 24 * x - 3 * y - 3 * z + 3,
                    18 * x**2 - 9 * x,
                    18 * x**2 - 9 * x,
                ),
                (
                    -15 * x**2
                    - 15 * x * y
                    - 15 * x * z
                    + 15 * x
                    + 3 * y**2 / 2
                    + 3 * y * z
                    + 3 * z**2 / 2
                    - 3 / 2,
                    -15 * x**2 / 2 - 9 * x * y - 9 * x * z + 6 * x,
                    -15 * x**2 / 2 - 9 * x * y - 9 * x * z + 6 * x,
                ),
                (
                    12 * x * y + 12 * y**2 + 12 * y * z - 9 * y,
                    18 * x**2
                    + 48 * x * y
                    + 36 * x * z
                    - 27 * x
                    + 30 * y**2
                    + 48 * y * z
                    - 36 * y
                    + 18 * z**2
                    - 27 * z
                    + 9,
                    12 * x * y + 12 * y**2 + 12 * y * z - 9 * y,
                ),
                (
                    18 * y**2 - 9 * y,
                    12 * x * y - 3 * x + 30 * y**2 + 12 * y * z - 24 * y - 3 * z + 3,
                    18 * y**2 - 9 * y,
                ),
                (
                    -9 * x * y - 15 * y**2 / 2 - 9 * y * z + 6 * y,
                    3 * x**2 / 2
                    - 15 * x * y
                    + 3 * x * z
                    - 15 * y**2
                    - 15 * y * z
                    + 15 * y
                    + 3 * z**2 / 2
                    - 3 / 2,
                    -9 * x * y - 15 * y**2 / 2 - 9 * y * z + 6 * y,
                ),
                (
                    12 * x * z + 12 * y * z + 12 * z**2 - 9 * z,
                    12 * x * z + 12 * y * z + 12 * z**2 - 9 * z,
                    18 * x**2
                    + 36 * x * y
                    + 48 * x * z
                    - 27 * x
                    + 18 * y**2
                    + 48 * y * z
                    - 27 * y
                    + 30 * z**2
                    - 36 * z
                    + 9,
                ),
                (
                    18 * z**2 - 9 * z,
                    18 * z**2 - 9 * z,
                    12 * x * z - 3 * x + 12 * y * z - 3 * y + 30 * z**2 - 24 * z + 3,
                ),
                (
                    -9 * x * z - 9 * y * z - 15 * z**2 / 2 + 6 * z,
                    -9 * x * z - 9 * y * z - 15 * z**2 / 2 + 6 * z,
                    3 * x**2 / 2
                    + 3 * x * y
                    - 15 * x * z
                    + 3 * y**2 / 2
                    - 15 * y * z
                    - 15 * z**2
                    + 15 * z
                    - 3 / 2,
                ),
                (12 * x * y - 3 * y, 18 * x**2 - 9 * x, 0),
                (-18 * y**2 + 9 * y, -12 * x * y + 3 * x, 0),
                (-9 * x * y - 3 * y**2 / 2 + 3 * y, 3 * x**2 / 2 + 9 * x * y - 3 * x, 0),
                (12 * x * z - 3 * z, 0, 18 * x**2 - 9 * x),
                (-18 * z**2 + 9 * z, 0, -12 * x * z + 3 * x),
                (-9 * x * z - 3 * z**2 / 2 + 3 * z, 0, 3 * x**2 / 2 + 9 * x * z - 3 * x),
                (0, 12 * y * z - 3 * z, 18 * y**2 - 9 * y),
                (0, -18 * z**2 + 9 * z, -12 * y * z + 3 * y),
                (0, -9 * y * z - 3 * z**2 / 2 + 3 * z, 3 * y**2 / 2 + 9 * y * z - 3 * y),
                (
                    -24 * x * y - 12 * y**2 - 12 * y * z + 12 * y,
                    -36 * x**2 - 48 * x * y - 36 * x * z + 36 * x,
                    -12 * x * y,
                ),
                (
                    48 * x * y + 36 * y**2 + 36 * y * z - 36 * y,
                    12 * x**2 + 24 * x * y + 12 * x * z - 12 * x,
                    12 * x * y,
                ),
                (
                    -48 * x * y - 12 * y**2 - 12 * y * z + 12 * y,
                    -12 * x**2 - 48 * x * y - 12 * x * z + 12 * x,
                    -36 * x * y,
                ),
                (
                    -24 * x * z - 12 * y * z - 12 * z**2 + 12 * z,
                    -12 * x * z,
                    -36 * x**2 - 36 * x * y - 48 * x * z + 36 * x,
                ),
                (
                    48 * x * z + 36 * y * z + 36 * z**2 - 36 * z,
                    12 * x * z,
                    12 * x**2 + 12 * x * y + 24 * x * z - 12 * x,
                ),
                (
                    -48 * x * z - 12 * y * z - 12 * z**2 + 12 * z,
                    -36 * x * z,
                    -12 * x**2 - 12 * x * y - 48 * x * z + 12 * x,
                ),
                (
                    -12 * y * z,
                    -12 * x * z - 24 * y * z - 12 * z**2 + 12 * z,
                    -36 * x * y - 36 * y**2 - 48 * y * z + 36 * y,
                ),
                (
                    12 * y * z,
                    36 * x * z + 48 * y * z + 36 * z**2 - 36 * z,
                    12 * x * y + 12 * y**2 + 24 * y * z - 12 * y,
                ),
                (
                    -36 * y * z,
                    -12 * x * z - 48 * y * z - 12 * z**2 + 12 * z,
                    -12 * x * y - 12 * y**2 - 48 * y * z + 12 * y,
                ),
                (12 * y * z, 12 * x * z, 36 * x * y),
                (-12 * y * z, -36 * x * z, -12 * x * y),
                (36 * y * z, 12 * x * z, 12 * x * y),
            ]

    raise NotImplementedError


@pytest.mark.parametrize("degree", [1, 2])
def test_tri(degree):
    celltype = basix.CellType.triangle
    g = sympy_nedelec(celltype, degree)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    nedelec = basix.create_element(
        basix.ElementFamily.N2E, basix.CellType.triangle, degree, basix.LagrangeVariant.equispaced
    )
    pts = basix.create_lattice(celltype, 6, basix.LatticeType.equispaced, True)
    nderiv = 3
    wtab = nedelec.tabulate(nderiv, pts)
    for kx in range(nderiv + 1):
        for ky in range(nderiv + 1 - kx):
            wsym = np.zeros_like(wtab[0])
            for i, gi in enumerate(g):
                for j, gij in enumerate(gi):
                    wd = sympy.diff(gij, x, kx, y, ky)
                    for k, p in enumerate(pts):
                        wsym[k, i, j] = wd.subs([(x, p[0]), (y, p[1])])
            assert np.isclose(wtab[basix.index(kx, ky)], wsym).all()


@pytest.mark.parametrize("degree", [1, 2])
def test_tet(degree):
    celltype = basix.CellType.tetrahedron
    g = sympy_nedelec(celltype, degree)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    nedelec = basix.create_element(
        basix.ElementFamily.N2E,
        basix.CellType.tetrahedron,
        degree,
        basix.LagrangeVariant.equispaced,
    )
    pts = basix.create_lattice(celltype, 6, basix.LatticeType.equispaced, True)
    nderiv = 1
    wtab = nedelec.tabulate(nderiv, pts)
    for kx in range(nderiv + 1):
        for ky in range(nderiv + 1 - kx):
            for kz in range(nderiv + 1 - kx - ky):
                wsym = np.zeros_like(wtab[0])
                for i, gi in enumerate(g):
                    for j, gij in enumerate(gi):
                        wd = sympy.diff(gij, x, kx, y, ky, z, kz)
                        for k, p in enumerate(pts):
                            wsym[k, i, j] = wd.subs([(x, p[0]), (y, p[1]), (z, p[2])])
                assert np.isclose(wtab[basix.index(kx, ky, kz)], wsym).all()
