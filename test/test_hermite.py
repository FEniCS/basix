# Copyright (c) 2022 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import numpy
import sympy


def test_hermite_interval():
    hermite = basix.create_element(basix.ElementFamily.Hermite, basix.CellType.interval, 3)

    x = sympy.Symbol("x")

    # Hermite basis functions taken from Symfem
    sym_basis = [2*x**3 - 3*x**2 + 1, x**3 - 2*x**2 + x, -2*x**3 + 3*x**2, x**3 - x**2]

    pts = basix.create_lattice(basix.CellType.interval, 3, basix.LatticeType.equispaced, True)

    nderiv = 2
    wtab = hermite.tabulate(nderiv, pts)

    for kx in range(nderiv + 1):
        wsym = numpy.zeros_like(wtab[0])
        for i, f in enumerate(sym_basis):
            wd = sympy.diff(f, x, kx)
            for k, p in enumerate(pts):
                wsym[k, i] = wd.subs([(x, p[0])])

        assert numpy.allclose(wtab[kx], wsym)


def test_hermite_triangle():
    hermite = basix.create_element(basix.ElementFamily.Hermite, basix.CellType.triangle, 3)

    x = sympy.Symbol("x")
    y = sympy.Symbol("y")

    # Hermite basis functions taken from Symfem
    sym_basis = [
        2*x**3 + 13*x**2*y - 3*x**2 + 13*x*y**2 - 13*x*y + 2*y**3 - 3*y**2 + 1,
        x**3 + 3*x**2*y - 2*x**2 + 2*x*y**2 - 3*x*y + x,
        2*x**2*y + 3*x*y**2 - 3*x*y + y**3 - 2*y**2 + y,
        -2*x**3 + 7*x**2*y + 3*x**2 + 7*x*y**2 - 7*x*y,
        x**3 - 2*x**2*y - x**2 - 2*x*y**2 + 2*x*y,
        2*x**2*y + x*y**2 - x*y,
        7*x**2*y + 7*x*y**2 - 7*x*y - 2*y**3 + 3*y**2,
        x**2*y + 2*x*y**2 - x*y,
        -2*x**2*y - 2*x*y**2 + 2*x*y + y**3 - y**2,
        -27*x**2*y - 27*x*y**2 + 27*x*y
    ]

    pts = basix.create_lattice(basix.CellType.triangle, 3, basix.LatticeType.equispaced, True)

    nderiv = 2
    wtab = hermite.tabulate(nderiv, pts)

    for kx in range(nderiv + 1):
        for ky in range(nderiv + 1 - kx):
            wsym = numpy.zeros_like(wtab[0])
            for i, f in enumerate(sym_basis):
                wd = sympy.diff(f, x, kx, y, ky)
                for k, p in enumerate(pts):
                    wsym[k, i] = wd.subs([(x, p[0]), (y, p[1])])

            assert numpy.allclose(wtab[basix.index(kx, ky)], wsym)


def test_hermite_tetrahedron():
    hermite = basix.create_element(basix.ElementFamily.Hermite, basix.CellType.tetrahedron, 3)

    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")

    # Hermite basis functions taken from Symfem
    sym_basis = [
        (
            2*x**3 + 13*x**2*y + 13*x**2*z - 3*x**2 + 13*x*y**2 + 33*x*y*z - 13*x*y + 13*x*z**2
            - 13*x*z + 2*y**3 + 13*y**2*z - 3*y**2 + 13*y*z**2 - 13*y*z + 2*z**3 - 3*z**2 + 1
        ),
        x**3 + 3*x**2*y + 3*x**2*z - 2*x**2 + 2*x*y**2 + 4*x*y*z - 3*x*y + 2*x*z**2 - 3*x*z + x,
        2*x**2*y + 3*x*y**2 + 4*x*y*z - 3*x*y + y**3 + 3*y**2*z - 2*y**2 + 2*y*z**2 - 3*y*z + y,
        2*x**2*z + 4*x*y*z + 3*x*z**2 - 3*x*z + 2*y**2*z + 3*y*z**2 - 3*y*z + z**3 - 2*z**2 + z,
        -2*x**3 + 7*x**2*y + 7*x**2*z + 3*x**2 + 7*x*y**2 + 7*x*y*z - 7*x*y + 7*x*z**2 - 7*x*z,
        x**3 - 2*x**2*y - 2*x**2*z - x**2 - 2*x*y**2 - 2*x*y*z + 2*x*y - 2*x*z**2 + 2*x*z,
        2*x**2*y + x*y**2 - x*y,
        2*x**2*z + x*z**2 - x*z,
        7*x**2*y + 7*x*y**2 + 7*x*y*z - 7*x*y - 2*y**3 + 7*y**2*z + 3*y**2 + 7*y*z**2 - 7*y*z,
        x**2*y + 2*x*y**2 - x*y,
        -2*x**2*y - 2*x*y**2 - 2*x*y*z + 2*x*y + y**3 - 2*y**2*z - y**2 - 2*y*z**2 + 2*y*z,
        2*y**2*z + y*z**2 - y*z,
        7*x**2*z + 7*x*y*z + 7*x*z**2 - 7*x*z + 7*y**2*z + 7*y*z**2 - 7*y*z - 2*z**3 + 3*z**2,
        x**2*z + 2*x*z**2 - x*z,
        y**2*z + 2*y*z**2 - y*z,
        -2*x**2*z - 2*x*y*z - 2*x*z**2 + 2*x*z - 2*y**2*z - 2*y*z**2 + 2*y*z + z**3 - z**2,
        27*x*y*z,
        -27*x*y*z - 27*y**2*z - 27*y*z**2 + 27*y*z,
        -27*x**2*z - 27*x*y*z - 27*x*z**2 + 27*x*z,
        -27*x**2*y - 27*x*y**2 - 27*x*y*z + 27*x*y
    ]

    pts = basix.create_lattice(basix.CellType.tetrahedron, 3, basix.LatticeType.equispaced, True)

    nderiv = 2
    wtab = hermite.tabulate(nderiv, pts)

    for kx in range(nderiv + 1):
        for ky in range(nderiv + 1 - kx):
            for kz in range(nderiv + 1 - kx - ky):
                wsym = numpy.zeros_like(wtab[0])
                for i, f in enumerate(sym_basis):
                    wd = sympy.diff(f, x, kx, y, ky, z, kz)
                    for k, p in enumerate(pts):
                        wsym[k, i] = wd.subs([(x, p[0]), (y, p[1]), (z, p[2])])

                assert numpy.allclose(wtab[basix.index(kx, ky, kz)], wsym)
