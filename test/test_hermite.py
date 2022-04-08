# Copyright (c) 2022 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import numpy
import sympy


def test_hermite_triangle():
    hermite = basix.create_element(basix.ElementFamily.Hermite, basix.CellType.triangle, 3)

    x = sympy.Symbol("x")
    y = sympy.Symbol("y")

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
