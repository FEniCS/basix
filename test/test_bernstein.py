# Copyright (c) 2022 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import numpy
import pytest
import sympy


def get_bernstein_polynomials(celltype, degree):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")

    if celltype == basix.CellType.triangle:
        if degree == 0:
            return [sympy.Integer(1)]
        if degree == 1:
            return [1 - x - y, x, y]
        if degree == 2:
            return [(1 - x - y)**2, x**2, y**2, 2*x*y, 2*y*(1 - x - y), 2*x*(1 - x - y)]
        if degree == 3:
            return [
                (1 - x - y)**3, x**3, y**3,
                3*x**2*y, 3*x*y**2, 3*y*(1 - x - y)**2, 3*y**2*(1 - x - y), 3*x*(1 - x - y)**2, 3*x**2*(1 - x - y),
                6*x*y*(1 - x - y)
            ]
    raise NotImplementedError()


@pytest.mark.parametrize("degree", range(4))
def test_tri(degree):
    celltype = basix.CellType.triangle

    bern = get_bernstein_polynomials(celltype, degree)

    lagrange = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, degree,
                                    basix.LagrangeVariant.bernstein)
    pts = basix.create_lattice(celltype, 6, basix.LatticeType.equispaced, True)
    nderiv = 3
    wtab = lagrange.tabulate(nderiv, pts)

    for kx in range(nderiv):
        for ky in range(0, nderiv - kx):
            wsym = numpy.zeros_like(wtab[0])
            for i, b in enumerate(bern):
                wd = sympy.diff(b, x, kx, y, ky)
                for j, p in enumerate(pts):
                    wsym[j, i] = wd.subs([(x, p[0]), (y, p[1])])

            assert numpy.allclose(wtab[basix.index(kx, ky)], wsym)
