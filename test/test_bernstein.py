# Copyright (c) 2022 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import numpy
import pytest
import sympy

x = sympy.Symbol("x")
y = sympy.Symbol("y")
z = sympy.Symbol("z")


def get_bernstein_polynomials(celltype, degree):
    if celltype == basix.CellType.interval:
        if degree == 1:
            return [1 - x, x]
        if degree == 2:
            return [(1 - x)**2, x*(2 - 2*x), x**2]
        if degree == 3:
            return [(1 - x)**3, 3*x*(1 - x)**2, x**2*(3 - 3*x), x**3]
    if celltype == basix.CellType.triangle:
        if degree == 1:
            return [1 - x - y, x, y]
        if degree == 2:
            return [(1 - x - y)**2, 2*x*(1 - x - y), x**2, 2*y*(1 - x - y), 2*x*y, y**2]
        if degree == 3:
            return [
                (1 - x - y)**3, 3*x*(1 - x - y)**2, 3*x**2*(1 - x - y), x**3,
                3*y*(1 - x - y)**2, 6*x*y*(1 - x - y), 3*x**2*y, 3*y**2*(1 - x - y), 3*x*y**2, y**3
            ]
    if celltype == basix.CellType.tetrahedron:
        if degree == 1:
            return [-x - y - z + 1, x, y, z]
        if degree == 2:
            return [
                (-x - y - z + 1)**2, x*(-2*x - 2*y - 2*z + 2), x**2, y*(-2*x - 2*y - 2*z + 2),
                2*x*y, y**2, z*(-2*x - 2*y - 2*z + 2), 2*x*z, 2*y*z, z**2
            ]
        if degree == 3:
            return [
                (-x - y - z + 1)**3, 3*x*(-x - y - z + 1)**2, x**2*(-3*x - 3*y - 3*z + 3), x**3,
                3*y*(-x - y - z + 1)**2, x*y*(-6*x - 6*y - 6*z + 6), 3*x**2*y,
                y**2*(-3*x - 3*y - 3*z + 3), 3*x*y**2, y**3, 3*z*(-x - y - z + 1)**2,
                x*z*(-6*x - 6*y - 6*z + 6), 3*x**2*z, y*z*(-6*x - 6*y - 6*z + 6), 6*x*y*z,
                3*y**2*z, z**2*(-3*x - 3*y - 3*z + 3), 3*x*z**2, 3*y*z**2, z**3
            ]

    raise NotImplementedError()


def get_bernstein_polynomials_entity_order(celltype, degree):
    if celltype == basix.CellType.interval:
        if degree == 1:
            return [1 - x, x]
        if degree == 2:
            return [(1 - x)**2, x**2, x*(2 - 2*x)]
        if degree == 3:
            return [(1 - x)**3, x**3, 3*x*(1 - x)**2, x**2*(3 - 3*x)]

    if celltype == basix.CellType.triangle:
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


@pytest.mark.parametrize("celltype", [
    basix.CellType.interval,
    basix.CellType.triangle,
    basix.CellType.tetrahedron
])
@pytest.mark.parametrize("degree", range(1, 4))
def test_poly(celltype, degree):
    pts = basix.create_lattice(celltype, 6, basix.LatticeType.equispaced, True)
    wtab = basix.tabulate_polynomials(basix.PolynomialType.bernstein, celltype, degree, pts)
    bern = get_bernstein_polynomials(celltype, degree)
    wsym = numpy.array([[float(b.subs(list(zip([x, y, z], p)))) for p in pts] for b in bern])
    assert numpy.allclose(wtab, wsym)


@pytest.mark.parametrize("celltype", [
    basix.CellType.interval,
    basix.CellType.triangle
])
@pytest.mark.parametrize("degree", range(1, 4))
def test_element(celltype, degree):
    bern = get_bernstein_polynomials_entity_order(celltype, degree)
    lagrange = basix.create_element(basix.ElementFamily.P, celltype, degree,
                                    basix.LagrangeVariant.bernstein)
    pts = basix.create_lattice(celltype, 6, basix.LatticeType.equispaced, True)
    nderiv = 3
    wtab = lagrange.tabulate(nderiv, pts)

    if celltype == basix.CellType.interval:
        derivs = [(0,), (1,), (2,), (3,)]
    elif celltype == basix.CellType.triangle:
        derivs = [(n - i, i) for n in range(4) for i in range(n + 1)]

    for k in derivs:
        wsym = numpy.zeros_like(wtab[0])
        for i, b in enumerate(bern):
            wd = b
            for v, n in zip([x, y, z], k):
                wd = sympy.diff(wd, v, n)
            for j, p in enumerate(pts):
                wsym[j, i] = wd.subs(list(zip([x, y, z], p)))

        assert numpy.allclose(wtab[basix.index(*k)], wsym)


@pytest.mark.parametrize("celltype", [
    basix.CellType.interval,
    basix.CellType.triangle,
    basix.CellType.tetrahedron
])
@pytest.mark.parametrize("degree", range(1, 6))
def test_basis_is_polynomials(celltype, degree):
    lagrange = basix.create_element(basix.ElementFamily.P, celltype, degree, basix.LagrangeVariant.bernstein)
    pts = basix.create_lattice(celltype, 6, basix.LatticeType.equispaced, True)
    wtab = lagrange.tabulate(0, pts)[0, :, :, 0]
    bern = basix.tabulate_polynomials(basix.PolynomialType.bernstein, celltype, degree, pts)
    remaining = [i for i, _ in enumerate(bern)]
    for row in wtab.T:
        for i in remaining:
            if numpy.allclose(row, bern[i]):
                remaining.remove(i)
                break
        else:
            raise AssertionError

    assert len(remaining) == 0
