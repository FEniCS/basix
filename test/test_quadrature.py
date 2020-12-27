# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import pytest
import numpy as np
import sympy

@pytest.mark.parametrize("celltype", [(basix.CellType.quadrilateral, 1.0),
                                      (basix.CellType.hexahedron, 1.0),
                                      (basix.CellType.prism, 0.5),
                                      (basix.CellType.interval, 1.0),
                                      (basix.CellType.triangle, 0.5),
                                      (basix.CellType.tetrahedron, 1.0/6.0)])
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 7, 8])
def test_cell_quadrature(celltype, order):
    Qpts, Qwts = basix.make_quadrature("default", celltype[0], order)
    print(sum(Qwts))
    assert(np.isclose(sum(Qwts), celltype[1]))


@pytest.mark.parametrize("m", [0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("scheme", ['default','GLL'])
def test_qorder_line(m, scheme):
    Qpts, Qwts = basix.make_quadrature(scheme, basix.CellType.interval, m)
    x = sympy.Symbol('x')
    f = x**m
    print(f)
    q = sympy.integrate(f, (x, 0, (1)))
    s = 0.0
    for (pt,wt) in zip(Qpts, Qwts):
        s += wt * f.subs([(x, pt[0])])
    print(len(Qwts))
    assert(np.isclose(float(q), float(s)))


@pytest.mark.parametrize("m", [0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("scheme", ['default','Gauss-Jacobi'])
def test_qorder_tri(m, scheme):
    Qpts, Qwts = basix.make_quadrature(scheme, basix.CellType.triangle, m)
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    f = x**m + y**m
    q = sympy.integrate(sympy.integrate(f, (x, 0, (1-y))), (y, 0, 1))
    s = 0.0
    for (pt,wt) in zip(Qpts, Qwts):
        s += wt * f.subs([(x, pt[0]), (y, pt[1])])
    print(len(Qwts))
    assert(np.isclose(float(q), float(s)))


@pytest.mark.parametrize("m", [0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("scheme", ['default','Gauss-Jacobi'])
def test_qorder_tet(m, scheme):
    Qpts, Qwts = basix.make_quadrature(scheme, basix.CellType.tetrahedron, m)
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    z = sympy.Symbol('z')
    f = x**m + y**m + z**m
    q = sympy.integrate(sympy.integrate(sympy.integrate(f, (x, 0, (1-y-z))), (y, 0, 1-z)), (z, 0, 1))
    s = 0.0
    for (pt,wt) in zip(Qpts, Qwts):
        s += wt * f.subs([(x, pt[0]), (y, pt[1]), (z, pt[2])])
    print(len(Qwts))
    assert(np.isclose(float(q), float(s)))


def test_quadrature_function():
    Qpts, Qwts = basix.make_quadrature("default", basix.CellType.interval, 3)
    # Scale to interval [0.0, 2.0]
    Qpts *= 2.0
    Qwts *= 2.0

    def f(x):
        return x * x

    b = sum([w * f(pt[0]) for pt, w in zip(Qpts, Qwts)])

    assert np.isclose(b, 8.0 / 3.0)


def test_jacobi():
    pts = np.arange(0, 1, 0.1)
    f = basix.compute_jacobi_deriv(1.0, 4, 5, pts)
    print(f)


def test_gll():
    m = 6
    pts, wts = basix.gauss_lobatto_legendre_line_rule(m)
    ref_pts = np.array([-1., -0.76505532,
                        -0.28523152, 0.28523152,
                        0.76505532, 1.])
    assert (np.allclose(pts, ref_pts))
    ref_wts = np.array([0.06666667, 0.37847496,
                        0.55485838, 0.55485838,
                        0.37847496, 0.06666667])
    assert (np.allclose(wts, ref_wts))
    print(pts, wts)
    assert np.isclose(sum(pts * wts), 0)
    assert np.isclose(sum(wts), 2)
