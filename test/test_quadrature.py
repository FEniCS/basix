# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import sympy

import basix


@pytest.mark.parametrize(
    "celltype",
    [
        (basix.CellType.quadrilateral, 1.0),
        (basix.CellType.hexahedron, 1.0),
        (basix.CellType.prism, 0.5),
        (basix.CellType.interval, 1.0),
        (basix.CellType.triangle, 0.5),
        (basix.CellType.tetrahedron, 1.0 / 6.0),
    ],
)
@pytest.mark.parametrize("order", range(9))
def test_cell_quadrature(celltype, order):
    Qpts, Qwts = basix.make_quadrature(celltype[0], order)
    assert np.isclose(sum(Qwts), celltype[1])


@pytest.mark.parametrize("m", range(7))
@pytest.mark.parametrize("scheme", [basix.QuadratureType.default, basix.QuadratureType.gll])
def test_qorder_line(m, scheme):
    Qpts, Qwts = basix.make_quadrature(basix.CellType.interval, m, rule=scheme)
    x = sympy.Symbol("x")
    f = x**m
    q = sympy.integrate(f, (x, 0, 1))
    s = 0.0
    for pt, wt in zip(Qpts, Qwts):
        s += wt * f.subs([(x, pt[0])])
    assert np.isclose(float(q), float(s))


@pytest.mark.parametrize("m", range(6))
@pytest.mark.parametrize(
    "scheme", [basix.QuadratureType.default, basix.QuadratureType.gauss_jacobi]
)
def test_qorder_tri(m, scheme):
    Qpts, Qwts = basix.make_quadrature(basix.CellType.triangle, m, rule=scheme)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    f = x**m + y**m
    q = sympy.integrate(f, (x, 0, 1 - y), (y, 0, 1))
    s = 0.0
    for pt, wt in zip(Qpts, Qwts):
        s += wt * f.subs([(x, pt[0]), (y, pt[1])])
    assert np.isclose(float(q), float(s))


@pytest.mark.parametrize("m", range(1, 20))
@pytest.mark.parametrize("scheme", [basix.QuadratureType.xiao_gimbutas])
def test_xiao_gimbutas_tri(m, scheme):
    Qpts, Qwts = basix.make_quadrature(basix.CellType.triangle, m, rule=scheme)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    f = x**m + 2 * y**m
    q = sympy.integrate(f, (x, 0, 1 - y), (y, 0, 1))
    s = 0.0
    for pt, wt in zip(Qpts, Qwts):
        s += wt * f.subs([(x, pt[0]), (y, pt[1])])
    assert np.isclose(float(q), float(s))


@pytest.mark.parametrize("m", range(1, 16))
@pytest.mark.parametrize("scheme", [basix.QuadratureType.xiao_gimbutas])
def test_xiao_gimbutas_tet(m, scheme):
    Qpts, Qwts = basix.make_quadrature(basix.CellType.tetrahedron, m, rule=scheme)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    f = x**m + 2 * y**m + 3 * z**m
    q = sympy.integrate(f, (x, 0, 1 - y - z), (y, 0, 1 - z), (z, 0, 1))
    s = 0.0
    for pt, wt in zip(Qpts, Qwts):
        s += wt * f.subs([(x, pt[0]), (y, pt[1]), (z, pt[2])])
    assert np.isclose(float(q), float(s))


@pytest.mark.parametrize("m", range(9))
@pytest.mark.parametrize(
    "scheme", [basix.QuadratureType.default, basix.QuadratureType.gauss_jacobi]
)
def test_qorder_tet(m, scheme):
    Qpts, Qwts = basix.make_quadrature(basix.CellType.tetrahedron, m, rule=scheme)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    f = x**m + y**m + z**m
    q = sympy.integrate(f, (x, 0, 1 - y - z), (y, 0, 1 - z), (z, 0, 1))
    s = 0.0
    for pt, wt in zip(Qpts, Qwts):
        s += wt * f.subs([(x, pt[0]), (y, pt[1]), (z, pt[2])])
    assert np.isclose(float(q), float(s))


@pytest.mark.parametrize("m", range(9))
@pytest.mark.parametrize(
    "scheme", [basix.QuadratureType.default, basix.QuadratureType.gauss_jacobi]
)
def test_qorder_prism(m, scheme):
    Qpts, Qwts = basix.make_quadrature(basix.CellType.prism, m, rule=scheme)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    f = x**m + y**m + z**m
    q = sympy.integrate(f, (x, 0, 1 - y), (y, 0, 1), (z, 0, 1))
    s = 0.0
    for pt, wt in zip(Qpts, Qwts):
        s += wt * f.subs([(x, pt[0]), (y, pt[1]), (z, pt[2])])
    assert np.isclose(float(q), float(s))


@pytest.mark.parametrize("m", range(6))
@pytest.mark.parametrize(
    "scheme", [basix.QuadratureType.default, basix.QuadratureType.gauss_jacobi]
)
def test_qorder_pyramid(m, scheme):
    Qpts, Qwts = basix.make_quadrature(basix.CellType.pyramid, m, rule=scheme)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    polyset = [
        x**i * y**j * z**k
        for i in range(m + 1)
        for j in range(m + 1 - i)
        for k in range(m + 1 - i - j)
    ]
    polyset += [x * y**m / (1 - z) ** m, x**m * y / (1 - z) ** m]
    polyset += [x**a * y**b / (1 - z) ** min(a, b) for a in range(m) for b in range(m + 1 - a, m)]
    for f in polyset:
        q = sympy.integrate(f, (x, 0, 1 - z), (y, 0, 1 - z), (z, 0, 1))
        s = 0.0
        for pt, wt in zip(Qpts, Qwts):
            s += wt * f.subs([(x, pt[0]), (y, pt[1]), (z, pt[2])])
        assert np.isclose(float(q), float(s))


def test_quadrature_function():
    Qpts, Qwts = basix.make_quadrature(basix.CellType.interval, 3)
    # Scale to interval [0.0, 2.0]
    Qpts *= 2.0
    Qwts *= 2.0

    def f(x):
        return x * x

    b = sum([w * f(pt[0]) for pt, w in zip(Qpts, Qwts)])

    assert np.isclose(b, 8.0 / 3.0)


def test_gll():
    m = 5

    # 1D interval
    pts, wts = basix.make_quadrature(basix.CellType.interval, m + 1, rule=basix.QuadratureType.gll)
    pts, wts = 2 * pts.flatten() - 1, 2 * wts.flatten()
    ref_pts = np.array([-1.0, 1.0, -np.sqrt(3 / 7), 0.0, np.sqrt(3 / 7)])
    assert np.allclose(pts.flatten(), ref_pts)
    ref_wts = np.array([1 / 10, 1 / 10, 49 / 90, 32 / 45, 49 / 90])
    assert np.allclose(wts, ref_wts)
    assert np.isclose(sum(pts * wts), 0)
    assert np.isclose(sum(wts), 2)

    # 2D quad
    pts, wts = basix.make_quadrature(
        basix.CellType.quadrilateral, m + 1, rule=basix.QuadratureType.gll
    )
    pts, wts = 2 * pts - 1, 4 * wts
    ref_pts2 = np.array([[x, y] for x in ref_pts for y in ref_pts])
    assert np.allclose(pts, ref_pts2)
    ref_wts2 = np.array([w1 * w2 for w1 in ref_wts for w2 in ref_wts])
    assert np.allclose(wts, ref_wts2)
    assert np.isclose((pts * wts.reshape(-1, 1)).sum(), 0)
    assert np.isclose(sum(wts), 4)

    # 3D hex
    pts, wts = basix.make_quadrature(
        basix.CellType.hexahedron, m + 1, rule=basix.QuadratureType.gll
    )
    pts, wts = 2 * pts - 1, 8 * wts
    ref_pts3 = np.array([[x, y, z] for x in ref_pts for y in ref_pts for z in ref_pts])
    assert np.allclose(pts, ref_pts3)
    ref_wts3 = np.array([w1 * w2 * w3 for w1 in ref_wts for w2 in ref_wts for w3 in ref_wts])
    assert np.allclose(wts, ref_wts3)
    assert np.isclose((pts * wts.reshape(-1, 1)).sum(), 0)
    assert np.isclose(sum(wts), 8)


@pytest.mark.parametrize("alpha", [0, 0.0, 1, 1.0, 1.5, 2.0, 3.0, 4.2])
@pytest.mark.parametrize("degree", range(6))
def test_gauss_jacobi_rule(alpha, degree):
    pts, wts = basix.quadrature.gauss_jacobi_rule(alpha, degree + 1)
    integral = sum(w * (1 - p) ** degree for p, w in zip(pts, wts))

    expected = 1 / (alpha + degree + 1)
    assert np.isclose(integral, expected)
