# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import functools

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
    _Qpts, Qwts = basix.make_quadrature(celltype[0], order)
    assert np.isclose(sum(Qwts), celltype[1])


@functools.cache
def _qorder_line_exact(m):
    x = sympy.Symbol("x")
    f = x**m
    return sympy.integrate(f, (x, 0, 1))


@pytest.mark.parametrize("m", range(7))
@pytest.mark.parametrize("scheme", [basix.QuadratureType.default, basix.QuadratureType.gll])
def test_qorder_line(m, scheme):
    Qpts, Qwts = basix.make_quadrature(basix.CellType.interval, m, rule=scheme)
    x = sympy.Symbol("x")
    f = x**m
    q = _qorder_line_exact(m)
    s = 0.0
    for pt, wt in zip(Qpts, Qwts):
        s += wt * f.subs([(x, pt[0])])
    assert np.isclose(float(q), float(s))


@functools.cache
def _qorder_tri_exact(m):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    f = x**m + y**m
    return sympy.integrate(f, (x, 0, 1 - y), (y, 0, 1))


@pytest.mark.parametrize("m", range(6))
@pytest.mark.parametrize(
    "scheme", [basix.QuadratureType.default, basix.QuadratureType.gauss_jacobi]
)
def test_qorder_tri(m, scheme):
    Qpts, Qwts = basix.make_quadrature(basix.CellType.triangle, m, rule=scheme)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    f = x**m + y**m
    q = _qorder_tri_exact(m)
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


@functools.cache
def _qorder_tet_exact(m):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    f = x**m + y**m + z**m
    return sympy.integrate(f, (x, 0, 1 - y - z), (y, 0, 1 - z), (z, 0, 1))


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
    q = _qorder_tet_exact(m)
    s = 0.0
    for pt, wt in zip(Qpts, Qwts):
        s += wt * f.subs([(x, pt[0]), (y, pt[1]), (z, pt[2])])
    assert np.isclose(float(q), float(s))


@functools.cache
def _qorder_prism_exact(m):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    f = x**m + y**m + z**m
    return sympy.integrate(f, (x, 0, 1 - y), (y, 0, 1), (z, 0, 1))


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
    q = _qorder_prism_exact(m)
    s = 0.0
    for pt, wt in zip(Qpts, Qwts):
        s += wt * f.subs([(x, pt[0]), (y, pt[1]), (z, pt[2])])
    assert np.isclose(float(q), float(s))


@functools.cache
def _qorder_pyramid_exact(m):
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
    return [(f, sympy.integrate(f, (x, 0, 1 - z), (y, 0, 1 - z), (z, 0, 1))) for f in polyset]


@pytest.mark.parametrize("m", range(6))
@pytest.mark.parametrize(
    "scheme", [basix.QuadratureType.default, basix.QuadratureType.gauss_jacobi]
)
def test_qorder_pyramid(m, scheme):
    Qpts, Qwts = basix.make_quadrature(basix.CellType.pyramid, m, rule=scheme)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    for f, q in _qorder_pyramid_exact(m):
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


_WITHERDEN_VINCENT_MAX_DEGREE = {
    basix.CellType.triangle: 20,
    basix.CellType.quadrilateral: 21,
    basix.CellType.tetrahedron: 10,
    basix.CellType.hexahedron: 11,
    basix.CellType.prism: 10,
    basix.CellType.pyramid: 10,
}


def _monomial_integral(celltype, e):
    """Integral of a monomial with exponents e over a reference cell."""
    f = sympy.factorial
    if celltype == basix.CellType.triangle:
        a, b = e
        return sympy.Rational(f(a) * f(b), f(a + b + 2))
    elif celltype == basix.CellType.quadrilateral:
        a, b = e
        return sympy.Rational(1, (a + 1) * (b + 1))
    elif celltype == basix.CellType.tetrahedron:
        a, b, c = e
        return sympy.Rational(f(a) * f(b) * f(c), f(a + b + c + 3))
    elif celltype == basix.CellType.hexahedron:
        a, b, c = e
        return sympy.Rational(1, (a + 1) * (b + 1) * (c + 1))
    elif celltype == basix.CellType.prism:
        a, b, c = e
        return sympy.Rational(f(a) * f(b), f(a + b + 2) * (c + 1))
    else:
        assert celltype == basix.CellType.pyramid
        a, b, c = e
        return sympy.Rational(f(c) * f(a + b + 2), (a + 1) * (b + 1) * f(a + b + c + 3))


def _monomials(tdim, degree):
    """Exponents of the monomials of total degree at most ``degree``."""
    if tdim == 2:
        return [(a, b) for a in range(degree + 1) for b in range(degree + 1 - a)]
    else:
        return [
            (a, b, c)
            for a in range(degree + 1)
            for b in range(degree + 1 - a)
            for c in range(degree + 1 - a - b)
        ]


@pytest.mark.parametrize(
    ("celltype", "m"),
    [(c, m) for c, d in _WITHERDEN_VINCENT_MAX_DEGREE.items() for m in range(1, d + 1)],
)
def test_witherden_vincent(celltype, m):
    Qpts, Qwts = basix.make_quadrature(celltype, m, rule=basix.QuadratureType.witherden_vincent)
    assert (Qwts > 0).all()
    for e in _monomials(Qpts.shape[1], m):
        s = (Qwts * np.prod(Qpts**e, axis=1)).sum()
        assert np.isclose(s, float(_monomial_integral(celltype, e)))


@pytest.mark.parametrize("celltype", _WITHERDEN_VINCENT_MAX_DEGREE)
def test_witherden_vincent_degree_too_high(celltype):
    m = _WITHERDEN_VINCENT_MAX_DEGREE[celltype] + 1
    with pytest.raises(RuntimeError):
        basix.make_quadrature(celltype, m, rule=basix.QuadratureType.witherden_vincent)


def test_witherden_vincent_unsupported_cell():
    with pytest.raises(RuntimeError):
        basix.make_quadrature(basix.CellType.point, 1, rule=basix.QuadratureType.witherden_vincent)
