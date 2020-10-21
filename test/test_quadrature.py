# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import libtab
import pytest
import numpy as np


@pytest.mark.parametrize("order", [1, 2, 4, 5, 8, 20, 40, 80])
def test_quadrature_interval(order):
    b = 7.0
    simplex = [[0], [b]]
    Qpts, Qwts = libtab.make_quadrature(simplex, order)
    w = sum(Qwts)
    assert np.isclose(w, b)


@pytest.mark.parametrize("order", [1, 2, 4, 20, 40])
def test_quadrature_triangle(order):
    b = 7.0
    h = 5.0
    simplex = [[0, 0], [b, 0], [0, h]]
    Qpts, Qwts = libtab.make_quadrature(simplex, order)
    w = sum(Qwts)
    assert np.isclose(w, 0.5 * b * h)


@pytest.mark.parametrize("order", [1, 2, 4, 20, 40])
def test_quadrature_tet(order):
    b = 7.0
    h = 5.0
    x = 3.0
    simplex = [[0, 0, 0], [b, 0, 0], [0, h, 0], [0, 0, x]]
    Qpts, Qwts = libtab.make_quadrature(simplex, order)
    w = sum(Qwts)
    assert np.isclose(w, b * h * x / 6.0)


def test_quadrature_function():
    simplex = [[0.0], [2.0]]
    Qpts, Qwts = libtab.make_quadrature(simplex, 3)

    def f(x):
        return x * x

    b = sum([w * f(pt[0]) for pt, w in zip(Qpts, Qwts)])

    assert np.isclose(b, 8.0 / 3.0)


def test_jacobi():
    pts = np.arange(0, 1, 0.1)
    f = libtab.compute_jacobi_deriv(1.0, 4, 5, pts)
    print(f)
