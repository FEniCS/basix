# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import fiatx
import pytest
import numpy as np


@pytest.mark.parametrize("order", [1, 2, 3, 4, 6, 8, 10, 11])
def test_quadrature_basic(order):
    b = 7.0
    h = 5.0
    simplex = [[0, 0], [b, 0], [0, h]]
    Qpts, Qwts = fiatx.make_quadrature(simplex, order)
    w = sum(Qwts)
    assert np.isclose(w, 0.5 * b * h)

    b = 7.0
    h = 5.0
    l = 3.0
    simplex = [[0, 0, 0], [b, 0, 0], [0, h, 0], [0, 0, l]]
    Qpts, Qwts = fiatx.make_quadrature(simplex, order)
    w = sum(Qwts)
    assert np.isclose(w, b * h * l / 6.0)


def test_quadrature_function():
    simplex = [[0.0], [2.0]]
    Qpts, Qwts = fiatx.make_quadrature(simplex, 3)

    def f(x):
        return x*x

    b = sum([w * f(pt[0])     for pt,w in zip(Qpts, Qwts)])

    assert np.isclose(b, 8.0/3.0)
