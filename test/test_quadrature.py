# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import pytest
import numpy as np


@pytest.mark.parametrize("celltype", [(basix.CellType.quadrilateral, 1.0),
                                      (basix.CellType.hexahedron, 1.0),
                                      (basix.CellType.prism, 0.5),
                                      (basix.CellType.interval, 1.0),
                                      (basix.CellType.triangle, 0.5),
                                      (basix.CellType.tetrahedron, 1.0/6.0)])
@pytest.mark.parametrize("order", [1, 2, 4, 8])
def test_cell_quadrature(celltype, order):
    Qpts, Qwts = basix.make_quadrature("default", celltype[0], order)
    print(sum(Qwts))
    assert(np.isclose(sum(Qwts), celltype[1]))


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
