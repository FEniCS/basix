# Copyright (c) 2024 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import numpy as np
import pytest


@pytest.mark.parametrize("degree", range(1, 5))
@pytest.mark.parametrize(
    "cell",
    [
        basix.CellType.interval,
        basix.CellType.quadrilateral,
        basix.CellType.hexahedron,
        basix.CellType.triangle,
    ],
)
def test_iso_element(degree, cell):
    if cell == basix.CellType.triangle and degree > 2:
        pytest.xfail("Degree > 2 edge macro polysets not implemented on triangles.")
    e = basix.create_element(
        basix.ElementFamily.iso, cell, degree, basix.LagrangeVariant.gll_warped
    )
    e2 = basix.create_element(
        basix.ElementFamily.P, cell, 2 * degree, basix.LagrangeVariant.gll_warped
    )
    assert e.dim == e2.dim


def test_iso_interval_1():
    e = basix.create_element(basix.ElementFamily.iso, basix.CellType.interval, 1)
    pts = np.array([[i / 20] for i in range(21)])
    values = e.tabulate(0, pts)
    for n, p in enumerate(pts):
        if p[0] <= 0.5:
            assert np.isclose(values[0, n, 0, 0], 1 - 2 * p[0])
            assert np.isclose(values[0, n, 1, 0], 0.0)
            assert np.isclose(values[0, n, 2, 0], 2 * p[0])
        else:
            assert np.isclose(values[0, n, 0, 0], 0.0)
            assert np.isclose(values[0, n, 1, 0], 2 * p[0] - 1)
            assert np.isclose(values[0, n, 2, 0], 2 - 2 * p[0])


def test_iso_quadrilateral_1():
    e = basix.create_element(basix.ElementFamily.iso, basix.CellType.quadrilateral, 1)
    pts = np.array([[i / 15, j / 15] for i in range(16) for j in range(16)])
    values = e.tabulate(0, pts)
    for n, p in enumerate(pts):
        f_values = np.ones(9)
        if p[0] <= 0.5:
            f_values[0] *= 1 - 2 * p[0]
            f_values[5] *= 1 - 2 * p[0]
            f_values[2] *= 1 - 2 * p[0]
            f_values[1] *= 0
            f_values[3] *= 0
            f_values[6] *= 0
            f_values[4] *= 2 * p[0]
            f_values[8] *= 2 * p[0]
            f_values[7] *= 2 * p[0]
        else:
            f_values[0] *= 0
            f_values[5] *= 0
            f_values[2] *= 0
            f_values[1] *= 2 * p[0] - 1
            f_values[3] *= 2 * p[0] - 1
            f_values[6] *= 2 * p[0] - 1
            f_values[4] *= 2 - 2 * p[0]
            f_values[8] *= 2 - 2 * p[0]
            f_values[7] *= 2 - 2 * p[0]
        if p[1] <= 0.5:
            f_values[0] *= 1 - 2 * p[1]
            f_values[4] *= 1 - 2 * p[1]
            f_values[1] *= 1 - 2 * p[1]
            f_values[2] *= 0
            f_values[7] *= 0
            f_values[3] *= 0
            f_values[5] *= 2 * p[1]
            f_values[8] *= 2 * p[1]
            f_values[6] *= 2 * p[1]
        else:
            f_values[0] *= 0
            f_values[4] *= 0
            f_values[1] *= 0
            f_values[2] *= 2 * p[1] - 1
            f_values[7] *= 2 * p[1] - 1
            f_values[3] *= 2 * p[1] - 1
            f_values[5] *= 2 - 2 * p[1]
            f_values[8] *= 2 - 2 * p[1]
            f_values[6] *= 2 - 2 * p[1]
        assert np.allclose(values[0, n, :, 0], f_values)
