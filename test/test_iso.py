import numpy as np
import basix
import pytest


@pytest.mark.parametrize("degree", range(1, 5))
@pytest.mark.parametrize("cell", [basix.CellType.interval])
def test_iso_element(degree, cell):
    e = basix.create_element(basix.ElementFamily.iso, cell, degree, basix.LagrangeVariant.gll_warped)
    e2 = basix.create_element(basix.ElementFamily.P, cell, 2 * degree, basix.LagrangeVariant.gll_warped)

    assert e.dim == e2.dim


def test_iso_interval_1():
    e = basix.create_element(basix.ElementFamily.iso, basix.CellType.interval, 1)
    pts = np.array([[i/20] for i in range(21)])

    values = e.tabulate(0, pts)
    print(values.shape)

    for n, p in enumerate(pts):
        if p[0] <= 0.5:
            assert np.isclose(values[0, n, 0, 0], 1 - 2 * p[0])
            assert np.isclose(values[0, n, 1, 0], 0.0)
            assert np.isclose(values[0, n, 2, 0], 2 * p[0])
        else:
            assert np.isclose(values[0, n, 0, 0], 0.0)
            assert np.isclose(values[0, n, 1, 0], 2 * p[0] - 1)
            assert np.isclose(values[0, n, 2, 0], 2 - 2 * p[0])
