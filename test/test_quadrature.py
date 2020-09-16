import fiatx
import pytest
import numpy as np


@pytest.mark.parametrize("order", [1, 2, 3, 4, 6, 8, 10, 11])
def test_quadrature(order):
    h = 5.0
    w = 7.0
    simplex = [[0, 0], [h, 0], [0, w]]
    Qpts, Qwts = fiatx.make_quadrature(simplex, order)
    w = sum(Qwts)
    assert np.isclose(w, 0.5 * w * h)
