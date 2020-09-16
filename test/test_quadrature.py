import fiatx
import pytest
import numpy as np


@pytest.mark.parametrize("order", [1, 2, 3, 4, 6, 8, 10, 11])
def test_quadrature(order):
    b = 7.0
    h = 5.0
    simplex = [[0, 0], [b, 0], [0, h]]
    Qpts, Qwts = fiatx.make_quadrature(simplex, order)
    w = sum(Qwts)
    assert np.isclose(w, 0.5 * b * h)
