import fiatx
import pytest
import numpy as np


def test_polynomial_basic():
    zero = fiatx.Polynomial.zero()
    one = fiatx.Polynomial.one()
    x = fiatx.Polynomial.x(1)

    w = one + x * x * 2.0
    result = w.tabulate([0.0,1.0,2.0,4.0])
    assert np.isclose(result,[1, 3, 9, 33]).all()

    w =  x * x * 2.0 - one
    w += w
    result = w.tabulate([0.0,1.0,2.0,4.0])
    assert np.isclose(result, [-2.,  2., 14., 62.]).all()

def test_polynomial_basic_2d():
    one = fiatx.Polynomial.one()
    x = fiatx.Polynomial.x(2)
    y = fiatx.Polynomial.y(2)

    w = x * y * (one - x) * (one - y)
    result = w.tabulate([[0.0, 0.0], [0.2, 0.3], [0.8, 0.4]])
    assert np.isclose(result, [0.0, 0.0336, 0.0384]).all()

def test_polynomial_basic_3d():
    one = fiatx.Polynomial.one()
    x = fiatx.Polynomial.x(3)
    y = fiatx.Polynomial.y(3)
    z = fiatx.Polynomial.z(3)

    w = x * y * z * (one - x) * (one - y) * (one - z)
    result = w.tabulate([[0.0, 0.0,0.0], [0.5, 0.5,0.5], [1.0, 1.0,1.0]])
    assert np.isclose(result, [0.0, 1.0/64.0, 0.0]).all()

def test_polynomial_differentiate():
    one = fiatx.Polynomial.one()
    x = fiatx.Polynomial.x(1)

    # 3x^3 + 2x^2 + 1
    w = one + x * x * 2.0 + x * x * x * 3.0

    # Should give 9x^2 + 4x
    dw = w.diff([1])

    result = dw.tabulate([0.0, 1.0, 2.0])
    assert np.isclose(result, [0.0, 13.0, 44.0]).all()

    # Should give 18x + 4
    dw = w.diff([2])
    result = dw.tabulate([0.0, 1.0, 2.0])
    assert np.isclose(result, [4.0, 22.0, 40.0]).all()

    x = fiatx.Polynomial.x(2)
    y = fiatx.Polynomial.y(2)
    # 4.0y^3 + 3x^2y + 2x^2 + 1
    w = one + x * x * 2.0 + x * x * y * 3.0 + y * y * y * 4.0

    # Should give 4x + 6xy
    dw = w.diff([1, 0])
    result = dw.tabulate([[0.0,0.0], [1.0,0.0], [0.0,2.0], [3.0,3.0]])
    assert np.isclose(result, [0.0, 4.0, 0.0, 66.0]).all()

    # Should give 3x^2 + 12y^2
    dw = w.diff([0, 1])
    result = dw.tabulate([[0.0,0.0], [1.0,0.0], [0.0,2.0], [3.0,3.0]])
    assert np.isclose(result, [0.0, 3.0, 48.0, 135.0]).all()

    # Should give 6x
    dw = w.diff([1, 1])
    result = dw.tabulate([[0.0,0.0], [1.0,0.0], [0.0,2.0], [3.0,3.0]])
    assert np.isclose(result, [0.0, 6.0, 0.0, 18.0]).all()
