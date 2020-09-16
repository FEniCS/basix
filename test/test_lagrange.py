import fiatx
import pytest

@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_lagrange(dim, order):
    lagrange = fiatx.Lagrange(dim, order)
    print(lagrange)
