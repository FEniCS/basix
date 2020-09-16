import fiatx
import pytest

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_rt(dim, order):
    rt = fiatx.RaviartThomas(dim, order)
    print(rt)
