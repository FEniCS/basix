import basix
import numpy as np


def test_tabulate_polyset():
    pts = [[p, p/2.0, 0.5] for p in np.linspace(0, 1, 100000)]
    basix._basixcpp.tabulate_polynomial_set(basix.CellType.tetrahedron, 3, 0, pts)
