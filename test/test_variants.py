# Copyright (c) 2020 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import pytest
import numpy as np


@pytest.mark.parametrize("cell_name", ["triangle", "quadrilateral", "tetrahedron", "hexahedron"])
@pytest.mark.parametrize("element_name", ["Lagrange"])
@pytest.mark.parametrize("order", [1, 2])
def test_equal(element_name, cell_name, order):
    """Test that low order element variants are the same."""
    e = basix.create_element(element_name, cell_name, order, "equispaced")
    e2 = basix.create_element(element_name, cell_name, order, "Gauss-Lobatto-Legendre")

    tdim = len(e.entity_dofs) - 1
    N = 5

    if tdim == 1:
        points = np.array([[i / N] for i in range(N + 1)])
    elif tdim == 2:
        points = np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1 - i)])
    elif tdim == 3:
        points = np.array([[i / N, j / N, k / N]
                           for i in range(N + 1) for j in range(N + 1 - i) for k in range(N + 1 - i - j)])

    tab1 = e.tabulate_x(0, points)[0]
    tab2 = e2.tabulate_x(0, points)[0]

    assert np.allclose(tab1, tab2)


@pytest.mark.parametrize("cell_name", ["triangle", "quadrilateral", "tetrahedron", "hexahedron"])
@pytest.mark.parametrize("element_name", ["Lagrange"])
@pytest.mark.parametrize("order", [3, 4])
def test_not_equal(element_name, cell_name, order):
    """Test that higher order element variants are not the same."""
    e = basix.create_element(element_name, cell_name, order, "equispaced")
    e2 = basix.create_element(element_name, cell_name, order, "Gauss-Lobatto-Legendre")

    tdim = len(e.entity_dofs) - 1
    N = 5

    if tdim == 1:
        points = np.array([[i / N] for i in range(N + 1)])
    elif tdim == 2:
        points = np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1 - i)])
    elif tdim == 3:
        points = np.array([[i / N, j / N, k / N]
                           for i in range(N + 1) for j in range(N + 1 - i) for k in range(N + 1 - i - j)])

    tab1 = e.tabulate_x(0, points)[0]
    tab2 = e2.tabulate_x(0, points)[0]

    assert not np.allclose(tab1, tab2)
