# Copyright (c) 2020 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import pytest
import numpy as np

interval_elements = ["Lagrange", "Discontinuous Lagrange"]
triangle_elements = [
    "Lagrange", "Discontinuous Lagrange",
    "Nedelec 1st kind H(curl)", "Nedelec 2nd kind H(curl)",
    "Raviart-Thomas", "Regge", "Crouzeix-Raviart"]
tetrahedron_elements = [
    "Lagrange", "Discontinuous Lagrange",
    "Nedelec 1st kind H(curl)", "Nedelec 2nd kind H(curl)",
    "Raviart-Thomas", "Regge", "Crouzeix-Raviart"]
quadrilateral_elements = ["Q"]
hexahedron_elements = ["Q"]


@pytest.mark.parametrize("element_name", interval_elements)
@pytest.mark.parametrize("order", [1, 2, 3])
def test_interval_permutation_size(element_name, order):
    e = basix.create_element(element_name, "interval", order)
    assert len(e.base_permutations) == 0


@pytest.mark.parametrize(
    "cell_name, element_name",
    [(cell, e) for cell, elements in [
        ("interval", interval_elements),
        ("triangle", triangle_elements),
        ("quadrilateral", quadrilateral_elements),
        ("tetrahedron", tetrahedron_elements),
        ("hexahedron", hexahedron_elements)]
     for e in elements])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_non_zero(cell_name, element_name, order):
    if element_name == "Crouzeix-Raviart" and order != 1:
        pytest.xfail()

    e = basix.create_element(element_name, cell_name, order)

    for perm in e.base_permutations:
        for row in perm:
            assert max(abs(i) for i in row) > 1e-6


@pytest.mark.parametrize("element_name", triangle_elements)
@pytest.mark.parametrize("order", [1, 2, 3])
def test_triangle_permutation_orders(element_name, order):
    if element_name == "Crouzeix-Raviart" and order != 1:
        pytest.xfail()

    e = basix.create_element(element_name, "triangle", order)
    assert len(e.base_permutations) == 3

    identity = np.identity(e.dim)
    for i, order in enumerate([2, 2, 2]):
        assert np.allclose(
            np.linalg.matrix_power(e.base_permutations[i], order),
            identity)


@pytest.mark.parametrize("element_name", tetrahedron_elements)
@pytest.mark.parametrize("order", [1, 2, 3])
def test_tetrahedron_permutation_orders(element_name, order):
    if element_name == "Crouzeix-Raviart" and order != 1:
        pytest.xfail()

    e = basix.create_element(element_name, "tetrahedron", order)
    assert len(e.base_permutations) == 14

    identity = np.identity(e.dim)
    for i, order in enumerate([2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 3, 2, 3, 2]):
        assert np.allclose(
            np.linalg.matrix_power(e.base_permutations[i], order),
            identity)


@pytest.mark.parametrize("element_name", quadrilateral_elements)
@pytest.mark.parametrize("order", [1, 2, 3])
def test_quadrilateral_permutation_orders(element_name, order):
    e = basix.create_element(element_name, "quadrilateral", order)
    assert len(e.base_permutations) == 4

    identity = np.identity(e.dim)
    for i, order in enumerate([2, 2, 2, 2]):
        assert np.allclose(
            np.linalg.matrix_power(e.base_permutations[i], order),
            identity)


@pytest.mark.parametrize("element_name", hexahedron_elements)
@pytest.mark.parametrize("order", [1, 2, 3])
def test_hexahedron_permutation_orders(element_name, order):
    e = basix.create_element(element_name, "hexahedron", order)
    assert len(e.base_permutations) == 24

    identity = np.identity(e.dim)
    for i, order in enumerate([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                               4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2]):
        assert np.allclose(
            np.linalg.matrix_power(e.base_permutations[i], order),
            identity)
