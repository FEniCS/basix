# Copyright (c) 2020 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import pytest


@pytest.mark.parametrize("element_name", [
    "Lagrange", "Discontinuous Lagrange"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_interval_permutation_size(element_name, order):
    e = basix.create_element(element_name, "interval", order)
    assert len(e.base_permutations) == 0


@pytest.mark.parametrize("element_name", [
    "Lagrange", "Discontinuous Lagrange",
    "Nedelec 1st kind H(curl)", "Nedelec 2nd kind H(curl)",
    "Raviart-Thomas", "Regge", "Crouzeix-Raviart"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_triangle_permutation_size(element_name, order):
    if element_name == "Crouzeix-Raviart" and order != 1:
        pytest.xfail()

    e = basix.create_element(element_name, "triangle", order)
    assert len(e.base_permutations) == 3


@pytest.mark.parametrize("element_name", [
    "Lagrange", "Discontinuous Lagrange",
    "Nedelec 1st kind H(curl)", "Nedelec 2nd kind H(curl)",
    "Raviart-Thomas", "Regge", "Crouzeix-Raviart"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_tetrahedron_permutation_size(element_name, order):
    if element_name == "Crouzeix-Raviart" and order != 1:
        pytest.xfail()

    e = basix.create_element(element_name, "tetrahedron", order)
    assert len(e.base_permutations) == 14


@pytest.mark.parametrize("cell_name, element_name",
    [("tetrahedron", "Nedelec 2nd kind H(curl)")])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_non_zero(cell_name, element_name, order):
    if element_name == "Crouzeix-Raviart" and order != 1:
        pytest.xfail()

    e = basix.create_element(element_name, cell_name, order)

    for perm in e.base_permutations:
        for row in perm:
            print(row)
            assert max(abs(i) for i in row) > 1e-6
