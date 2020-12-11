# Copyright (c) 2020 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import libtab
import pytest


@pytest.mark.parametrize("element_name", [
    "Lagrange", "Discontinuous Lagrange"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_interval_permutation_size(element_name, order):
    e = libtab.create_element(element_name, "interval", 1)
    assert len(e.base_permutations) == 0


@pytest.mark.parametrize("element_name", [
    "Lagrange", "Discontinuous Lagrange",
    "Nedelec 1st kind H(curl)", "Nedelec 2nd kind H(curl)",
    "Raviart-Thomas", "Regge", "Crouzeix-Raviart"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_triangle_permutation_size(element_name, order):
    if element_name == "Crouzeix-Raviart" and order != 1:
        pytest.xfail()

    e = libtab.create_element(element_name, "triangle", 1)
    assert len(e.base_permutations) == 3


@pytest.mark.parametrize("element_name", [
    "Lagrange", "Discontinuous Lagrange",
    "Nedelec 1st kind H(curl)", "Nedelec 2nd kind H(curl)",
    "Raviart-Thomas", "Regge", "Crouzeix-Raviart"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_tetrahedron_permutation_size(element_name, order):
    if element_name == "Crouzeix-Raviart" and order != 1:
        pytest.xfail()

    e = libtab.create_element(element_name, "tetrahedron", 1)
    assert len(e.base_permutations) == 14
