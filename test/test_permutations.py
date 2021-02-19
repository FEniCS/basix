# Copyright (c) 2020 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import pytest
import numpy as np
from .utils import parametrize_over_elements

interval_elements = ["Lagrange", "Discontinuous Lagrange"]
triangle_elements = [
    "Lagrange", "Discontinuous Lagrange",
    "Nedelec 1st kind H(curl)", "Nedelec 2nd kind H(curl)",
    "Raviart-Thomas", "Regge", "Crouzeix-Raviart"]
tetrahedron_elements = [
    "Lagrange", "Discontinuous Lagrange",
    "Nedelec 1st kind H(curl)", "Nedelec 2nd kind H(curl)",
    "Raviart-Thomas", "Regge", "Crouzeix-Raviart"]
quadrilateral_elements = [
    "Lagrange", "Raviart-Thomas", "Nedelec 1st kind H(curl)"]
hexahedron_elements = [
    "Lagrange", "Raviart-Thomas", "Nedelec 1st kind H(curl)"]

all_elements = [(cell, e) for cell, elements in [
    ("interval", interval_elements),
    ("triangle", triangle_elements),
    ("quadrilateral", quadrilateral_elements),
    ("tetrahedron", tetrahedron_elements),
    ("hexahedron", hexahedron_elements)] for e in elements]


@parametrize_over_elements(5)
def test_non_zero(cell_name, element_name, order):
    e = basix.create_element(element_name, cell_name, order)

    for perm in e.base_permutations:
        for row in perm:
            assert max(abs(i) for i in row) > 1e-6


@parametrize_over_elements(5, "interval")
def test_interval_permutation_size(element_name, order):
    e = basix.create_element(element_name, "interval", order)
    assert len(e.base_permutations) == 0


@parametrize_over_elements(5, "triangle")
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


@parametrize_over_elements(5, "tetrahedron")
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


@parametrize_over_elements(5, "quadrilateral")
def test_quadrilateral_permutation_orders(element_name, order):
    e = basix.create_element(element_name, "quadrilateral", order)
    assert len(e.base_permutations) == 4

    identity = np.identity(e.dim)
    for i, order in enumerate([2, 2, 2, 2]):
        assert np.allclose(
            np.linalg.matrix_power(e.base_permutations[i], order),
            identity)


@parametrize_over_elements(5, "hexahedron")
def test_hexahedron_permutation_orders(element_name, order):
    e = basix.create_element(element_name, "hexahedron", order)
    assert len(e.base_permutations) == 24

    identity = np.identity(e.dim)
    for i, order in enumerate([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                               4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2]):
        assert np.allclose(
            np.linalg.matrix_power(e.base_permutations[i], order),
            identity)


@parametrize_over_elements(5, "triangle")
def test_permutation_of_tabulated_data_triangle(element_name, order):
    if element_name == "Crouzeix-Raviart" and order != 1:
        pytest.xfail()
    if element_name == "Regge":
        pytest.skip("DOF permutations not yet implemented for Regge elements.")

    e = basix.create_element(element_name, "triangle", order)

    N = 4
    points = np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1 - i)])
    values = e.tabulate(0, points)[0]

    start = sum(e.entity_dofs[0])
    ndofs = e.entity_dofs[1][0]
    if ndofs != 0:
        # Check that the 0th permutation undoes the effect of reflecting edge 0
        reflected_points = np.array([[p[1], p[0]] for p in points])
        reflected_values = e.tabulate(0, reflected_points)[0]

        _J = np.array([[0, 1], [1, 0]])
        J = np.array([_J.reshape(4) for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J).reshape(4) for p in points])
        mapped_values = e.map_push_forward(reflected_values, J, detJ, K)

        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[d * e.dim:(d + 1) * e.dim]
                j_slice = j[d * e.dim:(d + 1) * e.dim]
                assert np.allclose((e.base_permutations[0].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])


@parametrize_over_elements(5, "quadrilateral")
def test_permutation_of_tabulated_data_quadrilateral(element_name, order):
    e = basix.create_element(element_name, "quadrilateral", order)

    N = 4
    points = np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1)])
    values = e.tabulate(0, points)[0]

    start = sum(e.entity_dofs[0])
    ndofs = e.entity_dofs[1][0]
    if ndofs != 0:
        # Check that the 0th permutation undoes the effect of reflecting edge 0
        reflected_points = np.array([[1 - p[0], p[1]] for p in points])
        reflected_values = e.tabulate(0, reflected_points)[0]

        _J = np.array([[-1, 0], [0, 1]])
        J = np.array([_J.reshape(4) for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J).reshape(4) for p in points])
        mapped_values = e.map_push_forward(reflected_values, J, detJ, K)

        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[d * e.dim:(d + 1) * e.dim]
                j_slice = j[d * e.dim:(d + 1) * e.dim]
                assert np.allclose((e.base_permutations[0].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])


@parametrize_over_elements(5, "tetrahedron")
def test_permutation_of_tabulated_data_tetrahedron(element_name, order):
    if element_name == "Crouzeix-Raviart" and order != 1:
        pytest.xfail()
    if element_name == "Regge":
        pytest.skip("DOF permutations not yet implemented for Regge elements.")

    e = basix.create_element(element_name, "tetrahedron", order)

    N = 4
    points = np.array([[i / N, j / N, k / N]
                       for i in range(N + 1) for j in range(N + 1 - i) for k in range(N + 1 - i - j)])
    values = e.tabulate(0, points)[0]

    start = sum(e.entity_dofs[0])
    ndofs = e.entity_dofs[1][0]
    if ndofs != 0:
        # Check that the 0th permutation undoes the effect of reflecting edge 0
        reflected_points = np.array([[p[0], p[2], p[1]] for p in points])
        reflected_values = e.tabulate(0, reflected_points)[0]

        _J = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        J = np.array([_J.reshape(9) for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J).reshape(9) for p in points])
        mapped_values = e.map_push_forward(reflected_values, J, detJ, K)

        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[d * e.dim:(d + 1) * e.dim]
                j_slice = j[d * e.dim:(d + 1) * e.dim]
                assert np.allclose((e.base_permutations[0].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])

    start = sum(e.entity_dofs[0]) + sum(e.entity_dofs[1])
    ndofs = e.entity_dofs[2][0]
    if ndofs != 0:
        # Check that the 6th permutation undoes the effect of rotating face 0
        rotated_points = np.array([[p[2], p[0], p[1]] for p in points])
        rotated_values = e.tabulate(0, rotated_points)[0]

        _J = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        J = np.array([_J.reshape(9) for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J).reshape(9) for p in points])
        mapped_values = e.map_push_forward(rotated_values, J, detJ, K)

        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[d * e.dim:(d + 1) * e.dim]
                j_slice = j[d * e.dim:(d + 1) * e.dim]
                assert np.allclose(e.base_permutations[6].dot(i_slice)[start: start + ndofs],
                                   j_slice[start: start + ndofs])

    if ndofs != 0:
        # Check that the 7th permutation undoes the effect of reflecting face 0
        reflected_points = np.array([[p[0], p[2], p[1]] for p in points])
        reflected_values = e.tabulate(0, reflected_points)[0]

        _J = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        J = np.array([_J.reshape(9) for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J).reshape(9) for p in points])
        mapped_values = e.map_push_forward(reflected_values, J, detJ, K)

        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[d * e.dim:(d + 1) * e.dim]
                j_slice = j[d * e.dim:(d + 1) * e.dim]
                assert np.allclose((e.base_permutations[7].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])


@parametrize_over_elements(5, "hexahedron")
def test_permutation_of_tabulated_data_hexahedron(element_name, order):
    if order > 4 and element_name in ["Raviart-Thomas", "Nedelec 1st kind H(curl)"]:
        pytest.xfail("High order Hdiv and Hcurl spaces on hexes based on "
                     "Lagrange spaces equally spaced points are unstable.")

    e = basix.create_element(element_name, "hexahedron", order)

    N = 4
    points = np.array([[i / N, j / N, k / N]
                       for i in range(N + 1) for j in range(N + 1) for k in range(N + 1)])
    values = e.tabulate(0, points)[0]

    start = sum(e.entity_dofs[0])
    ndofs = e.entity_dofs[1][0]
    if ndofs != 0:
        # Check that the 0th permutation undoes the effect of reflecting edge 0
        reflected_points = np.array([[1 - p[0], p[1], p[2]] for p in points])
        reflected_values = e.tabulate(0, reflected_points)[0]

        _J = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        J = np.array([_J.reshape(9) for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J).reshape(9) for p in points])
        mapped_values = e.map_push_forward(reflected_values, J, detJ, K)

        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[d * e.dim:(d + 1) * e.dim]
                j_slice = j[d * e.dim:(d + 1) * e.dim]
                assert np.allclose((e.base_permutations[0].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])

    start = sum(e.entity_dofs[0]) + sum(e.entity_dofs[1])
    ndofs = e.entity_dofs[2][0]
    if ndofs != 0:
        # Check that the 12th permutation undoes the effect of rotating face 0
        rotated_points = np.array([[1 - p[1], p[0], p[2]] for p in points])
        rotated_values = e.tabulate(0, rotated_points)[0]

        _J = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        J = np.array([_J.reshape(9) for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J).reshape(9) for p in points])
        mapped_values = e.map_push_forward(rotated_values, J, detJ, K)

        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[d * e.dim:(d + 1) * e.dim]
                j_slice = j[d * e.dim:(d + 1) * e.dim]
                assert np.allclose(e.base_permutations[12].dot(i_slice)[start: start + ndofs],
                                   j_slice[start: start + ndofs])

    if ndofs != 0:
        # Check that the 13th permutation undoes the effect of reflecting face 0
        reflected_points = np.array([[p[1], p[0], p[2]] for p in points])
        reflected_values = e.tabulate(0, reflected_points)[0]

        _J = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        J = np.array([_J.reshape(9) for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J).reshape(9) for p in points])
        mapped_values = e.map_push_forward(reflected_values, J, detJ, K)

        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[d * e.dim:(d + 1) * e.dim]
                j_slice = j[d * e.dim:(d + 1) * e.dim]
                assert np.allclose((e.base_permutations[13].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])
