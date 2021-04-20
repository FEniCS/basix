# Copyright (c) 2020 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import numpy as np
from .utils import parametrize_over_elements


@parametrize_over_elements(3)
def test_if_permutations(cell_name, element_name, order):
    e = basix.create_element(element_name, cell_name, order)

    for t in e.base_transformations():
        for row in t:
            a = np.argmax(row)
            if not np.isclose(row[a], 1) or not np.allclose(row[:a], 0) or not np.allclose(row[a + 1:], 0):
                assert not e.dof_transformations_are_permutations
                return
    assert e.dof_transformations_are_permutations


@parametrize_over_elements(3)
def test_if_identity(cell_name, element_name, order):
    e = basix.create_element(element_name, cell_name, order)

    for t in e.base_transformations():
        if not np.allclose(t, np.eye(t.shape[0])):
            assert not e.dof_transformations_are_identity
            return
    assert e.dof_transformations_are_identity


@parametrize_over_elements(5)
def test_non_zero(cell_name, element_name, order):
    e = basix.create_element(element_name, cell_name, order)
    for t in e.base_transformations():
        for row in t:
            assert max(abs(i) for i in row) > 1e-6


@parametrize_over_elements(5, "interval")
def test_interval_transformation_size(element_name, order):
    e = basix.create_element(element_name, "interval", order)
    assert len(e.base_transformations()) == 0


@parametrize_over_elements(5, "triangle")
def test_triangle_transformation_orders(element_name, order):
    e = basix.create_element(element_name, "triangle", order)
    bt = e.base_transformations()
    assert len(bt) == 3
    identity = np.identity(e.dim)
    for i, order in enumerate([2, 2, 2]):
        assert np.allclose(
            np.linalg.matrix_power(bt[i], order),
            identity)


@parametrize_over_elements(5, "tetrahedron")
def test_tetrahedron_transformation_orders(element_name, order):
    e = basix.create_element(element_name, "tetrahedron", order)
    bt = e.base_transformations()
    assert len(bt) == 14
    identity = np.identity(e.dim)
    for i, order in enumerate([2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 3, 2, 3, 2]):
        assert np.allclose(
            np.linalg.matrix_power(bt[i], order),
            identity)


@parametrize_over_elements(5, "quadrilateral")
def test_quadrilateral_transformation_orders(element_name, order):
    e = basix.create_element(element_name, "quadrilateral", order)
    bt = e.base_transformations()
    assert len(bt) == 4
    identity = np.identity(e.dim)
    for i, order in enumerate([2, 2, 2, 2]):
        assert np.allclose(
            np.linalg.matrix_power(bt[i], order),
            identity)


@parametrize_over_elements(5, "hexahedron")
def test_hexahedron_transformation_orders(element_name, order):
    e = basix.create_element(element_name, "hexahedron", order)
    bt = e.base_transformations()
    assert len(bt) == 24
    identity = np.identity(e.dim)
    for i, order in enumerate([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                               4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2]):
        assert np.allclose(
            np.linalg.matrix_power(bt[i], order),
            identity)


@parametrize_over_elements(5, "triangle")
def test_transformation_of_tabulated_data_triangle(element_name, order):
    e = basix.create_element(element_name, "triangle", order)
    bt = e.base_transformations()

    N = 4
    points = np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1 - i)])
    values = e.tabulate_x(0, points)[0]

    start = sum(e.entity_dofs[0])
    ndofs = e.entity_dofs[1][0]
    if ndofs != 0:
        # Check that the 0th transformation undoes the effect of reflecting edge 0
        reflected_points = np.array([[p[1], p[0]] for p in points])
        reflected_values = e.tabulate_x(0, reflected_points)[0]

        _J = np.array([[0, 1], [1, 0]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.map_push_forward(reflected_values, J, detJ, K)

        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose((bt[0].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])


@parametrize_over_elements(5, "quadrilateral")
def test_transformation_of_tabulated_data_quadrilateral(element_name, order):
    e = basix.create_element(element_name, "quadrilateral", order)
    bt = e.base_transformations()

    N = 4
    points = np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1)])
    values = e.tabulate_x(0, points)[0]

    start = sum(e.entity_dofs[0])
    ndofs = e.entity_dofs[1][0]
    if ndofs != 0:
        # Check that the 0th transformation undoes the effect of reflecting edge 0
        reflected_points = np.array([[1 - p[0], p[1]] for p in points])
        reflected_values = e.tabulate_x(0, reflected_points)[0]

        _J = np.array([[-1, 0], [0, 1]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.map_push_forward(reflected_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose((bt[0].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])


@parametrize_over_elements(5, "tetrahedron")
def test_transformation_of_tabulated_data_tetrahedron(element_name, order):
    e = basix.create_element(element_name, "tetrahedron", order)
    bt = e.base_transformations()

    N = 4
    points = np.array([[i / N, j / N, k / N]
                       for i in range(N + 1) for j in range(N + 1 - i) for k in range(N + 1 - i - j)])
    values = e.tabulate_x(0, points)[0]

    start = sum(e.entity_dofs[0])
    ndofs = e.entity_dofs[1][0]
    if ndofs != 0:
        # Check that the 0th transformation undoes the effect of reflecting edge 0
        reflected_points = np.array([[p[0], p[2], p[1]] for p in points])
        reflected_values = e.tabulate_x(0, reflected_points)[0]

        _J = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.map_push_forward(reflected_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose((bt[0].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])

    start = sum(e.entity_dofs[0]) + sum(e.entity_dofs[1])
    ndofs = e.entity_dofs[2][0]
    if ndofs != 0:
        # Check that the 6th transformation undoes the effect of rotating face 0
        rotated_points = np.array([[p[2], p[0], p[1]] for p in points])
        rotated_values = e.tabulate_x(0, rotated_points)[0]

        _J = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.map_push_forward(rotated_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose(bt[6].dot(i_slice)[start: start + ndofs],
                                   j_slice[start: start + ndofs])

    if ndofs != 0:
        # Check that the 7th transformation undoes the effect of reflecting face 0
        reflected_points = np.array([[p[0], p[2], p[1]] for p in points])
        reflected_values = e.tabulate_x(0, reflected_points)[0]

        _J = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.map_push_forward(reflected_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose((bt[7].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])


# @parametrize_over_elements(5, "hexahedron")
@parametrize_over_elements(2, "hexahedron")
def test_transformation_of_tabulated_data_hexahedron(element_name, order):
    e = basix.create_element(element_name, "hexahedron", order)
    bt = e.base_transformations()

    N = 4
    points = np.array([[i / N, j / N, k / N]
                       for i in range(N + 1) for j in range(N + 1) for k in range(N + 1)])
    values = e.tabulate_x(0, points)[0]

    start = sum(e.entity_dofs[0])
    ndofs = e.entity_dofs[1][0]
    if ndofs != 0:
        # Check that the 0th transformation undoes the effect of reflecting edge 0
        reflected_points = np.array([[1 - p[0], p[1], p[2]] for p in points])
        reflected_values = e.tabulate_x(0, reflected_points)[0]

        _J = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.map_push_forward(reflected_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose((bt[0].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs], rtol=1e-4)

    start = sum(e.entity_dofs[0]) + sum(e.entity_dofs[1])
    ndofs = e.entity_dofs[2][0]
    if ndofs != 0:
        # Check that the 12th transformation undoes the effect of rotating face 0
        rotated_points = np.array([[1 - p[1], p[0], p[2]] for p in points])
        rotated_values = e.tabulate_x(0, rotated_points)[0]

        _J = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.map_push_forward(rotated_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose(bt[12].dot(i_slice)[start: start + ndofs],
                                   j_slice[start: start + ndofs], rtol=1e-4)

    if ndofs != 0:
        # Check that the 13th transformation undoes the effect of reflecting face 0
        reflected_points = np.array([[p[1], p[0], p[2]] for p in points])
        reflected_values = e.tabulate_x(0, reflected_points)[0]

        _J = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.map_push_forward(reflected_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose((bt[13].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs], rtol=1e-4)
