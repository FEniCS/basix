# Copyright (c) 2020 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import pytest
import numpy as np
import random
from .utils import parametrize_over_elements


@parametrize_over_elements(3)
def test_if_permutations(cell_type, element_type, degree, element_args):
    e = basix.create_element(element_type, cell_type, degree, *element_args)

    for t in e.base_transformations():
        for row in t:
            a = np.argmax(row)
            if not np.isclose(row[a], 1) or not np.allclose(row[:a], 0) or not np.allclose(row[a + 1:], 0):
                assert not e.dof_transformations_are_permutations
                return
    assert e.dof_transformations_are_permutations


@parametrize_over_elements(3)
def test_if_identity(cell_type, element_type, degree, element_args):
    e = basix.create_element(element_type, cell_type, degree, *element_args)

    for t in e.base_transformations():
        if not np.allclose(t, np.eye(t.shape[0])):
            assert not e.dof_transformations_are_identity
            return
    assert e.dof_transformations_are_identity


@parametrize_over_elements(5)
def test_non_zero(cell_type, element_type, degree, element_args):
    e = basix.create_element(element_type, cell_type, degree, *element_args)
    for t in e.base_transformations():
        for row in t:
            assert max(abs(i) for i in row) > 1e-6


@parametrize_over_elements(5)
def test_apply_to_transpose(cell_type, element_type, degree, element_args):
    random.seed(42)

    e = basix.create_element(element_type, cell_type, degree, *element_args)
    size = e.dim

    for i in range(10):
        cell_info = random.randrange(2**30)

        data1 = np.array(list(range(size**2)), dtype=np.float32)
        data1 = e.apply_dof_transformation(data1, size, cell_info)
        data1 = data1.reshape((size, size))

        # This is the transpose of the data used above
        data2 = np.array([size * j + i for i in range(size) for j in range(size)], dtype=np.float32)
        data2 = e.apply_dof_transformation_to_transpose(data2, size, cell_info)
        data2 = data2.reshape((size, size))

        assert np.allclose(data1.transpose(), data2)


@parametrize_over_elements(5, basix.CellType.interval)
def test_interval_transformation_size(element_type, degree, element_args):
    e = basix.create_element(element_type, basix.CellType.interval, degree, *element_args)
    assert len(e.base_transformations()) == 0


@parametrize_over_elements(5, basix.CellType.triangle)
def test_triangle_transformation_degrees(element_type, degree, element_args):
    e = basix.create_element(element_type, basix.CellType.triangle, degree, *element_args)
    bt = e.base_transformations()
    assert len(bt) == 3
    identity = np.identity(e.dim)
    for i, degree in enumerate([2, 2, 2]):
        assert np.allclose(
            np.linalg.matrix_power(bt[i], degree),
            identity)


@parametrize_over_elements(5, basix.CellType.tetrahedron)
def test_tetrahedron_transformation_degrees(element_type, degree, element_args):
    e = basix.create_element(element_type, basix.CellType.tetrahedron, degree, *element_args)
    bt = e.base_transformations()
    assert len(bt) == 14
    identity = np.identity(e.dim)
    for i, degree in enumerate([2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 3, 2, 3, 2]):
        assert np.allclose(
            np.linalg.matrix_power(bt[i], degree),
            identity)


@parametrize_over_elements(5, basix.CellType.quadrilateral)
def test_quadrilateral_transformation_degrees(element_type, degree, element_args):
    e = basix.create_element(element_type, basix.CellType.quadrilateral, degree, *element_args)
    bt = e.base_transformations()
    assert len(bt) == 4
    identity = np.identity(e.dim)
    for i, degree in enumerate([2, 2, 2, 2]):
        assert np.allclose(
            np.linalg.matrix_power(bt[i], degree),
            identity)


@parametrize_over_elements(5, basix.CellType.hexahedron)
def test_hexahedron_transformation_degrees(element_type, degree, element_args):
    e = basix.create_element(element_type, basix.CellType.hexahedron, degree, *element_args)
    bt = e.base_transformations()
    assert len(bt) == 24
    identity = np.identity(e.dim)
    for i, degree in enumerate([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2]):
        # TODO: remove the atol here once non-equispaced variants are implemented
        assert np.allclose(
            np.linalg.matrix_power(bt[i], degree),
            identity, atol=1e-6)


@parametrize_over_elements(5, basix.CellType.prism)
def test_prism_transformation_degrees(element_type, degree, element_args):
    e = basix.create_element(element_type, basix.CellType.prism, degree, *element_args)
    bt = e.base_transformations()
    assert len(bt) == 19
    identity = np.identity(e.dim)
    for i, degree in enumerate([2, 2, 2, 2, 2, 2, 2, 2, 2,
                                3, 2, 4, 2, 4, 2, 4, 2, 3, 2]):
        assert np.allclose(
            np.linalg.matrix_power(bt[i], degree),
            identity)


@parametrize_over_elements(5, basix.CellType.pyramid)
def test_pyramid_transformation_degrees(element_type, degree, element_args):
    e = basix.create_element(element_type, basix.CellType.pyramid, degree, *element_args)
    bt = e.base_transformations()
    assert len(bt) == 18
    identity = np.identity(e.dim)
    for i, degree in enumerate([2, 2, 2, 2, 2, 2, 2, 2,
                                4, 2, 3, 2, 3, 2, 3, 2, 3, 2]):
        assert np.allclose(
            np.linalg.matrix_power(bt[i], degree),
            identity)


@parametrize_over_elements(5, basix.CellType.triangle)
def test_transformation_of_tabulated_data_triangle(element_type, degree, element_args):
    e = basix.create_element(element_type, basix.CellType.triangle, degree, *element_args)
    bt = e.base_transformations()

    N = 4
    points = np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1 - i)])
    values = e.tabulate(0, points)[0]

    start = sum(e.num_entity_dofs[0])
    ndofs = e.num_entity_dofs[1][0]
    if ndofs != 0:
        # Check that the 0th transformation undoes the effect of reflecting edge 0
        reflected_points = np.array([[p[1], p[0]] for p in points])
        reflected_values = e.tabulate(0, reflected_points)[0]

        _J = np.array([[0, 1], [1, 0]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.push_forward(reflected_values, J, detJ, K)

        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose((bt[0].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])


@parametrize_over_elements(5, basix.CellType.quadrilateral)
def test_transformation_of_tabulated_data_quadrilateral(element_type, degree, element_args):
    e = basix.create_element(element_type, basix.CellType.quadrilateral, degree, *element_args)
    bt = e.base_transformations()

    N = 4
    points = np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1)])
    values = e.tabulate(0, points)[0]

    start = sum(e.num_entity_dofs[0])
    ndofs = e.num_entity_dofs[1][0]
    if ndofs != 0:
        # Check that the 0th transformation undoes the effect of reflecting edge 0
        reflected_points = np.array([[1 - p[0], p[1]] for p in points])
        reflected_values = e.tabulate(0, reflected_points)[0]

        _J = np.array([[-1, 0], [0, 1]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.push_forward(reflected_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose((bt[0].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])


@parametrize_over_elements(5, basix.CellType.tetrahedron)
def test_transformation_of_tabulated_data_tetrahedron(element_type, degree, element_args):
    e = basix.create_element(element_type, basix.CellType.tetrahedron, degree, *element_args)
    bt = e.base_transformations()

    N = 4
    points = np.array([[i / N, j / N, k / N]
                       for i in range(N + 1) for j in range(N + 1 - i) for k in range(N + 1 - i - j)])
    values = e.tabulate(0, points)[0]

    start = sum(e.num_entity_dofs[0])
    ndofs = e.num_entity_dofs[1][0]
    if ndofs != 0:
        # Check that the 0th transformation undoes the effect of reflecting edge 0
        reflected_points = np.array([[p[0], p[2], p[1]] for p in points])
        reflected_values = e.tabulate(0, reflected_points)[0]

        _J = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.push_forward(reflected_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose((bt[0].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])

    start = sum(e.num_entity_dofs[0]) + sum(e.num_entity_dofs[1])
    ndofs = e.num_entity_dofs[2][0]
    if ndofs != 0:
        # Check that the 6th transformation undoes the effect of rotating face 0
        rotated_points = np.array([[p[2], p[0], p[1]] for p in points])
        rotated_values = e.tabulate(0, rotated_points)[0]

        _J = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.push_forward(rotated_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose(bt[6].dot(i_slice)[start: start + ndofs],
                                   j_slice[start: start + ndofs])

    if ndofs != 0:
        # Check that the 7th transformation undoes the effect of reflecting face 0
        reflected_points = np.array([[p[0], p[2], p[1]] for p in points])
        reflected_values = e.tabulate(0, reflected_points)[0]

        _J = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.push_forward(reflected_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose((bt[7].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])


@parametrize_over_elements(3, basix.CellType.hexahedron)
def test_transformation_of_tabulated_data_hexahedron(element_type, degree, element_args):
    if degree > 4 and element_type in [basix.ElementFamily.RT, basix.ElementFamily.N1E]:
        pytest.xfail("High degree Hdiv and Hcurl spaces on hexes based on "
                     "Lagrange spaces equally spaced points are unstable.")

    e = basix.create_element(element_type, basix.CellType.hexahedron, degree, *element_args)
    bt = e.base_transformations()

    N = 4
    points = np.array([[i / N, j / N, k / N]
                       for i in range(N + 1) for j in range(N + 1) for k in range(N + 1)])
    values = e.tabulate(0, points)[0]

    start = sum(e.num_entity_dofs[0])
    ndofs = e.num_entity_dofs[1][0]
    if ndofs != 0:
        # Check that the 0th transformation undoes the effect of reflecting edge 0
        reflected_points = np.array([[1 - p[0], p[1], p[2]] for p in points])
        reflected_values = e.tabulate(0, reflected_points)[0]

        _J = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.push_forward(reflected_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose((bt[0].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])

    start = sum(e.num_entity_dofs[0]) + sum(e.num_entity_dofs[1])
    ndofs = e.num_entity_dofs[2][0]
    if ndofs != 0:
        # Check that the 12th transformation undoes the effect of rotating face 0
        rotated_points = np.array([[1 - p[1], p[0], p[2]] for p in points])
        rotated_values = e.tabulate(0, rotated_points)[0]

        _J = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.push_forward(rotated_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose(bt[12].dot(i_slice)[start: start + ndofs],
                                   j_slice[start: start + ndofs])

    if ndofs != 0:
        # Check that the 13th transformation undoes the effect of reflecting face 0
        reflected_points = np.array([[p[1], p[0], p[2]] for p in points])
        reflected_values = e.tabulate(0, reflected_points)[0]

        _J = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.push_forward(reflected_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose((bt[13].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])


@parametrize_over_elements(3, basix.CellType.prism)
def test_transformation_of_tabulated_data_prism(element_type, degree, element_args):
    e = basix.create_element(element_type, basix.CellType.prism, degree, *element_args)
    bt = e.base_transformations()

    N = 4
    points = np.array([[i / N, j / N, k / N]
                       for i in range(N + 1) for j in range(N + 1 - i) for k in range(N + 1)])
    values = e.tabulate(0, points)[0]

    start = sum(e.num_entity_dofs[0])
    ndofs = e.num_entity_dofs[1][0]
    if ndofs != 0:
        # Check that the 0th transformation undoes the effect of reflecting edge 0
        reflected_points = np.array([[1 - p[1] - p[0], p[1], p[2]] for p in points])
        reflected_values = e.tabulate(0, reflected_points)[0]

        _J = np.array([[-1, 0, 0], [-1, 1, 0], [0, 0, 1]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.push_forward(reflected_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose((bt[0].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])

    start = sum(e.num_entity_dofs[0]) + sum(e.num_entity_dofs[1])
    ndofs = e.num_entity_dofs[2][0]
    if ndofs != 0:
        # Check that the 10th transformation undoes the effect of rotating face 0
        rotated_points = np.array([[1 - p[0] - p[1], p[0], p[2]] for p in points])
        rotated_values = e.tabulate(0, rotated_points)[0]

        _J = np.array([[-1, 1, 0], [-1, 0, 0], [0, 0, 1]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.push_forward(rotated_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose(bt[10].dot(i_slice)[start: start + ndofs],
                                   j_slice[start: start + ndofs])

    if ndofs != 0:
        # Check that the 11th transformation undoes the effect of reflecting face 0
        reflected_points = np.array([[p[1], p[0], p[2]] for p in points])
        reflected_values = e.tabulate(0, reflected_points)[0]

        _J = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.push_forward(reflected_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose((bt[11].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])


@parametrize_over_elements(3, basix.CellType.pyramid)
def test_transformation_of_tabulated_data_pyramid(element_type, degree, element_args):
    e = basix.create_element(element_type, basix.CellType.pyramid, degree, *element_args)
    bt = e.base_transformations()

    N = 4
    points = np.array([[i / N, j / N, k / N]
                       for i in range(N + 1) for j in range(N + 1 - i) for k in range(N + 1)])
    values = e.tabulate(0, points)[0]

    start = sum(e.num_entity_dofs[0])
    ndofs = e.num_entity_dofs[1][0]
    if ndofs != 0:
        # Check that the 0th transformation undoes the effect of reflecting edge 0
        reflected_points = np.array([[1 - p[2] - p[0], p[1], p[2]] for p in points])
        reflected_values = e.tabulate(0, reflected_points)[0]

        _J = np.array([[-1, 0, 0], [0, 1, 0], [-1, 0, 1]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.push_forward(reflected_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose((bt[0].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])

    start = sum(e.num_entity_dofs[0]) + sum(e.num_entity_dofs[1])
    ndofs = e.num_entity_dofs[2][0]
    if ndofs != 0:
        # Check that the 8th transformation undoes the effect of rotating face 0
        rotated_points = np.array([[1 - p[1] - p[2], p[0], p[2]] for p in points])
        rotated_values = e.tabulate(0, rotated_points)[0]

        _J = np.array([[0, 1, 0], [-1, 0, 0], [-1, 0, 1]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.push_forward(rotated_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose(bt[8].dot(i_slice)[start: start + ndofs],
                                   j_slice[start: start + ndofs])

    if ndofs != 0:
        # Check that the 9th transformation undoes the effect of reflecting face 0
        reflected_points = np.array([[p[1], p[0], p[2]] for p in points])
        reflected_values = e.tabulate(0, reflected_points)[0]

        _J = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = e.push_forward(reflected_values, J, detJ, K)
        for i, j in zip(values, mapped_values):
            for d in range(e.value_size):
                i_slice = i[:, d]
                j_slice = j[:, d]
                assert np.allclose((bt[9].dot(i_slice))[start: start + ndofs],
                                   j_slice[start: start + ndofs])
