# Copyright (c) 2021 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import numpy as np
import basix
import pytest


def run_test(lower_element, higher_element, power, value_size):
    l_points = lower_element.points
    l_eval = np.concatenate([
        l_points[:, 0] ** power if i == 0 else 0 * l_points[:, 0]
        for i in range(value_size)
    ])
    l_coeffs = lower_element.interpolation_matrix @ l_eval

    i_m = basix.interpolation.compute_interpolation_between_elements(lower_element, higher_element)
    h_coeffs = i_m @ l_coeffs

    h_points = higher_element.points
    h_eval = np.concatenate([
        h_points[:, 0] ** power if i == 0 else 0 * h_points[:, 0]
        for i in range(value_size)
    ])
    h_coeffs2 = higher_element.interpolation_matrix @ h_eval

    assert np.allclose(h_coeffs, h_coeffs2)


@pytest.mark.parametrize("cell_type", [basix.CellType.interval, basix.CellType.triangle, basix.CellType.tetrahedron,
                                       basix.CellType.quadrilateral, basix.CellType.hexahedron, basix.CellType.prism])
@pytest.mark.parametrize("orders", [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
def test_different_order_interpolation_lagrange(cell_type, orders):
    lower_element = basix.create_element(basix.ElementFamily.P, cell_type, orders[0], basix.LagrangeVariant.gll_warped)
    higher_element = basix.create_element(basix.ElementFamily.P, cell_type, orders[1], basix.LagrangeVariant.gll_warped)

    run_test(lower_element, higher_element, orders[0], lower_element.value_size)


@pytest.mark.parametrize("variant1", [basix.LagrangeVariant.equispaced, basix.LagrangeVariant.gll_warped,
                                      basix.LagrangeVariant.gll_isaac])
@pytest.mark.parametrize("variant2", [basix.LagrangeVariant.equispaced, basix.LagrangeVariant.gll_warped,
                                      basix.LagrangeVariant.gll_isaac])
@pytest.mark.parametrize("cell_type", [basix.CellType.interval, basix.CellType.triangle, basix.CellType.tetrahedron,
                                       basix.CellType.quadrilateral, basix.CellType.hexahedron, basix.CellType.prism])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_different_variant_interpolation(cell_type, order, variant1, variant2):
    lower_element = basix.create_element(basix.ElementFamily.P, cell_type, order, variant1)
    higher_element = basix.create_element(basix.ElementFamily.P, cell_type, order, variant2)

    run_test(lower_element, higher_element, order, lower_element.value_size)


@pytest.mark.parametrize("family, args", [
    [basix.ElementFamily.RT, tuple()],
    [basix.ElementFamily.N1E, tuple()],
    [basix.ElementFamily.BDM, tuple()],
    [basix.ElementFamily.N2E, tuple()],
])
@pytest.mark.parametrize("cell_type", [basix.CellType.triangle, basix.CellType.tetrahedron,
                                       basix.CellType.quadrilateral, basix.CellType.hexahedron])
@pytest.mark.parametrize("orders", [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
def test_different_order_interpolation_vector(family, args, cell_type, orders):
    lower_element = basix.create_element(family, cell_type, orders[0], *args)
    higher_element = basix.create_element(family, cell_type, orders[1], *args)

    run_test(lower_element, higher_element, orders[0] - 1, lower_element.value_size)


@pytest.mark.parametrize("family, args", [[basix.ElementFamily.Regge, tuple()]])
@pytest.mark.parametrize("cell_type", [basix.CellType.triangle, basix.CellType.tetrahedron])
@pytest.mark.parametrize("orders", [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
def test_different_order_interpolation_matrix(family, args, cell_type, orders):
    lower_element = basix.create_element(family, cell_type, orders[0], *args)
    higher_element = basix.create_element(family, cell_type, orders[1], *args)

    run_test(lower_element, higher_element, orders[0] - 1, lower_element.value_size)


@pytest.mark.parametrize("family1, args1", [
    [basix.ElementFamily.RT, tuple()],
    [basix.ElementFamily.N1E, tuple()],
    [basix.ElementFamily.BDM, tuple()],
    [basix.ElementFamily.N2E, tuple()],
])
@pytest.mark.parametrize("family2, args2", [
    [basix.ElementFamily.RT, tuple()],
    [basix.ElementFamily.N1E, tuple()],
    [basix.ElementFamily.BDM, tuple()],
    [basix.ElementFamily.N2E, tuple()],
])
@pytest.mark.parametrize("cell_type", [basix.CellType.triangle, basix.CellType.tetrahedron,
                                       basix.CellType.quadrilateral, basix.CellType.hexahedron])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_different_element_interpolation(family1, args1, family2, args2, cell_type, order):
    lower_element = basix.create_element(family1, cell_type, order, *args1)
    higher_element = basix.create_element(family2, cell_type, order, *args2)

    run_test(lower_element, higher_element, order - 1, lower_element.value_size)
