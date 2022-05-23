# Copyright (c) 2021 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import numpy as np
import basix
import pytest
from .utils import parametrize_over_elements


def run_test(lower_element, higher_element, power, value_size):
    l_points = lower_element.points
    l_eval = np.concatenate([
        l_points[:, 0] ** power if i == 0 else 0 * l_points[:, 0]
        for i in range(value_size)
    ])
    l_coeffs = lower_element.interpolation_matrix @ l_eval

    i_m = basix.compute_interpolation_operator(lower_element, higher_element)
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
@pytest.mark.parametrize("orders", [(1, 2), (2, 4), (4, 5)])
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
@pytest.mark.parametrize("order", [1, 4])
def test_different_variant_interpolation(cell_type, order, variant1, variant2):
    lower_element = basix.create_element(basix.ElementFamily.P, cell_type, order, variant1)
    higher_element = basix.create_element(basix.ElementFamily.P, cell_type, order, variant2)

    run_test(lower_element, higher_element, order, lower_element.value_size)


@pytest.mark.parametrize("family, args", [
    [basix.ElementFamily.RT, (basix.LagrangeVariant.legendre, )],
    [basix.ElementFamily.N1E, (basix.LagrangeVariant.legendre, )],
    [basix.ElementFamily.BDM, (basix.LagrangeVariant.legendre, basix.DPCVariant.legendre)],
    [basix.ElementFamily.N2E, (basix.LagrangeVariant.legendre, basix.DPCVariant.legendre)],
])
@pytest.mark.parametrize("cell_type", [basix.CellType.triangle, basix.CellType.tetrahedron,
                                       basix.CellType.quadrilateral, basix.CellType.hexahedron])
@pytest.mark.parametrize("orders", [(1, 2), (2, 4), (4, 5)])
def test_different_order_interpolation_vector(family, args, cell_type, orders):
    lower_element = basix.create_element(family, cell_type, orders[0], *args)
    higher_element = basix.create_element(family, cell_type, orders[1], *args)

    run_test(lower_element, higher_element, orders[0] - 1, lower_element.value_size)


@pytest.mark.parametrize("family, args", [[basix.ElementFamily.Regge, tuple()]])
@pytest.mark.parametrize("cell_type", [basix.CellType.triangle, basix.CellType.tetrahedron])
@pytest.mark.parametrize("orders", [(1, 2), (2, 4), (4, 5)])
def test_different_order_interpolation_matrix(family, args, cell_type, orders):
    lower_element = basix.create_element(family, cell_type, orders[0], *args)
    higher_element = basix.create_element(family, cell_type, orders[1], *args)

    run_test(lower_element, higher_element, orders[0] - 1, lower_element.value_size)


@pytest.mark.parametrize("family1, args1", [
    [basix.ElementFamily.RT, (basix.LagrangeVariant.legendre, )],
    [basix.ElementFamily.N1E, (basix.LagrangeVariant.legendre, )],
    [basix.ElementFamily.BDM, (basix.LagrangeVariant.legendre, basix.DPCVariant.legendre)],
    [basix.ElementFamily.N2E, (basix.LagrangeVariant.legendre, basix.DPCVariant.legendre)],
])
@pytest.mark.parametrize("family2, args2", [
    [basix.ElementFamily.RT, (basix.LagrangeVariant.legendre, )],
    [basix.ElementFamily.N1E, (basix.LagrangeVariant.legendre, )],
    [basix.ElementFamily.BDM, (basix.LagrangeVariant.legendre, basix.DPCVariant.legendre)],
    [basix.ElementFamily.N2E, (basix.LagrangeVariant.legendre, basix.DPCVariant.legendre)],
])
@pytest.mark.parametrize("cell_type", [basix.CellType.triangle, basix.CellType.tetrahedron,
                                       basix.CellType.quadrilateral, basix.CellType.hexahedron])
@pytest.mark.parametrize("order", [1, 4])
def test_different_element_interpolation(family1, args1, family2, args2, cell_type, order):
    lower_element = basix.create_element(family1, cell_type, order, *args1)
    higher_element = basix.create_element(family2, cell_type, order, *args2)

    run_test(lower_element, higher_element, order - 1, lower_element.value_size)


@pytest.mark.parametrize("cell_type", [basix.CellType.triangle, basix.CellType.tetrahedron,
                                       basix.CellType.quadrilateral, basix.CellType.hexahedron])
@pytest.mark.parametrize("order", [1, 4])
def test_blocked_interpolation(cell_type, order):
    """Test interpolation of Nedelec's componenets into a Lagrange space."""
    nedelec = basix.create_element(
        basix.ElementFamily.N2E, cell_type, order, basix.LagrangeVariant.legendre, basix.DPCVariant.legendre)
    lagrange = basix.create_element(basix.ElementFamily.P, cell_type, order, basix.LagrangeVariant.gll_isaac)

    n_points = nedelec.points
    if nedelec.value_size == 2:
        n_eval = np.concatenate([n_points[:, 0] ** order, n_points[:, 1] ** order])
    else:
        n_eval = np.concatenate([n_points[:, 0] ** order, 0 * n_points[:, 0], n_points[:, 1] ** order])
    n_coeffs = nedelec.interpolation_matrix @ n_eval

    l_points = lagrange.points
    if nedelec.value_size == 2:
        values = [l_points[:, 0] ** order, l_points[:, 1] ** order]
    else:
        values = [l_points[:, 0] ** order, 0 * l_points[:, 0], l_points[:, 1] ** order]
    l_coeffs = np.empty(lagrange.dim * nedelec.value_size)
    for i, v in enumerate(values):
        l_coeffs[i::nedelec.value_size] = v

    # Test interpolation from Nedelec to blocked Lagrange
    i_m = basix.compute_interpolation_operator(nedelec, lagrange)
    assert np.allclose(l_coeffs, i_m @ n_coeffs)

    # Test interpolation from blocked Lagrange to Nedelec
    i_m = basix.compute_interpolation_operator(lagrange, nedelec)
    assert np.allclose(n_coeffs, i_m @ l_coeffs)


@parametrize_over_elements(5)
def test_degree_bounds(cell_type, degree, element_type, element_args):
    np.random.seed(13)

    element = basix.create_element(element_type, cell_type, degree, *element_args)

    points = basix.create_lattice(cell_type, 10, basix.LatticeType.equispaced, True)
    tab = element.tabulate(0, points)[0]

    # Test that this element's basis functions are contained in Lagrange space with
    # degree element.highest_degree
    coeffs = np.random.rand(element.dim)
    values = np.array([tab[:, :, i] @ coeffs
                       for i in range(element.value_size)])

    if element.highest_degree >= 0:
        # The element being tested should be a subset of this Lagrange space
        lagrange = basix.create_element(
            basix.ElementFamily.P, cell_type, element.highest_degree, basix.LagrangeVariant.equispaced, True)
        lagrange_coeffs = basix.compute_interpolation_operator(element, lagrange) @ coeffs
        lagrange_tab = lagrange.tabulate(0, points)[0]
        lagrange_values = np.array([lagrange_tab[:, :, 0] @ lagrange_coeffs[i::element.value_size]
                                    for i in range(element.value_size)])

        assert np.allclose(values, lagrange_values)

    if element.highest_degree >= 1:
        # The element being tested should be NOT a subset of this Lagrange space
        lagrange = basix.create_element(
            basix.ElementFamily.P, cell_type, element.highest_degree - 1,
            basix.LagrangeVariant.equispaced, True)
        lagrange_coeffs = basix.compute_interpolation_operator(element, lagrange) @ coeffs
        lagrange_tab = lagrange.tabulate(0, points)[0]
        lagrange_values = np.array([lagrange_tab[:, :, 0] @ lagrange_coeffs[i::element.value_size]
                                    for i in range(element.value_size)])

        assert not np.allclose(values, lagrange_values)

    # Test that the basis functions of Lagrange space with degree element.highest_complete_degree
    # are contained in this space

    if element.highest_complete_degree >= 0:
        # This Lagrange space should be a subset to the element being tested
        lagrange = basix.create_element(
            basix.ElementFamily.P, cell_type, element.highest_complete_degree,
            basix.LagrangeVariant.equispaced, True)
        lagrange_coeffs = np.random.rand(lagrange.dim * element.value_size)
        lagrange_tab = lagrange.tabulate(0, points)[0]
        lagrange_values = np.array([lagrange_tab[:, :, 0] @ lagrange_coeffs[i::element.value_size]
                                    for i in range(element.value_size)])

        coeffs = basix.compute_interpolation_operator(lagrange, element) @ lagrange_coeffs
        values = np.array([tab[:, :, i] @ coeffs
                           for i in range(element.value_size)])

        assert np.allclose(values, lagrange_values)

    if element.highest_complete_degree >= -1:
        # This Lagrange space should NOT be a subset to the element being tested
        lagrange = basix.create_element(
            basix.ElementFamily.P, cell_type, element.highest_complete_degree + 1,
            basix.LagrangeVariant.equispaced, True)
        lagrange_coeffs = np.random.rand(lagrange.dim * element.value_size)
        lagrange_tab = lagrange.tabulate(0, points)[0]
        lagrange_values = np.array([lagrange_tab[:, :, 0] @ lagrange_coeffs[i::element.value_size]
                                    for i in range(element.value_size)])

        coeffs = basix.compute_interpolation_operator(lagrange, element) @ lagrange_coeffs
        values = np.array([tab[:, :, i] @ coeffs
                           for i in range(element.value_size)])

        assert not np.allclose(values, lagrange_values)
