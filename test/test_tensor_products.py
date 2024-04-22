# Copyright (c) 2021 Matthew Scroggs, Igor A. Baratta
# FEniCS Project
# SPDX-License-Identifier: MIT

from itertools import product

import basix
import numpy as np
import pytest

from .utils import parametrize_over_elements


def tensor_product(*data):
    if len(data) == 1:
        return data[0]
    if len(data) > 2:
        return tensor_product(tensor_product(data[0], data[1]), *data[2:])

    a, b = data
    return np.outer(a, b).reshape(-1)


@parametrize_over_elements(4)
def test_tensor_product_factorisation(cell_type, degree, element_type, element_args):
    try:
        element = basix.create_tp_element(element_type, cell_type, degree, *element_args)
    except RuntimeError:
        # These elements should have a factorisation
        if (
            cell_type in [basix.CellType.quadrilateral, basix.CellType.hexahedron]
            and element_type in [basix.ElementFamily.P]
            and basix.LagrangeVariant.equispaced in element_args
        ):
            raise RuntimeError("Could not create tensor product element")
        pytest.skip()

    assert element.has_tensor_product_factorisation

    tdim = len(basix.topology(cell_type)) - 1

    factors = element.get_tensor_product_representation()

    lattice = basix.create_lattice(cell_type, 1, basix.LatticeType.equispaced, True)
    tab1 = element.tabulate(1, lattice)

    for i, point in enumerate(lattice):
        for ds in product(range(2), repeat=tdim):
            if sum(ds) > 1:
                continue
            deriv = basix.index(*ds)
            values1 = tab1[deriv, i, :, 0]

            for fs in factors:
                evals = [
                    e.tabulate(d, p.reshape(1, -1))[d, 0, :, 0] for e, p, d in zip(fs, point, ds)
                ]
                values2 = tensor_product(*evals)
            assert np.allclose(values1, values2)


@pytest.mark.parametrize("degree", range(1, 9))
def test_tensor_product_factorisation_quadrilateral(degree):
    P = degree
    cell_type = basix.CellType.quadrilateral
    element = basix.create_tp_element(
        basix.ElementFamily.P, cell_type, P, basix.LagrangeVariant.gll_warped
    )
    factors = element.get_tensor_product_representation()[0]
    # FIXME: This test assumes all factors formed by a single element
    element0 = factors[0]

    # Quadrature degree
    Q = 2 * P + 2
    points, w = basix.make_quadrature(cell_type, Q)

    data = element.tabulate(1, points)
    dphi_x = data[1, :, :, 0]
    dphi_y = data[2, :, :, 0]

    assert points.shape[0] == (P + 2) * (P + 2)

    cell1d = element0.cell_type
    points, _ = basix.make_quadrature(cell1d, Q)
    data = element0.tabulate(1, points)
    phi0 = data[0, :, :, 0]
    dphi0 = data[1, :, :, 0]

    # number of dofs in each direction
    Nd = P + 1
    # number of quadrature points in each direction
    Nq = P + 2

    # Compute derivative of basis function in the x direction
    dphi_tensor = np.zeros([Nq, Nq, Nd, Nd])
    for q0 in range(Nq):
        for q1 in range(Nq):
            for i0 in range(Nd):
                for i1 in range(Nd):
                    dphi_tensor[q0, q1, i0, i1] = dphi0[q0, i0] * phi0[q1, i1]
    dphi_tensor = dphi_tensor.reshape([Nq * Nq, Nd * Nd])
    assert np.allclose(dphi_x, dphi_tensor)

    # Compute derivative of basis function in the y direction
    dphi_tensor = np.zeros([Nq, Nq, Nd, Nd])
    for q0 in range(Nq):
        for q1 in range(Nq):
            for i0 in range(Nd):
                for i1 in range(Nd):
                    dphi_tensor[q0, q1, i0, i1] = phi0[q0, i0] * dphi0[q1, i1]

    dphi_tensor = dphi_tensor.reshape([Nq * Nq, Nd * Nd])
    assert np.allclose(dphi_y, dphi_tensor)


@pytest.mark.parametrize("degree", range(1, 6))
def test_tensor_product_factorisation_hexahedron(degree):
    P = degree
    element = basix.create_tp_element(
        basix.ElementFamily.P, basix.CellType.hexahedron, P, basix.LagrangeVariant.gll_warped
    )
    factors = element.get_tensor_product_representation()[0]

    # Quadrature degree
    Q = 2 * P + 2
    points, _ = basix.make_quadrature(basix.CellType.hexahedron, Q)

    # FIXME: This test assumes all factors formed by a single element
    element0 = factors[0]

    data = element.tabulate(1, points)
    dphi_x = data[1, :, :, 0]
    dphi_y = data[2, :, :, 0]
    dphi_z = data[3, :, :, 0]

    assert points.shape[0] == (P + 2) * (P + 2) * (P + 2)

    cell1d = element0.cell_type
    points, w = basix.make_quadrature(cell1d, Q)
    data = element0.tabulate(1, points)
    phi0 = data[0, :, :, 0]
    dphi0 = data[1, :, :, 0]

    # number of dofs in each direction
    Nd = P + 1
    # number of quadrature points in each direction
    Nq = P + 2

    # Compute derivative of basis function in the x direction
    dphi_tensor = np.zeros([Nq, Nq, Nq, Nd, Nd, Nd])
    for q0 in range(Nq):
        for q1 in range(Nq):
            for q2 in range(Nq):
                for i0 in range(Nd):
                    for i1 in range(Nd):
                        for i2 in range(Nd):
                            dphi_tensor[q0, q1, q2, i0, i1, i2] = (
                                dphi0[q0, i0] * phi0[q1, i1] * phi0[q2, i2]
                            )

    dphi_tensor = dphi_tensor.reshape([Nq * Nq * Nq, Nd * Nd * Nd])
    assert np.allclose(dphi_x, dphi_tensor)

    # Compute derivative of basis function in the y direction
    dphi_tensor = np.zeros([Nq, Nq, Nq, Nd, Nd, Nd])
    for q0 in range(Nq):
        for q1 in range(Nq):
            for q2 in range(Nq):
                for i0 in range(Nd):
                    for i1 in range(Nd):
                        for i2 in range(Nd):
                            dphi_tensor[q0, q1, q2, i0, i1, i2] = (
                                phi0[q0, i0] * dphi0[q1, i1] * phi0[q2, i2]
                            )

    dphi_tensor = dphi_tensor.reshape([Nq * Nq * Nq, Nd * Nd * Nd])
    assert np.allclose(dphi_y, dphi_tensor)

    # Compute the derivative of basis function in the z direction
    dphi_tensor = np.zeros([Nq, Nq, Nq, Nd, Nd, Nd])
    for q0 in range(Nq):
        for q1 in range(Nq):
            for q2 in range(Nq):
                for i0 in range(Nd):
                    for i1 in range(Nd):
                        for i2 in range(Nd):
                            dphi_tensor[q0, q1, q2, i0, i1, i2] = (
                                phi0[q0, i0] * phi0[q1, i1] * dphi0[q2, i2]
                            )

    dphi_tensor = dphi_tensor.reshape([Nq * Nq * Nq, Nd * Nd * Nd])
    assert np.allclose(dphi_z, dphi_tensor)


@pytest.mark.parametrize(
    "cell_type",
    [
        basix.CellType.quadrilateral,
        basix.CellType.hexahedron,
    ],
)
@pytest.mark.parametrize(
    "family, args",
    [
        (basix.ElementFamily.P, (basix.LagrangeVariant.equispaced,)),
        (basix.ElementFamily.P, (basix.LagrangeVariant.gll_warped,)),
    ],
)
@pytest.mark.parametrize("degree", range(1, 5))
def test_dof_ordering(cell_type, family, args, degree):
    e = basix.create_tp_element(family, cell_type, degree, *args)
    perm = e.dof_ordering
    e2 = basix.create_element(family, cell_type, degree, *args, dof_ordering=perm)
    assert e2.has_tensor_product_factorisation

    assert perm == basix.finite_element.tp_dof_ordering(family, cell_type, degree, *args)
