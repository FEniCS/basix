# Copyright (c) 2024 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

import basix

np.set_printoptions(suppress=True)


@pytest.mark.parametrize(
    "lagrange_variant",
    [
        basix.LagrangeVariant.gll_warped,
        basix.LagrangeVariant.legendre,
        basix.LagrangeVariant.chebyshev_warped,
    ],
)
def test_ordering(lagrange_variant):
    pt = np.array([[1 / 3, 1 / 3], [0.3, 0.2]])

    is_dc = lagrange_variant in [
        basix.LagrangeVariant.legendre,
        basix.LagrangeVariant.chebyshev_warped,
    ]

    # reordered element
    el = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.triangle,
        2,
        lagrange_variant,
        basix.DPCVariant.unset,
        is_dc,
        dof_ordering=[0, 3, 5, 1, 2, 4],
    )
    order = el.dof_ordering
    result1 = el.tabulate(1, pt)

    # standard element
    el = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.triangle,
        2,
        lagrange_variant,
        basix.DPCVariant.unset,
        is_dc,
    )
    result2 = el.tabulate(1, pt)
    presult = np.zeros_like(result2)

    # permute standard data
    presult[:, :, order, :] = result2

    assert np.allclose(result1, presult)


@pytest.mark.parametrize(
    "cell_type, degree, dof_ordering",
    [
        (basix.CellType.interval, 1, []),
        (basix.CellType.interval, 1, [0, 1]),
        (basix.CellType.interval, 1, [1, 0]),
        (basix.CellType.interval, 2, [1, 2, 0]),
        (basix.CellType.triangle, 3, []),
        (basix.CellType.triangle, 3, list(range(10))),
        (basix.CellType.triangle, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_legendre_interpolation(cell_type, degree, dof_ordering, dtype):
    el = basix.create_element(
        basix.ElementFamily.P,
        cell_type,
        degree,
        basix.LagrangeVariant.legendre,
        discontinuous=True,
        dof_ordering=dof_ordering,
        dtype=dtype,
    )
    samples = np.column_stack((np.ones(len(el.points)), el.points[:, 0]))
    coefficients = el.interpolation_matrix @ samples

    # Exact Legendre moments of 1 and x on the reference cell.
    expected = np.zeros((el.dim, 2))
    if cell_type == basix.CellType.interval:
        expected[0] = [1, 0.5]
        expected[1, 1] = np.sqrt(3) / 6
        points = np.array([[0.25], [0.75]], dtype=dtype)
    else:
        # Degree three uses 12 interpolation points for 10 DOFs.
        assert len(el.points) != el.dim
        expected[0] = [np.sqrt(2) / 2, np.sqrt(2) / 6]
        expected[1, 1] = -1 / 12
        expected[2, 1] = np.sqrt(3) / 12
        points = np.array([[0.25, 0.25], [0.6, 0.2]], dtype=dtype)
    if dof_ordering:
        expected[dof_ordering] = expected.copy()

    tol = 1e-6 if dtype == np.float32 else 1e-13
    assert np.allclose(coefficients, expected, atol=tol, rtol=0)

    values = el.tabulate(0, points)[0, :, :, 0] @ coefficients
    assert np.allclose(
        values, np.column_stack((np.ones(len(points)), points[:, 0])), atol=tol, rtol=0
    )
    basis = el.tabulate(0, el.points)[0, :, :, 0]
    assert np.allclose(el.interpolation_matrix @ basis, np.eye(el.dim), atol=tol, rtol=0)
