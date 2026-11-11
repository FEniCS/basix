# Copyright (c) 2020 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

import basix

from .utils import cached_create_element, parametrize_over_elements


@pytest.mark.parametrize("n", range(1, 6))
@pytest.mark.parametrize(
    "cell_type", [basix.CellType.interval, basix.CellType.triangle, basix.CellType.tetrahedron]
)
@pytest.mark.parametrize("element_type", [basix.ElementFamily.P])
def test_interpolation(cell_type, n, element_type):
    element = cached_create_element(element_type, cell_type, n, (basix.LagrangeVariant.gll_warped,))
    assert element.interpolation_matrix.shape[0] == element.dim
    assert element.interpolation_matrix.shape[1] == element.points.shape[0]
    assert element.points.shape[1] == len(basix.topology(element.cell_type)) - 1


@parametrize_over_elements(5)
def test_interpolation_matrix(cell_type, degree, element_type, element_args):
    if degree > 4:
        if cell_type in [
            basix.CellType.quadrilateral,
            basix.CellType.hexahedron,
        ] and element_type in [
            basix.ElementFamily.RT,
            basix.ElementFamily.N1E,
            basix.ElementFamily.BDM,
            basix.ElementFamily.N2E,
        ]:
            pytest.xfail(
                "High degree Hdiv and Hcurl spaces on hexes based on "
                "Lagrange spaces with equally spaced points are unstable."
            )

    element = cached_create_element(element_type, cell_type, degree, tuple(element_args))

    i_m = element.interpolation_matrix
    tabulated = element.tabulate(element.interpolation_nderivs, element.points)
    weights = i_m.reshape(element.dim, element.value_size, len(element.points), tabulated.shape[0])
    assert np.shares_memory(weights, i_m)

    # Applying the DOF functionals to the basis functions gives the identity.
    coeffs = np.einsum("icpd,dpjc->ij", weights, tabulated)
    assert np.allclose(coeffs, np.identity(coeffs.shape[0]))


@pytest.mark.parametrize(
    "family",
    [
        basix.ElementFamily.P,
        basix.ElementFamily.RT,
        basix.ElementFamily.N1E,
        basix.ElementFamily.Hermite,
    ],
)
def test_entity_interpolation_tensor(family):
    variant = (
        basix.LagrangeVariant.unset
        if family == basix.ElementFamily.Hermite
        else basix.LagrangeVariant.equispaced
    )
    element = basix.create_element(family, basix.CellType.triangle, 3, variant)
    identity = np.eye(element.dim)
    for d, entities in enumerate(element.M):
        for e, weights in enumerate(entities):
            if not element.entity_dofs[d][e]:
                continue
            tabulated = element.tabulate(element.interpolation_nderivs, element.x[d][e])
            coeffs = np.einsum("icpd,dpjc->ij", weights, tabulated)
            assert np.allclose(coeffs, identity[element.entity_dofs[d][e]], atol=1e-12)


@parametrize_over_elements(4)
def test_interpolation_is_identity(cell_type, degree, element_type, element_args):
    element = cached_create_element(element_type, cell_type, degree, tuple(element_args))
    i_m = element.interpolation_matrix
    if i_m.shape[0] == i_m.shape[1]:
        assert element.interpolation_is_identity == np.allclose(i_m, np.eye(i_m.shape[0]))
    else:
        assert not element.interpolation_is_identity
