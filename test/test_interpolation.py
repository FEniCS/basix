# Copyright (c) 2020 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import numpy as np
import basix
import pytest
from .utils import parametrize_over_elements


@pytest.mark.parametrize("n", range(1, 6))
@pytest.mark.parametrize("cell_type", [basix.CellType.interval, basix.CellType.triangle, basix.CellType.tetrahedron])
@pytest.mark.parametrize("element_type", [basix.ElementFamily.P])
def test_interpolation(cell_type, n, element_type):
    element = basix.create_element(element_type, cell_type, n, basix.LagrangeVariant.gll_warped)
    assert element.interpolation_matrix.shape[0] == element.dim
    assert element.interpolation_matrix.shape[1] == element.points.shape[0]
    assert element.points.shape[1] == len(basix.topology(element.cell_type)) - 1


@parametrize_over_elements(5)
def test_interpolation_matrix(cell_type, degree, element_type, element_args):
    if degree > 4:
        if cell_type in [
            basix.CellType.quadrilateral, basix.CellType.hexahedron
        ] and element_type in [
            basix.ElementFamily.RT, basix.ElementFamily.N1E, basix.ElementFamily.BDM,
            basix.ElementFamily.N2E
        ]:
            pytest.xfail("High degree Hdiv and Hcurl spaces on hexes based on "
                         "Lagrange spaces with equally spaced points are unstable.")

    element = basix.create_element(element_type, cell_type, degree, *element_args)
    i_m = element.interpolation_matrix
    tabulated = element.tabulate(0, element.points)[0]

    # Loop over dofs
    coeffs = np.zeros((i_m.shape[0], i_m.shape[0]))
    for i in range(i_m.shape[0]):
        coeffs[i, :] = i_m @ tabulated[:, i::i_m.shape[0]].T.reshape(i_m.shape[1])

    assert np.allclose(coeffs, np.identity(coeffs.shape[0]))


@parametrize_over_elements(4)
def test_interpolation_is_identity(cell_type, degree, element_type, element_args):
    element = basix.create_element(element_type, cell_type, degree, *element_args)
    i_m = element.interpolation_matrix

    if i_m.shape[0] == i_m.shape[1]:
        assert element.interpolation_is_identity == np.allclose(i_m, np.eye(i_m.shape[0]))
    else:
        assert not element.interpolation_is_identity
