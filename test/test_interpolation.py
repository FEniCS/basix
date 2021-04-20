# Copyright (c) 2020 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import numpy as np
import basix
import pytest
from .utils import parametrize_over_elements


@pytest.mark.parametrize("n", range(1, 6))
@pytest.mark.parametrize("cell_name", ["interval", "triangle", "tetrahedron"])
@pytest.mark.parametrize("element_name", ["Lagrange"])
def test_interpolation(cell_name, n, element_name):
    element = basix.create_element(element_name, cell_name, n)
    assert element.interpolation_matrix.shape[0] == element.dim
    assert element.interpolation_matrix.shape[1] == element.points.shape[0]
    assert element.points.shape[1] == len(basix.topology(element.cell_type)) - 1


@parametrize_over_elements(5)
def test_interpolation_matrix(cell_name, order, element_name):
    element = basix.create_element(element_name, cell_name, order)
    i_m = element.interpolation_matrix
    tabulated = element.tabulate(0, element.points)[0]

    # Loop over dofs
    coeffs = np.zeros((i_m.shape[0], i_m.shape[0]))
    for i in range(i_m.shape[0]):
        coeffs[i, :] = i_m @ tabulated[:, i::i_m.shape[0]].T.reshape(i_m.shape[1])

    assert np.allclose(coeffs, np.identity(coeffs.shape[0]), atol=1e-6)
