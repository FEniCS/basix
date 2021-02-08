# Copyright (c) 2020 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import numpy as np
import basix
import pytest


@pytest.mark.parametrize("n", range(1, 6))
@pytest.mark.parametrize("cellname", ["interval", "triangle", "tetrahedron"])
@pytest.mark.parametrize("element_name", ["Lagrange"])
def test_interpolation(cellname, n, element_name):
    element = basix.create_element(element_name, cellname, n)
    assert element.interpolation_matrix.shape[0] == element.dim
    assert element.interpolation_matrix.shape[1] == element.points.shape[0]
    assert element.points.shape[1] == len(basix.topology(element.cell_type)) - 1


@pytest.mark.parametrize("order", range(1, 6))
@pytest.mark.parametrize("cellname, element_name", [
    ("interval", "Lagrange"), ("triangle", "Lagrange"), ("tetrahedron", "Lagrange"),
    ("quadrilateral", "Lagrange"), ("hexahedron", "Lagrange"),
    ("triangle", "Nedelec 1st kind H(curl)"), ("tetrahedron", "Nedelec 1st kind H(curl)"),
    ("quadrilateral", "Nedelec 1st kind H(curl)"), ("hexahedron", "Nedelec 1st kind H(curl)"),
])
def test_interpolation_matrix(cellname, order, element_name):
    if order > 4:
        if cellname in ["quadrilateral", "hexahedron"] and element_name in [
            "Raviart-Thomas", "Nedelec 1st kind H(curl)"
        ]:
            pytest.xfail("High order Hdiv and Hcurl spaces on hexes based on "
                         "Lagrange spaces with equally spaced points are unstable.")

    element = basix.create_element(element_name, cellname, order)

    i_m = element.interpolation_matrix
    tabulated = element.tabulate(0, element.points)[0]

    coeffs = np.zeros((i_m.shape[0], i_m.shape[0]))
    for i in range(i_m.shape[0]):
        coeffs[i, :] = i_m @ tabulated[:, i::i_m.shape[0]].T.reshape(i_m.shape[1])

    assert np.allclose(coeffs, np.identity(coeffs.shape[0]))
