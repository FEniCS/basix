# Copyright (c) 2023 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

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


@parametrize_over_elements(5)
def test_orthonormal(cell_type, degree, element_type, element_args):
    if (
        element_type == basix.ElementFamily.iso
        and cell_type == basix.CellType.hexahedron
        and degree > 3
    ):
        pytest.skip()  # Skip slow test

    element = basix.create_element(element_type, cell_type, degree, *element_args)

    wcoeffs = element.wcoeffs
    for i, rowi in enumerate(wcoeffs):
        for j, rowj in enumerate(wcoeffs):
            assert np.isclose(np.dot(rowi, rowj), int(i == j))
