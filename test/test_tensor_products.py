# Copyright (c) 2021 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import numpy as np
import basix
import pytest
from itertools import product
from .utils import parametrize_over_elements


def tensor_product(*data):
    if len(data) == 1:
        return data[0]
    if len(data) > 2:
        return tensor_product(tensor_product(data[0], data[1]), *data[2:])

    a, b = data
    assert a.shape[1] == b.shape[1] == 1  # TODO: implement for non-scalar elements
    return np.outer(a[:, 0], b[:, 0]).reshape(-1)


@parametrize_over_elements(4)
def test_tensor_product_factorisation(cell_type, degree, element_type, element_args):
    element = basix.create_element(element_type, cell_type, degree, *element_args)
    tdim = len(basix.topology(cell_type)) - 1

    if not element.has_tensor_product_factorisation:
        with pytest.raises(RuntimeError):
            element.get_tensor_product_representation()
        pytest.skip()

    factors = element.get_tensor_product_representation()

    lattice = basix.create_lattice(cell_type, 1, basix.LatticeType.equispaced, True)

    tab1 = element.tabulate(0, lattice)

    for i, point in enumerate(lattice):
        values1 = tab1[0, i, :, :]

        values2 = np.empty((element.dim, 1))

        for fs, perm in factors:
            evals = [e.tabulate(0, [p]) for e, p in zip(fs, point)]
            tab2 = tensor_product(*evals)
            for i, j in enumerate(perm):
                if j != -1:
                    values2[j, 0] = tab2[i]

        assert np.allclose(values1, values2)
