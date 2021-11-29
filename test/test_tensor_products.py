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
    assert a.shape[0] == b.shape[0] == 1  # TODO: implement for derivatives too
    assert a.shape[3] == b.shape[3] == 1  # TODO: implement for non-scalar elements
    out = np.empty((1, a.shape[1] * b.shape[1], a.shape[2] * b.shape[2], 1))
    for n, (i, j) in enumerate(product(range(a.shape[1]), range(b.shape[1]))):
        out[0, n, :, 0] = np.outer(a[0, i, :, 0], b[0, j, :, 0]).reshape(-1)
    return out


@parametrize_over_elements(4)
def test_tensor_product_factorisation(cell_type, degree, element_type, element_args):
    element = basix.create_element(element_type, cell_type, degree, *element_args)
    tdim = len(basix.topology(cell_type)) - 1

    if not element.has_tensor_product_factorisation:
        with pytest.raises(RuntimeError):
            element.get_tensor_product_representation()
        pytest.skip()

    factors = element.get_tensor_product_representation()

    lattice_1d = basix.create_lattice(basix.CellType.interval, 1, basix.LatticeType.equispaced, True)
    lattice_nd = np.array(list(product(*[lattice_1d for i in range(tdim)])))

    tab1 = element.tabulate(0, lattice_nd)

    tab2 = np.empty((1, lattice_nd.shape[0], element.dim, 1))

    for fs, perm in factors:
        evals = [e.tabulate(0, lattice_1d) for e in fs]
        tab = tensor_product(*evals)
        for i, j in enumerate(perm):
            if j != -1:
                tab2[:, :, j, :] = tab[:, :, i, :]

    assert np.allclose(tab1, tab2)
