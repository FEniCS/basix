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
    return np.outer(a, b).reshape(-1)


@parametrize_over_elements(4)
def test_tensor_product_factorisation(cell_type, degree, element_type, element_args):
    element = basix.create_element(element_type, cell_type, degree, *element_args)
    tdim = len(basix.topology(cell_type)) - 1

    # These elements should have a factorisation
    if cell_type in [
        basix.CellType.quadrilateral, basix.CellType.hexahedron
    ] and element_type in [
        basix.ElementFamily.P
    ]:
        assert element.has_tensor_product_factorisation

    if not element.has_tensor_product_factorisation:
        with pytest.raises(RuntimeError):
            element.get_tensor_product_representation()
        pytest.skip()

    factors = element.get_tensor_product_representation()

    lattice = basix.create_lattice(cell_type, 1, basix.LatticeType.equispaced, True)
    tab1 = element.tabulate(1, lattice)

    for i, point in enumerate(lattice):
        for ds in product(range(2), repeat=tdim):
            if sum(ds) > 1:
                continue
            deriv = basix.index(*ds)
            values1 = tab1[deriv, i, :, :]

            values2 = np.empty((element.dim, 1))
            for fs, perm in factors:
                evals = [e.tabulate(d, [p])[d, 0, :, 0] for e, p, d in zip(fs, point, ds)]
                tab2 = tensor_product(*evals)
                for a, b in enumerate(perm):
                    if b != -1:
                        values2[b, 0] = tab2[a]
            assert np.allclose(values1, values2)
