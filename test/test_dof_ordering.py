# Copyright (c) 2024 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import numpy as np

np.set_printoptions(suppress=True)


def test_ordering():
    pt = np.array([[1 / 3, 1 / 3], [0.3, 0.2]])

    # reordered element
    el = basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, 2, dof_ordering=[0, 3, 5, 1, 2, 4]
    )
    order = el.dof_ordering
    result1 = el.tabulate(1, pt)

    # standard element
    el = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 2)
    result2 = el.tabulate(1, pt)
    presult = np.zeros_like(result2)

    # permute standard data
    presult[:, :, order, :] = result2

    assert np.allclose(result1, presult)
    print(result1)
