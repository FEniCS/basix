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
        basix.LagrangeVariant.bernstein,
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
    print(result1)
