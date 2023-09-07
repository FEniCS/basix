# Copyright (c) 2023 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

import basix

cells = [
    basix.CellType.point,
    basix.CellType.interval,
    basix.CellType.triangle,
    basix.CellType.quadrilateral,
    basix.CellType.tetrahedron,
    basix.CellType.hexahedron,
    basix.CellType.prism,
    basix.CellType.pyramid,
]

elements = [
    basix.ElementFamily.P,
    basix.ElementFamily.RT,
    basix.ElementFamily.BDM,
    basix.ElementFamily.N1E,
    basix.ElementFamily.N2E,
    basix.ElementFamily.Regge,
    basix.ElementFamily.HHJ,
    basix.ElementFamily.bubble,
    basix.ElementFamily.serendipity,
    basix.ElementFamily.DPC,
    basix.ElementFamily.CR,
    basix.ElementFamily.Hermite,
    basix.ElementFamily.iso,
    basix.ElementFamily.custom,
]

variants = [
    [basix.LagrangeVariant.gll_isaac],
    [basix.LagrangeVariant.gll_warped],
    [basix.LagrangeVariant.legendre],
    [basix.LagrangeVariant.bernstein],
    [basix.LagrangeVariant.unset, basix.DPCVariant.diagonal_gll],
    [basix.LagrangeVariant.unset, basix.DPCVariant.legendre],
    [basix.LagrangeVariant.legendre, basix.DPCVariant.legendre],
]


@pytest.mark.parametrize("cell", cells)
@pytest.mark.parametrize("degree", range(-1, 5))
@pytest.mark.parametrize("family", elements)
@pytest.mark.parametrize("variant", variants)
def test_create_element(cell, degree, family, variant):
    """Check that either the element is created or a RuntimeError is thrown."""
    try:
        element = basix.create_element(family, cell, degree, *variant)
    except RuntimeError:
        pytest.xfail("Element not supported")

    wcoeffs = element.wcoeffs
    for i, rowi in enumerate(wcoeffs):
        for j, rowj in enumerate(wcoeffs):
            assert np.isclose(np.dot(rowi, rowj), int(i == j))
