# Copyright (c) 2022 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import pytest

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
    basix.ElementFamily.custom,
]

variants = [
    [basix.LagrangeVariant.gll_isaac],
    [basix.LagrangeVariant.gll_warped],
    [basix.LagrangeVariant.legendre],
    [basix.LagrangeVariant.bernstein],
    [basix.DPCVariant.diagonal_gll],
    [basix.DPCVariant.legendre],
    [basix.LagrangeVariant.legendre, basix.DPCVariant.legendre],
]


def test_all_cells_included():
    all_cells = set()
    for c in dir(basix.CellType):
        if not c.startswith("_") and c not in ["name", "value"]:
            all_cells.add(getattr(basix.CellType, c))

    assert all_cells == set(cells)


def test_all_elements_included():
    all_elements = set()
    for c in dir(basix.ElementFamily):
        if not c.startswith("_") and c not in ["name", "value"]:
            all_elements.add(getattr(basix.ElementFamily, c))

    assert all_elements == set(elements)


@pytest.mark.parametrize("cell", cells)
@pytest.mark.parametrize("degree", range(-1, 5))
@pytest.mark.parametrize("family", elements)
@pytest.mark.parametrize("variant", variants)
def test_create_element(cell, degree, family, variant):
    """Check that either the element is created or a RuntimeError is thrown."""
    try:
        element = basix.create_element(family, cell, degree, *variant)
        assert element.degree == degree
    except RuntimeError as e:
        # Don't allow cryptic "dgesv failed" messages
        if len(e.args) == 0 or "dgesv" in e.args[0]:
            raise e

    try:
        element = basix.create_element(family, cell, degree, *variant, True)
        assert element.degree == degree
    except RuntimeError as e:
        if len(e.args) == 0 or "dgesv" in e.args[0]:
            raise e
