# Copyright (c) 2020 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT
"""Test utilities."""

import pytest
from basix import ElementFamily, CellType, LagrangeVariant, DPCVariant


def parametrize_over_elements(degree, reference=None, discontinuous=False):
    """Parametrize a test over a wide range of elements."""
    elementlist = []

    for k in range(1, degree + 1):
        # Elements on all cells
        for c in [CellType.interval, CellType.triangle, CellType.tetrahedron, CellType.quadrilateral,
                  CellType.hexahedron, CellType.prism, CellType.pyramid]:
            if k < 4:
                elementlist.append((c, ElementFamily.P, k, [LagrangeVariant.equispaced]))

        # Elements on all cells except pyramid
        for c in [CellType.interval, CellType.triangle, CellType.tetrahedron, CellType.quadrilateral,
                  CellType.hexahedron, CellType.prism]:
            elementlist.append((c, ElementFamily.P, k, [LagrangeVariant.gll_isaac]))
            elementlist.append((c, ElementFamily.P, k, [LagrangeVariant.gll_warped]))

        # Elements on all cells except prism and pyramid
        for c in [CellType.interval, CellType.triangle, CellType.tetrahedron, CellType.quadrilateral,
                  CellType.hexahedron]:
            if discontinuous:
                elementlist.append((c, ElementFamily.P, k, [LagrangeVariant.legendre]))

        # Elements on all cells except prism, pyramid and interval
        for c in [CellType.triangle, CellType.tetrahedron, CellType.quadrilateral, CellType.hexahedron]:
            elementlist.append((c, ElementFamily.N1E, k, [LagrangeVariant.legendre]))
            elementlist.append((c, ElementFamily.N2E, k, [LagrangeVariant.legendre, DPCVariant.legendre]))
            elementlist.append((c, ElementFamily.RT, k, [LagrangeVariant.legendre]))
            elementlist.append((c, ElementFamily.BDM, k, [LagrangeVariant.legendre, DPCVariant.legendre]))

        # Elements on simplex cells
        for c in [CellType.triangle, CellType.tetrahedron]:
            if k == 1:
                elementlist.append((c, ElementFamily.CR, k, []))
            elementlist.append((c, ElementFamily.Regge, k, []))

        # Elements on tensor product cells
        for c in [CellType.interval, CellType.quadrilateral, CellType.hexahedron]:
            for lv in [LagrangeVariant.equispaced, LagrangeVariant.gll_warped]:
                for dv in [DPCVariant.simplex_equispaced, DPCVariant.diagonal_gll]:
                    elementlist.append((c, ElementFamily.serendipity, k, [lv, dv]))

        # Elements on quads and hexes
        for c in [CellType.quadrilateral, CellType.hexahedron]:
            if discontinuous:
                for v in [DPCVariant.simplex_equispaced, DPCVariant.simplex_gll,
                          DPCVariant.horizontal_equispaced, DPCVariant.horizontal_gll,
                          DPCVariant.diagonal_equispaced, DPCVariant.diagonal_gll]:
                    elementlist.append((c, ElementFamily.dpc, k, [v]))

        # Bubble elements
        if k >= 2:
            elementlist.append((CellType.interval, ElementFamily.bubble, k, []))
            elementlist.append((CellType.quadrilateral, ElementFamily.bubble, k, []))
            elementlist.append((CellType.hexahedron, ElementFamily.bubble, k, []))
        if k >= 3:
            elementlist.append((CellType.triangle, ElementFamily.bubble, k, []))
        if k >= 4:
            elementlist.append((CellType.tetrahedron, ElementFamily.bubble, k, []))

    if reference is None:
        if len(elementlist) == 0:
            raise ValueError(f"No elements will be tested with reference: {reference}")
        return pytest.mark.parametrize("cell_type, element_type, degree, element_args", elementlist)
    else:
        elementlist = [(b, c, d) for a, b, c, d in elementlist if a == reference]
        if len(elementlist) == 0:
            raise ValueError(f"No elements will be tested with reference: {reference}")
        return pytest.mark.parametrize("element_type, degree, element_args", elementlist)
