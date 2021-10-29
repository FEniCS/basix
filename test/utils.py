# Copyright (c) 2020 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import pytest
import basix


def parametrize_over_elements(degree, reference=None):
    elementlist = []

    elementlist += [(c, basix.ElementFamily.p, o, [basix.LagrangeVariant.gll_isaac])
                    for c in [basix.CellType.interval, basix.CellType.triangle,
                              basix.CellType.tetrahedron,
                              basix.CellType.quadrilateral, basix.CellType.hexahedron,
                              basix.CellType.prism]
                    for o in range(1, degree + 1)]
    elementlist += [(c, basix.ElementFamily.p, o, [basix.LagrangeVariant.gll_warped])
                    for c in [basix.CellType.interval, basix.CellType.triangle,
                              basix.CellType.tetrahedron,
                              basix.CellType.quadrilateral, basix.CellType.hexahedron,
                              basix.CellType.prism]
                    for o in range(1, degree + 1)]
    elementlist += [(c, basix.ElementFamily.p, o, [basix.LagrangeVariant.equispaced])
                    for c in [basix.CellType.interval, basix.CellType.triangle,
                              basix.CellType.tetrahedron,
                              basix.CellType.quadrilateral, basix.CellType.hexahedron,
                              basix.CellType.prism, basix.CellType.pyramid]
                    for o in range(1, min(4, degree + 1))]
    elementlist += [(c, basix.ElementFamily.n1e, o, [])
                    for c in [basix.CellType.triangle, basix.CellType.tetrahedron,
                              basix.CellType.quadrilateral, basix.CellType.hexahedron]
                    for o in range(1, degree + 1)]
    elementlist += [(c, basix.ElementFamily.rt, o, [])
                    for c in [basix.CellType.triangle, basix.CellType.tetrahedron,
                              basix.CellType.quadrilateral, basix.CellType.hexahedron]
                    for o in range(1, degree + 1)]
    elementlist += [(c, basix.ElementFamily.n2e, o, [])
                    for c in [basix.CellType.triangle, basix.CellType.tetrahedron,
                              basix.CellType.quadrilateral, basix.CellType.hexahedron]
                    for o in range(1, degree + 1)]
    elementlist += [(c, basix.ElementFamily.bdm, o, [])
                    for c in [basix.CellType.triangle, basix.CellType.tetrahedron,
                              basix.CellType.quadrilateral, basix.CellType.hexahedron]
                    for o in range(1, degree + 1)]
    elementlist += [(c, basix.ElementFamily.cr, o, [])
                    for c in [basix.CellType.triangle, basix.CellType.tetrahedron]
                    for o in range(1, min(2, degree + 1))]
    elementlist += [(c, basix.ElementFamily.regge, o, [])
                    for c in [basix.CellType.triangle, basix.CellType.tetrahedron]
                    for o in range(1, degree + 1)]
    elementlist += [(basix.CellType.interval, basix.ElementFamily.bubble, o, [])
                    for o in range(2, degree + 1)]
    elementlist += [(basix.CellType.triangle, basix.ElementFamily.bubble, o, [])
                    for o in range(3, degree + 1)]
    elementlist += [(basix.CellType.tetrahedron, basix.ElementFamily.bubble, o, [])
                    for o in range(4, degree + 1)]
    elementlist += [(basix.CellType.quadrilateral, basix.ElementFamily.bubble, o, [])
                    for o in range(2, degree + 1)]
    elementlist += [(basix.CellType.hexahedron, basix.ElementFamily.bubble, o, [])
                    for o in range(2, degree + 1)]
    elementlist += [(c, basix.ElementFamily.serendipity, o, [])
                    for c in [basix.CellType.interval, basix.CellType.quadrilateral,
                              basix.CellType.hexahedron]
                    for o in range(1, degree + 1)]

    if reference is None:
        return pytest.mark.parametrize("cell_type, element_type, degree, element_args", elementlist)
    else:
        return pytest.mark.parametrize("element_type, degree, element_args",
                                       [(b, c, d) for a, b, c, d in elementlist if a == reference])
