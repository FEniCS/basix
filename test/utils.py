# Copyright (c) 2020 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import pytest
import basix


def parametrize_over_elements(degree, reference=None):
    elementlist = []

    elementlist += [(c, basix.ElementFamily.P, o, [basix.LatticeType.gll_isaac])
                    for c in [basix.CellType.interval, basix.CellType.triangle,
                              basix.CellType.tetrahedron,
                              basix.CellType.quadrilateral, basix.CellType.hexahedron,
                              basix.CellType.prism]
                    for o in range(1, degree + 1)]
    elementlist += [(c, basix.ElementFamily.P, o, [basix.LatticeType.gll_warped])
                    for c in [basix.CellType.interval, basix.CellType.triangle,
                              basix.CellType.tetrahedron,
                              basix.CellType.quadrilateral, basix.CellType.hexahedron,
                              basix.CellType.prism, basix.CellType.pyramid]
                    for o in range(1, degree + 1)]
    elementlist += [(c, basix.ElementFamily.P, o, [basix.LatticeType.gll_isaac, True])
                    for c in [basix.CellType.interval, basix.CellType.triangle,
                              basix.CellType.tetrahedron,
                              basix.CellType.quadrilateral, basix.CellType.hexahedron]
                    for o in range(0, degree + 1)]
    elementlist += [(c, basix.ElementFamily.P, o, [basix.LatticeType.equispaced])
                    for c in [basix.CellType.interval, basix.CellType.triangle,
                              basix.CellType.tetrahedron,
                              basix.CellType.quadrilateral, basix.CellType.hexahedron,
                              basix.CellType.prism, basix.CellType.pyramid]
                    for o in range(1, min(4, degree + 1))]
    elementlist += [(c, basix.ElementFamily.N1E, o, [])
                    for c in [basix.CellType.triangle, basix.CellType.tetrahedron,
                              basix.CellType.quadrilateral, basix.CellType.hexahedron]
                    for o in range(1, degree + 1)]
    elementlist += [(c, basix.ElementFamily.RT, o, [])
                    for c in [basix.CellType.triangle, basix.CellType.tetrahedron,
                              basix.CellType.quadrilateral, basix.CellType.hexahedron]
                    for o in range(1, degree + 1)]
    elementlist += [(c, basix.ElementFamily.N2E, o, [])
                    for c in [basix.CellType.triangle, basix.CellType.tetrahedron,
                              basix.CellType.quadrilateral, basix.CellType.hexahedron]
                    for o in range(1, degree + 1)]
    elementlist += [(c, basix.ElementFamily.BDM, o, [])
                    for c in [basix.CellType.triangle, basix.CellType.tetrahedron,
                              basix.CellType.quadrilateral, basix.CellType.hexahedron]
                    for o in range(1, degree + 1)]
    elementlist += [(c, basix.ElementFamily.CR, o, [])
                    for c in [basix.CellType.triangle, basix.CellType.tetrahedron]
                    for o in range(1, min(2, degree + 1))]
    elementlist += [(c, basix.ElementFamily.Regge, o, [])
                    for c in [basix.CellType.triangle, basix.CellType.tetrahedron]
                    for o in range(1, degree + 1)]
    elementlist += [(basix.CellType.interval, basix.ElementFamily.Bubble, o, [])
                    for o in range(2, degree + 1)]
    elementlist += [(basix.CellType.triangle, basix.ElementFamily.Bubble, o, [])
                    for o in range(3, degree + 1)]
    elementlist += [(basix.CellType.tetrahedron, basix.ElementFamily.Bubble, o, [])
                    for o in range(4, degree + 1)]
    elementlist += [(basix.CellType.quadrilateral, basix.ElementFamily.Bubble, o, [])
                    for o in range(2, degree + 1)]
    elementlist += [(basix.CellType.hexahedron, basix.ElementFamily.Bubble, o, [])
                    for o in range(2, degree + 1)]
    elementlist += [(c, basix.ElementFamily.Serendipity, o, [])
                    for c in [basix.CellType.interval, basix.CellType.quadrilateral,
                              basix.CellType.hexahedron]
                    for o in range(1, degree + 1)]

    if reference is None:
        return pytest.mark.parametrize("cell_type, element_type, degree, element_args", elementlist)
    else:
        return pytest.mark.parametrize("element_type, degree, element_args",
                                       [(b, c, d) for a, b, c, d in elementlist if a == reference])
