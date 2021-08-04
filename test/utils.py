# Copyright (c) 2020 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import pytest


def parametrize_over_elements(degree, reference=None):
    elementlist = []

    elementlist += [(c, "Lagrange", o, {"lattice_type": "gll_warped"})
                    for c in ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron",
                              "prism", "pyramid"]
                    for o in range(1, degree + 1)]
    elementlist += [(c, "Discontinuous Lagrange", o, {})
                    for c in ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"]
                    for o in range(0, degree + 1)]
    elementlist += [(c, "Nedelec 1st kind H(curl)", o, {})
                    for c in ["triangle", "tetrahedron", "quadrilateral", "hexahedron"]
                    for o in range(1, degree + 1)]
    elementlist += [(c, "Raviart-Thomas", o, {})
                    for c in ["triangle", "tetrahedron", "quadrilateral", "hexahedron"]
                    for o in range(1, degree + 1)]
    elementlist += [(c, "Nedelec 2nd kind H(curl)", o, {})
                    for c in ["triangle", "tetrahedron", "quadrilateral", "hexahedron"]
                    for o in range(1, degree + 1)]
    elementlist += [(c, "Brezzi-Douglas-Marini", o, {})
                    for c in ["triangle", "tetrahedron", "quadrilateral", "hexahedron"]
                    for o in range(1, degree + 1)]
    elementlist += [(c, "Crouzeix-Raviart", o, {})
                    for c in ["triangle", "tetrahedron"]
                    for o in range(1, min(2, degree + 1))]
    elementlist += [(c, "Regge", o, {})
                    for c in ["triangle", "tetrahedron"]
                    for o in range(1, degree + 1)]
    elementlist += [("interval", "Bubble", o, {}) for o in range(2, degree + 1)]
    elementlist += [("triangle", "Bubble", o, {}) for o in range(3, degree + 1)]
    elementlist += [("tetrahedron", "Bubble", o, {}) for o in range(4, degree + 1)]
    elementlist += [("quadrilateral", "Bubble", o, {}) for o in range(2, degree + 1)]
    elementlist += [("hexahedron", "Bubble", o, {}) for o in range(2, degree + 1)]
    elementlist += [(c, "Serendipity", o, {})
                    for c in ["interval", "quadrilateral", "hexahedron"]
                    for o in range(1, degree + 1)]

    if reference is None:
        return pytest.mark.parametrize("cell_name, element_name, degree, element_kwargs", elementlist)
    else:
        return pytest.mark.parametrize("element_name, degree, element_kwargs",
                                       [(b, c, d) for a, b, c, d in elementlist if a == reference])
