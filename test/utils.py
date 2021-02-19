# Copyright (c) 2020 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import pytest


def parametrize_over_elements(order, reference=None):
    elementlist = []

    elementlist += [(c, "Lagrange", o)
                    for c in ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"]
                    for o in range(1, order + 1)]
    elementlist += [(c, "Discontinuous Lagrange", o)
                    for c in ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"]
                    for o in range(0, order + 1)]
    elementlist += [(c, "Nedelec 1st kind H(curl)", o)
                    for c in ["triangle", "tetrahedron", "quadrilateral", "hexahedron"]
                    for o in range(1, order + 1)]
    elementlist += [(c, "Raviart-Thomas", o)
                    for c in ["triangle", "tetrahedron", "quadrilateral", "hexahedron"]
                    for o in range(1, order + 1)]
    elementlist += [(c, "Nedelec 2nd kind H(curl)", o)
                    for c in ["triangle", "tetrahedron"]
                    for o in range(1, order + 1)]
    elementlist += [(c, "Brezzi-Douglas-Marini", o)
                    for c in ["triangle", "tetrahedron"]
                    for o in range(1, order + 1)]
    elementlist += [(c, "Crouzeix-Raviart", o)
                    for c in ["triangle", "tetrahedron"]
                    for o in range(1, min(2, order + 1))]
    elementlist += [(c, "Regge", o)
                    for c in ["triangle", "tetrahedron"]
                    for o in range(1, order + 1)]

    if reference is None:
        return pytest.mark.parametrize("cell_name, element_name, order", elementlist)
    else:
        return pytest.mark.parametrize("element_name, order", [(j, k) for i, j, k in elementlist if i == reference])
