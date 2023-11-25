# Copyright (c) 2020 Chris Richardson & Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import numpy
import pytest
import sympy


def sympy_nedelec(celltype, n):
    # These basis functions were computed using symfem. They can be recomputed
    # by running (eg):
    #    import symfem
    #    e = symfem.create_element("triangle", "N1curl", 2)
    #    print(e.get_basis_functions())
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")

    if celltype == basix.CellType.triangle:
        if n == 1:
            return [[-y, x], [y, 1 - x], [1 - y, x]]
        if n == 2:
            return [[-8*x*y + 2*y, 8*x**2 - 4*x],
                    [-8*y**2 + 4*y, 8*x*y - 2*x],
                    [-8*x*y - 8*y**2 + 6*y, 8*x**2 + 8*x*y - 12*x - 6*y + 4],
                    [8*y**2 - 4*y, -8*x*y + 2*x + 6*y - 2],
                    [8*x*y - 6*x + 8*y**2 - 12*y + 4, -8*x**2 - 8*x*y + 6*x],
                    [-8*x*y + 6*x + 2*y - 2, 8*x**2 - 4*x],
                    [-8*x*y - 16*y**2 + 16*y, 8*x**2 + 16*x*y - 8*x],
                    [16*x*y + 8*y**2 - 8*y, -16*x**2 - 8*x*y + 16*x]]
        if n == 3:
            return [
                [-45*x**2*y + 30*x*y - 3*y, 45*x**3 - 45*x**2 + 9*x],
                [-45*y**3 + 45*y**2 - 9*y, 45*x*y**2 - 30*x*y + 3*x],
                [-45*x**2*y/4 - 45*x*y**2 + 45*x*y/2 - 45*y**3/4 + 75*y**2/4 - 6*y, 45*x**3/4 + 45*x**2*y - 75*x**2/4 + 45*x*y**2/4 - 45*x*y/2 + 6*x],  # noqa: E501
                [45*x**2*y + 90*x*y**2 - 60*x*y + 45*y**3 - 60*y**2 + 18*y, -45*x**3 - 90*x**2*y + 90*x**2 - 45*x*y**2 + 120*x*y - 54*x + 30*y**2 - 36*y + 9],  # noqa: E501
                [45*y**3 - 45*y**2 + 9*y, -45*x*y**2 + 30*x*y - 3*x + 30*y**2 - 24*y + 3],
                [45*x**2*y/4 - 45*x*y**2/2 - 45*y**3/2 + 105*y**2/4 - 21*y/4, -45*x**3/4 + 45*x**2*y/2 + 15*x**2 + 45*x*y**2/2 - 75*x*y/2 - 9*x/4 - 15*y**2 + 15*y - 3/2],  # noqa: E501
                [-45*x**2*y + 30*x**2 - 90*x*y**2 + 120*x*y - 36*x - 45*y**3 + 90*y**2 - 54*y + 9, 45*x**3 + 90*x**2*y - 60*x**2 + 45*x*y**2 - 60*x*y + 18*x],  # noqa: E501
                [-45*x**2*y + 30*x**2 + 30*x*y - 24*x - 3*y + 3, 45*x**3 - 45*x**2 + 9*x],
                [45*x**2*y/2 - 15*x**2 + 45*x*y**2/2 - 75*x*y/2 + 15*x - 45*y**3/4 + 15*y**2 - 9*y/4 - 3/2, -45*x**3/2 - 45*x**2*y/2 + 105*x**2/4 + 45*x*y**2/4 - 21*x/4],  # noqa: E501
                [90*x**2*y + 360*x*y**2 - 300*x*y + 270*y**3 - 450*y**2 + 180*y, -90*x**3 - 360*x**2*y + 150*x**2 - 270*x*y**2 + 300*x*y - 60*x],  # noqa: E501
                [-270*x**2*y - 360*x*y**2 + 300*x*y - 90*y**3 + 150*y**2 - 60*y, 270*x**3 + 360*x**2*y - 450*x**2 + 90*x*y**2 - 300*x*y + 180*x],  # noqa: E501
                [-180*x**2*y - 360*x*y**2 + 360*x*y + 60*y**2 - 60*y, 180*x**3 + 360*x**2*y - 240*x**2 - 120*x*y + 60*x],  # noqa: E501
                [270*x**2*y + 180*x*y**2 - 240*x*y - 30*y**2 + 30*y, -270*x**3 - 180*x**2*y + 360*x**2 + 60*x*y - 90*x],
                [-180*x*y**2 + 60*x*y - 270*y**3 + 360*y**2 - 90*y, 180*x**2*y - 30*x**2 + 270*x*y**2 - 240*x*y + 30*x],
                [360*x*y**2 - 120*x*y + 180*y**3 - 240*y**2 + 60*y, -360*x**2*y + 60*x**2 - 180*x*y**2 + 360*x*y - 60*x]
            ]
    if celltype == basix.CellType.tetrahedron:
        if n == 1:
            return [[0, -z, y],
                    [-z, 0, x],
                    [-y, x, 0],
                    [z, z, -x - y + 1],
                    [y, -x - z + 1, y],
                    [-y - z + 1, x, x]]
        if n == 2:
            return [
                [0, -8*y*z + 2*z, 8*y**2 - 4*y],
                [0, -8*z**2 + 4*z, 8*y*z - 2*y],
                [-8*x*z + 2*z, 0, 8*x**2 - 4*x],
                [-8*z**2 + 4*z, 0, 8*x*z - 2*x],
                [-8*x*y + 2*y, 8*x**2 - 4*x, 0],
                [-8*y**2 + 4*y, 8*x*y - 2*x, 0],
                [-8*x*z - 8*y*z - 8*z**2 + 6*z, -8*x*z - 8*y*z - 8*z**2 + 6*z, 8*x**2 + 16*x*y + 8*x*z - 12*x + 8*y**2 + 8*y*z - 12*y - 6*z + 4],  # noqa: E501
                [8*z**2 - 4*z, 8*z**2 - 4*z, -8*x*z + 2*x - 8*y*z + 2*y + 6*z - 2],
                [-8*x*y - 8*y**2 - 8*y*z + 6*y, 8*x**2 + 8*x*y + 16*x*z - 12*x + 8*y*z - 6*y + 8*z**2 - 12*z + 4, -8*x*y - 8*y**2 - 8*y*z + 6*y],  # noqa: E501
                [8*y**2 - 4*y, -8*x*y + 2*x - 8*y*z + 6*y + 2*z - 2, 8*y**2 - 4*y],
                [8*x*y + 8*x*z - 6*x + 8*y**2 + 16*y*z - 12*y + 8*z**2 - 12*z + 4, -8*x**2 - 8*x*y - 8*x*z + 6*x, -8*x**2 - 8*x*y - 8*x*z + 6*x],  # noqa: E501
                [-8*x*y - 8*x*z + 6*x + 2*y + 2*z - 2, 8*x**2 - 4*x, 8*x**2 - 4*x],
                [-8*y*z, 16*x*z, -8*x*y],
                [-8*y*z, -8*x*z, 16*x*y],
                [8*y*z, -16*x*z - 8*y*z - 16*z**2 + 16*z, 8*x*y + 8*y**2 + 16*y*z - 8*y],
                [8*y*z, 8*x*z + 16*y*z + 8*z**2 - 8*z, -16*x*y - 16*y**2 - 8*y*z + 16*y],
                [-8*x*z - 16*y*z - 16*z**2 + 16*z, 8*x*z, 8*x**2 + 8*x*y + 16*x*z - 8*x],
                [16*x*z + 8*y*z + 8*z**2 - 8*z, 8*x*z, -16*x**2 - 16*x*y - 8*x*z + 16*x],
                [-8*x*y - 16*y**2 - 16*y*z + 16*y, 8*x**2 + 16*x*y + 8*x*z - 8*x, 8*x*y],
                [16*x*y + 8*y**2 + 8*y*z - 8*y, -16*x**2 - 8*x*y - 16*x*z + 16*x, 8*x*y]
            ]
        if n == 3:
            return [
                [0, -45*y**2*z + 30*y*z - 3*z, 45*y**3 - 45*y**2 + 9*y],
                [0, -45*z**3 + 45*z**2 - 9*z, 45*y*z**2 - 30*y*z + 3*y],
                [0, -45*y**2*z/4 - 45*y*z**2 + 45*y*z/2 - 45*z**3/4 + 75*z**2/4 - 6*z, 45*y**3/4 + 45*y**2*z - 75*y**2/4 + 45*y*z**2/4 - 45*y*z/2 + 6*y],  # noqa: E501
                [-45*x**2*z + 30*x*z - 3*z, 0, 45*x**3 - 45*x**2 + 9*x],
                [-45*z**3 + 45*z**2 - 9*z, 0, 45*x*z**2 - 30*x*z + 3*x],
                [-45*x**2*z/4 - 45*x*z**2 + 45*x*z/2 - 45*z**3/4 + 75*z**2/4 - 6*z, 0, 45*x**3/4 + 45*x**2*z - 75*x**2/4 + 45*x*z**2/4 - 45*x*z/2 + 6*x],  # noqa: E501
                [-45*x**2*y + 30*x*y - 3*y, 45*x**3 - 45*x**2 + 9*x, 0],
                [-45*y**3 + 45*y**2 - 9*y, 45*x*y**2 - 30*x*y + 3*x, 0],
                [-45*x**2*y/4 - 45*x*y**2 + 45*x*y/2 - 45*y**3/4 + 75*y**2/4 - 6*y, 45*x**3/4 + 45*x**2*y - 75*x**2/4 + 45*x*y**2/4 - 45*x*y/2 + 6*x, 0],  # noqa: E501
                [45*x**2*z + 90*x*y*z + 90*x*z**2 - 60*x*z + 45*y**2*z + 90*y*z**2 - 60*y*z + 45*z**3 - 60*z**2 + 18*z, 45*x**2*z + 90*x*y*z + 90*x*z**2 - 60*x*z + 45*y**2*z + 90*y*z**2 - 60*y*z + 45*z**3 - 60*z**2 + 18*z, -45*x**3 - 135*x**2*y - 90*x**2*z + 90*x**2 - 135*x*y**2 - 180*x*y*z + 180*x*y - 45*x*z**2 + 120*x*z - 54*x - 45*y**3 - 90*y**2*z + 90*y**2 - 45*y*z**2 + 120*y*z - 54*y + 30*z**2 - 36*z + 9],  # noqa: E501
                [45*z**3 - 45*z**2 + 9*z, 45*z**3 - 45*z**2 + 9*z, -45*x*z**2 + 30*x*z - 3*x - 45*y*z**2 + 30*y*z - 3*y + 30*z**2 - 24*z + 3],  # noqa: E501
                [45*x**2*z/4 + 45*x*y*z/2 - 45*x*z**2/2 + 45*y**2*z/4 - 45*y*z**2/2 - 45*z**3/2 + 105*z**2/4 - 21*z/4, 45*x**2*z/4 + 45*x*y*z/2 - 45*x*z**2/2 + 45*y**2*z/4 - 45*y*z**2/2 - 45*z**3/2 + 105*z**2/4 - 21*z/4, -45*x**3/4 - 135*x**2*y/4 + 45*x**2*z/2 + 15*x**2 - 135*x*y**2/4 + 45*x*y*z + 30*x*y + 45*x*z**2/2 - 75*x*z/2 - 9*x/4 - 45*y**3/4 + 45*y**2*z/2 + 15*y**2 + 45*y*z**2/2 - 75*y*z/2 - 9*y/4 - 15*z**2 + 15*z - 3/2],  # noqa: E501
                [45*x**2*y + 90*x*y**2 + 90*x*y*z - 60*x*y + 45*y**3 + 90*y**2*z - 60*y**2 + 45*y*z**2 - 60*y*z + 18*y, -45*x**3 - 90*x**2*y - 135*x**2*z + 90*x**2 - 45*x*y**2 - 180*x*y*z + 120*x*y - 135*x*z**2 + 180*x*z - 54*x - 45*y**2*z + 30*y**2 - 90*y*z**2 + 120*y*z - 36*y - 45*z**3 + 90*z**2 - 54*z + 9, 45*x**2*y + 90*x*y**2 + 90*x*y*z - 60*x*y + 45*y**3 + 90*y**2*z - 60*y**2 + 45*y*z**2 - 60*y*z + 18*y],  # noqa: E501
                [45*y**3 - 45*y**2 + 9*y, -45*x*y**2 + 30*x*y - 3*x - 45*y**2*z + 30*y**2 + 30*y*z - 24*y - 3*z + 3, 45*y**3 - 45*y**2 + 9*y],  # noqa: E501
                [45*x**2*y/4 - 45*x*y**2/2 + 45*x*y*z/2 - 45*y**3/2 - 45*y**2*z/2 + 105*y**2/4 + 45*y*z**2/4 - 21*y/4, -45*x**3/4 + 45*x**2*y/2 - 135*x**2*z/4 + 15*x**2 + 45*x*y**2/2 + 45*x*y*z - 75*x*y/2 - 135*x*z**2/4 + 30*x*z - 9*x/4 + 45*y**2*z/2 - 15*y**2 + 45*y*z**2/2 - 75*y*z/2 + 15*y - 45*z**3/4 + 15*z**2 - 9*z/4 - 3/2, 45*x**2*y/4 - 45*x*y**2/2 + 45*x*y*z/2 - 45*y**3/2 - 45*y**2*z/2 + 105*y**2/4 + 45*y*z**2/4 - 21*y/4],  # noqa: E501
                [-45*x**2*y - 45*x**2*z + 30*x**2 - 90*x*y**2 - 180*x*y*z + 120*x*y - 90*x*z**2 + 120*x*z - 36*x - 45*y**3 - 135*y**2*z + 90*y**2 - 135*y*z**2 + 180*y*z - 54*y - 45*z**3 + 90*z**2 - 54*z + 9, 45*x**3 + 90*x**2*y + 90*x**2*z - 60*x**2 + 45*x*y**2 + 90*x*y*z - 60*x*y + 45*x*z**2 - 60*x*z + 18*x, 45*x**3 + 90*x**2*y + 90*x**2*z - 60*x**2 + 45*x*y**2 + 90*x*y*z - 60*x*y + 45*x*z**2 - 60*x*z + 18*x],  # noqa: E501
                [-45*x**2*y - 45*x**2*z + 30*x**2 + 30*x*y + 30*x*z - 24*x - 3*y - 3*z + 3, 45*x**3 - 45*x**2 + 9*x, 45*x**3 - 45*x**2 + 9*x],  # noqa: E501
                [45*x**2*y/2 + 45*x**2*z/2 - 15*x**2 + 45*x*y**2/2 + 45*x*y*z - 75*x*y/2 + 45*x*z**2/2 - 75*x*z/2 + 15*x - 45*y**3/4 - 135*y**2*z/4 + 15*y**2 - 135*y*z**2/4 + 30*y*z - 9*y/4 - 45*z**3/4 + 15*z**2 - 9*z/4 - 3/2, -45*x**3/2 - 45*x**2*y/2 - 45*x**2*z/2 + 105*x**2/4 + 45*x*y**2/4 + 45*x*y*z/2 + 45*x*z**2/4 - 21*x/4, -45*x**3/2 - 45*x**2*y/2 - 45*x**2*z/2 + 105*x**2/4 + 45*x*y**2/4 + 45*x*y*z/2 + 45*x*z**2/4 - 21*x/4],  # noqa: E501
                [-180*x*y*z + 30*y*z, 270*x**2*z - 90*x*z, -90*x**2*y + 30*x*y],
                [-180*x*y*z + 30*y*z, -90*x**2*z + 30*x*z, 270*x**2*y - 90*x*y],
                [-180*y**2*z + 60*y*z, 360*x*y*z - 60*x*z, -180*x*y**2 + 60*x*y],
                [-90*y**2*z + 30*y*z, -180*x*y*z + 30*x*z, 270*x*y**2 - 90*x*y],
                [-90*y*z**2 + 30*y*z, 270*x*z**2 - 90*x*z, -180*x*y*z + 30*x*y],
                [-180*y*z**2 + 60*y*z, -180*x*z**2 + 60*x*z, 360*x*y*z - 60*x*y],
                [-180*x*y*z - 180*y**2*z - 180*y*z**2 + 150*y*z, 270*x**2*z + 360*x*y*z + 540*x*z**2 - 450*x*z + 90*y**2*z + 360*y*z**2 - 300*y*z + 270*z**3 - 450*z**2 + 180*z, -90*x**2*y - 180*x*y**2 - 360*x*y*z + 150*x*y - 90*y**3 - 360*y**2*z + 150*y**2 - 270*y*z**2 + 300*y*z - 60*y],  # noqa: E501
                [-180*x*y*z - 180*y**2*z - 180*y*z**2 + 150*y*z, -90*x**2*z - 360*x*y*z - 180*x*z**2 + 150*x*z - 270*y**2*z - 360*y*z**2 + 300*y*z - 90*z**3 + 150*z**2 - 60*z, 270*x**2*y + 540*x*y**2 + 360*x*y*z - 450*x*y + 270*y**3 + 360*y**2*z - 450*y**2 + 90*y*z**2 - 300*y*z + 180*y],  # noqa: E501
                [180*y**2*z - 60*y*z, -360*x*y*z + 60*x*z - 180*y**2*z - 360*y*z**2 + 360*y*z + 60*z**2 - 60*z, 180*x*y**2 - 60*x*y + 180*y**3 + 360*y**2*z - 240*y**2 - 120*y*z + 60*y],  # noqa: E501
                [90*y**2*z - 30*y*z, 180*x*y*z - 30*x*z + 270*y**2*z + 180*y*z**2 - 240*y*z - 30*z**2 + 30*z, -270*x*y**2 + 90*x*y - 270*y**3 - 180*y**2*z + 360*y**2 + 60*y*z - 90*y],  # noqa: E501
                [90*y*z**2 - 30*y*z, -270*x*z**2 + 90*x*z - 180*y*z**2 + 60*y*z - 270*z**3 + 360*z**2 - 90*z, 180*x*y*z - 30*x*y + 180*y**2*z - 30*y**2 + 270*y*z**2 - 240*y*z + 30*y],  # noqa: E501
                [180*y*z**2 - 60*y*z, 180*x*z**2 - 60*x*z + 360*y*z**2 - 120*y*z + 180*z**3 - 240*z**2 + 60*z, -360*x*y*z + 60*x*y - 360*y**2*z + 60*y**2 - 180*y*z**2 + 360*y*z - 60*y],  # noqa: E501
                [90*x**2*z + 360*x*y*z + 360*x*z**2 - 300*x*z + 270*y**2*z + 540*y*z**2 - 450*y*z + 270*z**3 - 450*z**2 + 180*z, -180*x**2*z - 180*x*y*z - 180*x*z**2 + 150*x*z, -90*x**3 - 180*x**2*y - 360*x**2*z + 150*x**2 - 90*x*y**2 - 360*x*y*z + 150*x*y - 270*x*z**2 + 300*x*z - 60*x],  # noqa: E501
                [-270*x**2*z - 360*x*y*z - 360*x*z**2 + 300*x*z - 90*y**2*z - 180*y*z**2 + 150*y*z - 90*z**3 + 150*z**2 - 60*z, -180*x**2*z - 180*x*y*z - 180*x*z**2 + 150*x*z, 270*x**3 + 540*x**2*y + 360*x**2*z - 450*x**2 + 270*x*y**2 + 360*x*y*z - 450*x*y + 90*x*z**2 - 300*x*z + 180*x],  # noqa: E501
                [-180*x**2*z - 360*x*y*z - 360*x*z**2 + 360*x*z + 60*y*z + 60*z**2 - 60*z, 180*x**2*z - 60*x*z, 180*x**3 + 180*x**2*y + 360*x**2*z - 240*x**2 - 60*x*y - 120*x*z + 60*x],  # noqa: E501
                [270*x**2*z + 180*x*y*z + 180*x*z**2 - 240*x*z - 30*y*z - 30*z**2 + 30*z, 90*x**2*z - 30*x*z, -270*x**3 - 270*x**2*y - 180*x**2*z + 360*x**2 + 90*x*y + 60*x*z - 90*x],  # noqa: E501
                [-180*x*z**2 + 60*x*z - 270*y*z**2 + 90*y*z - 270*z**3 + 360*z**2 - 90*z, 90*x*z**2 - 30*x*z, 180*x**2*z - 30*x**2 + 180*x*y*z - 30*x*y + 270*x*z**2 - 240*x*z + 30*x],  # noqa: E501
                [360*x*z**2 - 120*x*z + 180*y*z**2 - 60*y*z + 180*z**3 - 240*z**2 + 60*z, 180*x*z**2 - 60*x*z, -360*x**2*z + 60*x**2 - 360*x*y*z + 60*x*y - 180*x*z**2 + 360*x*z - 60*x],  # noqa: E501
                [90*x**2*y + 360*x*y**2 + 360*x*y*z - 300*x*y + 270*y**3 + 540*y**2*z - 450*y**2 + 270*y*z**2 - 450*y*z + 180*y, -90*x**3 - 360*x**2*y - 180*x**2*z + 150*x**2 - 270*x*y**2 - 360*x*y*z + 300*x*y - 90*x*z**2 + 150*x*z - 60*x, -180*x**2*y - 180*x*y**2 - 180*x*y*z + 150*x*y],  # noqa: E501
                [-270*x**2*y - 360*x*y**2 - 360*x*y*z + 300*x*y - 90*y**3 - 180*y**2*z + 150*y**2 - 90*y*z**2 + 150*y*z - 60*y, 270*x**3 + 360*x**2*y + 540*x**2*z - 450*x**2 + 90*x*y**2 + 360*x*y*z - 300*x*y + 270*x*z**2 - 450*x*z + 180*x, -180*x**2*y - 180*x*y**2 - 180*x*y*z + 150*x*y],  # noqa: E501
                [-180*x**2*y - 360*x*y**2 - 360*x*y*z + 360*x*y + 60*y**2 + 60*y*z - 60*y, 180*x**3 + 360*x**2*y + 180*x**2*z - 240*x**2 - 120*x*y - 60*x*z + 60*x, 180*x**2*y - 60*x*y],  # noqa: E501
                [270*x**2*y + 180*x*y**2 + 180*x*y*z - 240*x*y - 30*y**2 - 30*y*z + 30*y, -270*x**3 - 180*x**2*y - 270*x**2*z + 360*x**2 + 60*x*y + 90*x*z - 90*x, 90*x**2*y - 30*x*y],  # noqa: E501
                [-180*x*y**2 + 60*x*y - 270*y**3 - 270*y**2*z + 360*y**2 + 90*y*z - 90*y, 180*x**2*y - 30*x**2 + 270*x*y**2 + 180*x*y*z - 240*x*y - 30*x*z + 30*x, 90*x*y**2 - 30*x*y],  # noqa: E501
                [360*x*y**2 - 120*x*y + 180*y**3 + 180*y**2*z - 240*y**2 - 60*y*z + 60*y, -360*x**2*y + 60*x**2 - 180*x*y**2 - 360*x*y*z + 360*x*y + 60*x*z - 60*x, 180*x*y**2 - 60*x*y],  # noqa: E501
                [-360*x*y*z - 540*y**2*z - 540*y*z**2 + 540*y*z, 180*x**2*z + 360*x*y*z + 180*x*z**2 - 180*x*z, 180*x**2*y + 180*x*y**2 + 360*x*y*z - 180*x*y],  # noqa: E501
                [360*x*y*z + 180*y**2*z + 180*y*z**2 - 180*y*z, -540*x**2*z - 360*x*y*z - 540*x*z**2 + 540*x*z, 180*x**2*y + 180*x*y**2 + 360*x*y*z - 180*x*y],  # noqa: E501
                [360*x*y*z + 180*y**2*z + 180*y*z**2 - 180*y*z, 180*x**2*z + 360*x*y*z + 180*x*z**2 - 180*x*z, -540*x**2*y - 540*x*y**2 - 360*x*y*z + 540*x*y]]  # noqa: E501

    raise NotImplementedError


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_tri(degree):
    celltype = basix.CellType.triangle
    g = sympy_nedelec(celltype, degree)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    nedelec = basix.create_element(
        basix.ElementFamily.N1E, basix.CellType.triangle, degree, basix.LagrangeVariant.equispaced)
    pts = basix.create_lattice(celltype, 6, basix.LatticeType.equispaced, True)
    nderiv = 3
    wtab = nedelec.tabulate(nderiv, pts)

    for kx in range(nderiv + 1):
        for ky in range(nderiv + 1 - kx):
            wsym = numpy.zeros_like(wtab[0])
            for i, gi in enumerate(g):
                for j, gij in enumerate(gi):
                    wd = sympy.diff(gij, x, kx, y, ky)
                    for k, p in enumerate(pts):
                        wsym[k, i, j] = wd.subs([(x, p[0]), (y, p[1])])

            assert numpy.isclose(wtab[basix.index(kx, ky)], wsym).all()


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_tet(degree):
    celltype = basix.CellType.tetrahedron
    g = sympy_nedelec(celltype, degree)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    nedelec = basix.create_element(
        basix.ElementFamily.N1E, basix.CellType.tetrahedron, degree, basix.LagrangeVariant.equispaced)

    pts = basix.create_lattice(celltype, 6, basix.LatticeType.equispaced, True)
    nderiv = 1
    wtab = nedelec.tabulate(nderiv, pts)

    for kx in range(nderiv + 1):
        for ky in range(nderiv + 1 - kx):
            for kz in range(nderiv + 1 - kx - ky):
                wsym = numpy.zeros_like(wtab[0])
                for i, gi in enumerate(g):
                    for j, gij in enumerate(gi):
                        wd = sympy.diff(gij, x, kx, y, ky, z, kz)
                        for k, p in enumerate(pts):
                            wsym[k, i, j] = wd.subs([(x, p[0]), (y, p[1]), (z, p[2])])

                assert numpy.isclose(wtab[basix.index(kx, ky, kz)], wsym).all()
