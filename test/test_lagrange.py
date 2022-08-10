# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import numpy
import pytest
import sympy


def sympy_lagrange(celltype, n):
    # These basis functions were computed using symfem. They can be recomputed
    # by running (eg):
    #    import symfem
    #    e = symfem.create_element("triangle", "Lagrange", 2)
    #    print(e.get_basis_functions())
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")

    if celltype == basix.CellType.interval:
        if n == 1:
            return [1 - x, x]
        if n == 2:
            return [2*x**2 - 3*x + 1, 2*x**2 - x, -4*x**2 + 4*x]
        if n == 3:
            return [-9*x**3/2 + 9*x**2 - 11*x/2 + 1,
                    9*x**3/2 - 9*x**2/2 + x,
                    27*x**3/2 - 45*x**2/2 + 9*x,
                    -27*x**3/2 + 18*x**2 - 9*x/2]
        if n == 4:
            return [32*x**4/3 - 80*x**3/3 + 70*x**2/3 - 25*x/3 + 1,
                    32*x**4/3 - 16*x**3 + 22*x**2/3 - x,
                    -128*x**4/3 + 96*x**3 - 208*x**2/3 + 16*x,
                    64*x**4 - 128*x**3 + 76*x**2 - 12*x,
                    -128*x**4/3 + 224*x**3/3 - 112*x**2/3 + 16*x/3]
        if n == 5:
            return [-625*x**5/24 + 625*x**4/8 - 2125*x**3/24 + 375*x**2/8 - 137*x/12 + 1,
                    625*x**5/24 - 625*x**4/12 + 875*x**3/24 - 125*x**2/12 + x,
                    3125*x**5/24 - 4375*x**4/12 + 8875*x**3/24 - 1925*x**2/12 + 25*x,
                    -3125*x**5/12 + 8125*x**4/12 - 7375*x**3/12 + 2675*x**2/12 - 25*x,
                    3125*x**5/12 - 625*x**4 + 6125*x**3/12 - 325*x**2/2 + 50*x/3,
                    -3125*x**5/24 + 6875*x**4/24 - 5125*x**3/24 + 1525*x**2/24 - 25*x/4]
    if celltype == basix.CellType.triangle:
        if n == 1:
            return [-x - y + 1, x, y]
        if n == 2:
            return [2*x**2 + 4*x*y - 3*x + 2*y**2 - 3*y + 1,
                    2*x**2 - x,
                    2*y**2 - y,
                    4*x*y,
                    -4*x*y - 4*y**2 + 4*y,
                    -4*x**2 - 4*x*y + 4*x]
        if n == 3:
            return [
                -9*x**3/2 - 27*x**2*y/2 + 9*x**2 - 27*x*y**2/2 + 18*x*y - 11*x/2 - 9*y**3/2 + 9*y**2 - 11*y/2 + 1,
                9*x**3/2 - 9*x**2/2 + x,
                9*y**3/2 - 9*y**2/2 + y,
                27*x**2*y/2 - 9*x*y/2,
                27*x*y**2/2 - 9*x*y/2,
                27*x**2*y/2 + 27*x*y**2 - 45*x*y/2 + 27*y**3/2 - 45*y**2/2 + 9*y,
                -27*x*y**2/2 + 9*x*y/2 - 27*y**3/2 + 18*y**2 - 9*y/2,
                27*x**3/2 + 27*x**2*y - 45*x**2/2 + 27*x*y**2/2 - 45*x*y/2 + 9*x,
                -27*x**3/2 - 27*x**2*y/2 + 18*x**2 + 9*x*y/2 - 9*x/2,
                -27*x**2*y - 27*x*y**2 + 27*x*y
            ]
        if n == 4:
            return [
                32*x**4/3 + 128*x**3*y/3 - 80*x**3/3 + 64*x**2*y**2 - 80*x**2*y + 70*x**2/3 + 128*x*y**3/3 - 80*x*y**2 + 140*x*y/3 - 25*x/3 + 32*y**4/3 - 80*y**3/3 + 70*y**2/3 - 25*y/3 + 1,  # noqa: E501
                32*x**4/3 - 16*x**3 + 22*x**2/3 - x,
                32*y**4/3 - 16*y**3 + 22*y**2/3 - y,
                128*x**3*y/3 - 32*x**2*y + 16*x*y/3,
                64*x**2*y**2 - 16*x**2*y - 16*x*y**2 + 4*x*y,
                128*x*y**3/3 - 32*x*y**2 + 16*x*y/3,
                -128*x**3*y/3 - 128*x**2*y**2 + 96*x**2*y - 128*x*y**3 + 192*x*y**2 - 208*x*y/3 - 128*y**4/3 + 96*y**3 - 208*y**2/3 + 16*y,  # noqa: E501
                64*x**2*y**2 - 16*x**2*y + 128*x*y**3 - 144*x*y**2 + 28*x*y + 64*y**4 - 128*y**3 + 76*y**2 - 12*y,
                -128*x*y**3/3 + 32*x*y**2 - 16*x*y/3 - 128*y**4/3 + 224*y**3/3 - 112*y**2/3 + 16*y/3,
                -128*x**4/3 - 128*x**3*y + 96*x**3 - 128*x**2*y**2 + 192*x**2*y - 208*x**2/3 - 128*x*y**3/3 + 96*x*y**2 - 208*x*y/3 + 16*x,  # noqa: E501
                64*x**4 + 128*x**3*y - 128*x**3 + 64*x**2*y**2 - 144*x**2*y + 76*x**2 - 16*x*y**2 + 28*x*y - 12*x,
                -128*x**4/3 - 128*x**3*y/3 + 224*x**3/3 + 32*x**2*y - 112*x**2/3 - 16*x*y/3 + 16*x/3,
                128*x**3*y + 256*x**2*y**2 - 224*x**2*y + 128*x*y**3 - 224*x*y**2 + 96*x*y,
                -128*x**3*y - 128*x**2*y**2 + 160*x**2*y + 32*x*y**2 - 32*x*y,
                -128*x**2*y**2 + 32*x**2*y - 128*x*y**3 + 160*x*y**2 - 32*x*y
            ]
        if n == 5:
            return [
                -625*x**5/24 - 3125*x**4*y/24 + 625*x**4/8 - 3125*x**3*y**2/12 + 625*x**3*y/2 - 2125*x**3/24 - 3125*x**2*y**3/12 + 1875*x**2*y**2/4 - 2125*x**2*y/8 + 375*x**2/8 - 3125*x*y**4/24 + 625*x*y**3/2 - 2125*x*y**2/8 + 375*x*y/4 - 137*x/12 - 625*y**5/24 + 625*y**4/8 - 2125*y**3/24 + 375*y**2/8 - 137*y/12 + 1,  # noqa: E501
                625*x**5/24 - 625*x**4/12 + 875*x**3/24 - 125*x**2/12 + x,
                625*y**5/24 - 625*y**4/12 + 875*y**3/24 - 125*y**2/12 + y,
                3125*x**4*y/24 - 625*x**3*y/4 + 1375*x**2*y/24 - 25*x*y/4,
                3125*x**3*y**2/12 - 625*x**3*y/12 - 625*x**2*y**2/4 + 125*x**2*y/4 + 125*x*y**2/6 - 25*x*y/6,
                3125*x**2*y**3/12 - 625*x**2*y**2/4 + 125*x**2*y/6 - 625*x*y**3/12 + 125*x*y**2/4 - 25*x*y/6,
                3125*x*y**4/24 - 625*x*y**3/4 + 1375*x*y**2/24 - 25*x*y/4,
                3125*x**4*y/24 + 3125*x**3*y**2/6 - 4375*x**3*y/12 + 3125*x**2*y**3/4 - 4375*x**2*y**2/4 + 8875*x**2*y/24 + 3125*x*y**4/6 - 4375*x*y**3/4 + 8875*x*y**2/12 - 1925*x*y/12 + 3125*y**5/24 - 4375*y**4/12 + 8875*y**3/24 - 1925*y**2/12 + 25*y,  # noqa: E501
                -3125*x**3*y**2/12 + 625*x**3*y/12 - 3125*x**2*y**3/4 + 3125*x**2*y**2/4 - 125*x**2*y - 3125*x*y**4/4 + 5625*x*y**3/4 - 8875*x*y**2/12 + 1175*x*y/12 - 3125*y**5/12 + 8125*y**4/12 - 7375*y**3/12 + 2675*y**2/12 - 25*y,  # noqa: E501
                3125*x**2*y**3/12 - 625*x**2*y**2/4 + 125*x**2*y/6 + 3125*x*y**4/6 - 3125*x*y**3/4 + 3875*x*y**2/12 - 75*x*y/2 + 3125*y**5/12 - 625*y**4 + 6125*y**3/12 - 325*y**2/2 + 50*y/3,  # noqa: E501
                -3125*x*y**4/24 + 625*x*y**3/4 - 1375*x*y**2/24 + 25*x*y/4 - 3125*y**5/24 + 6875*y**4/24 - 5125*y**3/24 + 1525*y**2/24 - 25*y/4,  # noqa: E501
                3125*x**5/24 + 3125*x**4*y/6 - 4375*x**4/12 + 3125*x**3*y**2/4 - 4375*x**3*y/4 + 8875*x**3/24 + 3125*x**2*y**3/6 - 4375*x**2*y**2/4 + 8875*x**2*y/12 - 1925*x**2/12 + 3125*x*y**4/24 - 4375*x*y**3/12 + 8875*x*y**2/24 - 1925*x*y/12 + 25*x,  # noqa: E501
                -3125*x**5/12 - 3125*x**4*y/4 + 8125*x**4/12 - 3125*x**3*y**2/4 + 5625*x**3*y/4 - 7375*x**3/12 - 3125*x**2*y**3/12 + 3125*x**2*y**2/4 - 8875*x**2*y/12 + 2675*x**2/12 + 625*x*y**3/12 - 125*x*y**2 + 1175*x*y/12 - 25*x,  # noqa: E501
                3125*x**5/12 + 3125*x**4*y/6 - 625*x**4 + 3125*x**3*y**2/12 - 3125*x**3*y/4 + 6125*x**3/12 - 625*x**2*y**2/4 + 3875*x**2*y/12 - 325*x**2/2 + 125*x*y**2/6 - 75*x*y/2 + 50*x/3,  # noqa: E501
                -3125*x**5/24 - 3125*x**4*y/24 + 6875*x**4/24 + 625*x**3*y/4 - 5125*x**3/24 - 1375*x**2*y/24 + 1525*x**2/24 + 25*x*y/4 - 25*x/4,  # noqa: E501
                -3125*x**4*y/6 - 3125*x**3*y**2/2 + 1250*x**3*y - 3125*x**2*y**3/2 + 2500*x**2*y**2 - 5875*x**2*y/6 - 3125*x*y**4/6 + 1250*x*y**3 - 5875*x*y**2/6 + 250*x*y,  # noqa: E501
                3125*x**4*y/4 + 3125*x**3*y**2/2 - 3125*x**3*y/2 + 3125*x**2*y**3/4 - 6875*x**2*y**2/4 + 3625*x**2*y/4 - 625*x*y**3/4 + 1125*x*y**2/4 - 125*x*y,  # noqa: E501
                -3125*x**4*y/6 - 3125*x**3*y**2/6 + 2500*x**3*y/3 + 625*x**2*y**2/2 - 2125*x**2*y/6 - 125*x*y**2/3 + 125*x*y/3,  # noqa: E501
                3125*x**3*y**2/4 - 625*x**3*y/4 + 3125*x**2*y**3/2 - 6875*x**2*y**2/4 + 1125*x**2*y/4 + 3125*x*y**4/4 - 3125*x*y**3/2 + 3625*x*y**2/4 - 125*x*y,  # noqa: E501
                -3125*x**3*y**2/4 + 625*x**3*y/4 - 3125*x**2*y**3/4 + 4375*x**2*y**2/4 - 375*x**2*y/2 + 625*x*y**3/4 - 375*x*y**2/2 + 125*x*y/4,  # noqa: E501
                -3125*x**2*y**3/6 + 625*x**2*y**2/2 - 125*x**2*y/3 - 3125*x*y**4/6 + 2500*x*y**3/3 - 2125*x*y**2/6 + 125*x*y/3  # noqa: E501
            ]
    if celltype == basix.CellType.tetrahedron:
        if n == 1:
            return [-x - y - z + 1, x, y, z]
        if n == 2:
            return [2*x**2 + 4*x*y + 4*x*z - 3*x + 2*y**2 + 4*y*z - 3*y + 2*z**2 - 3*z + 1,
                    2*x**2 - x,
                    2*y**2 - y,
                    2*z**2 - z,
                    4*y*z,
                    4*x*z,
                    4*x*y,
                    -4*x*z - 4*y*z - 4*z**2 + 4*z,
                    -4*x*y - 4*y**2 - 4*y*z + 4*y,
                    -4*x**2 - 4*x*y - 4*x*z + 4*x]
        if n == 3:
            return [
                -9*x**3/2 - 27*x**2*y/2 - 27*x**2*z/2 + 9*x**2 - 27*x*y**2/2 - 27*x*y*z + 18*x*y - 27*x*z**2/2 + 18*x*z - 11*x/2 - 9*y**3/2 - 27*y**2*z/2 + 9*y**2 - 27*y*z**2/2 + 18*y*z - 11*y/2 - 9*z**3/2 + 9*z**2 - 11*z/2 + 1,  # noqa: E501
                9*x**3/2 - 9*x**2/2 + x,
                9*y**3/2 - 9*y**2/2 + y,
                9*z**3/2 - 9*z**2/2 + z,
                27*y**2*z/2 - 9*y*z/2,
                27*y*z**2/2 - 9*y*z/2,
                27*x**2*z/2 - 9*x*z/2,
                27*x*z**2/2 - 9*x*z/2,
                27*x**2*y/2 - 9*x*y/2,
                27*x*y**2/2 - 9*x*y/2,
                27*x**2*z/2 + 27*x*y*z + 27*x*z**2 - 45*x*z/2 + 27*y**2*z/2 + 27*y*z**2 - 45*y*z/2 + 27*z**3/2 - 45*z**2/2 + 9*z,  # noqa: E501
                -27*x*z**2/2 + 9*x*z/2 - 27*y*z**2/2 + 9*y*z/2 - 27*z**3/2 + 18*z**2 - 9*z/2,
                27*x**2*y/2 + 27*x*y**2 + 27*x*y*z - 45*x*y/2 + 27*y**3/2 + 27*y**2*z - 45*y**2/2 + 27*y*z**2/2 - 45*y*z/2 + 9*y,  # noqa: E501
                -27*x*y**2/2 + 9*x*y/2 - 27*y**3/2 - 27*y**2*z/2 + 18*y**2 + 9*y*z/2 - 9*y/2,
                27*x**3/2 + 27*x**2*y + 27*x**2*z - 45*x**2/2 + 27*x*y**2/2 + 27*x*y*z - 45*x*y/2 + 27*x*z**2/2 - 45*x*z/2 + 9*x,  # noqa: E501
                -27*x**3/2 - 27*x**2*y/2 - 27*x**2*z/2 + 18*x**2 + 9*x*y/2 + 9*x*z/2 - 9*x/2,
                27*x*y*z,
                -27*x*y*z - 27*y**2*z - 27*y*z**2 + 27*y*z,
                -27*x**2*z - 27*x*y*z - 27*x*z**2 + 27*x*z,
                -27*x**2*y - 27*x*y**2 - 27*x*y*z + 27*x*y
            ]
        if n == 4:
            return [
                32*x**4/3 + 128*x**3*y/3 + 128*x**3*z/3 - 80*x**3/3 + 64*x**2*y**2 + 128*x**2*y*z - 80*x**2*y + 64*x**2*z**2 - 80*x**2*z + 70*x**2/3 + 128*x*y**3/3 + 128*x*y**2*z - 80*x*y**2 + 128*x*y*z**2 - 160*x*y*z + 140*x*y/3 + 128*x*z**3/3 - 80*x*z**2 + 140*x*z/3 - 25*x/3 + 32*y**4/3 + 128*y**3*z/3 - 80*y**3/3 + 64*y**2*z**2 - 80*y**2*z + 70*y**2/3 + 128*y*z**3/3 - 80*y*z**2 + 140*y*z/3 - 25*y/3 + 32*z**4/3 - 80*z**3/3 + 70*z**2/3 - 25*z/3 + 1,  # noqa: E501
                32*x**4/3 - 16*x**3 + 22*x**2/3 - x,
                32*y**4/3 - 16*y**3 + 22*y**2/3 - y,
                32*z**4/3 - 16*z**3 + 22*z**2/3 - z,
                128*y**3*z/3 - 32*y**2*z + 16*y*z/3,
                64*y**2*z**2 - 16*y**2*z - 16*y*z**2 + 4*y*z,
                128*y*z**3/3 - 32*y*z**2 + 16*y*z/3,
                128*x**3*z/3 - 32*x**2*z + 16*x*z/3,
                64*x**2*z**2 - 16*x**2*z - 16*x*z**2 + 4*x*z,
                128*x*z**3/3 - 32*x*z**2 + 16*x*z/3,
                128*x**3*y/3 - 32*x**2*y + 16*x*y/3,
                64*x**2*y**2 - 16*x**2*y - 16*x*y**2 + 4*x*y,
                128*x*y**3/3 - 32*x*y**2 + 16*x*y/3,
                -128*x**3*z/3 - 128*x**2*y*z - 128*x**2*z**2 + 96*x**2*z - 128*x*y**2*z - 256*x*y*z**2 + 192*x*y*z - 128*x*z**3 + 192*x*z**2 - 208*x*z/3 - 128*y**3*z/3 - 128*y**2*z**2 + 96*y**2*z - 128*y*z**3 + 192*y*z**2 - 208*y*z/3 - 128*z**4/3 + 96*z**3 - 208*z**2/3 + 16*z,  # noqa: E501
                64*x**2*z**2 - 16*x**2*z + 128*x*y*z**2 - 32*x*y*z + 128*x*z**3 - 144*x*z**2 + 28*x*z + 64*y**2*z**2 - 16*y**2*z + 128*y*z**3 - 144*y*z**2 + 28*y*z + 64*z**4 - 128*z**3 + 76*z**2 - 12*z,  # noqa: E501
                -128*x*z**3/3 + 32*x*z**2 - 16*x*z/3 - 128*y*z**3/3 + 32*y*z**2 - 16*y*z/3 - 128*z**4/3 + 224*z**3/3 - 112*z**2/3 + 16*z/3,  # noqa: E501
                -128*x**3*y/3 - 128*x**2*y**2 - 128*x**2*y*z + 96*x**2*y - 128*x*y**3 - 256*x*y**2*z + 192*x*y**2 - 128*x*y*z**2 + 192*x*y*z - 208*x*y/3 - 128*y**4/3 - 128*y**3*z + 96*y**3 - 128*y**2*z**2 + 192*y**2*z - 208*y**2/3 - 128*y*z**3/3 + 96*y*z**2 - 208*y*z/3 + 16*y,  # noqa: E501
                64*x**2*y**2 - 16*x**2*y + 128*x*y**3 + 128*x*y**2*z - 144*x*y**2 - 32*x*y*z + 28*x*y + 64*y**4 + 128*y**3*z - 128*y**3 + 64*y**2*z**2 - 144*y**2*z + 76*y**2 - 16*y*z**2 + 28*y*z - 12*y,  # noqa: E501
                -128*x*y**3/3 + 32*x*y**2 - 16*x*y/3 - 128*y**4/3 - 128*y**3*z/3 + 224*y**3/3 + 32*y**2*z - 112*y**2/3 - 16*y*z/3 + 16*y/3,  # noqa: E501
                -128*x**4/3 - 128*x**3*y - 128*x**3*z + 96*x**3 - 128*x**2*y**2 - 256*x**2*y*z + 192*x**2*y - 128*x**2*z**2 + 192*x**2*z - 208*x**2/3 - 128*x*y**3/3 - 128*x*y**2*z + 96*x*y**2 - 128*x*y*z**2 + 192*x*y*z - 208*x*y/3 - 128*x*z**3/3 + 96*x*z**2 - 208*x*z/3 + 16*x,  # noqa: E501
                64*x**4 + 128*x**3*y + 128*x**3*z - 128*x**3 + 64*x**2*y**2 + 128*x**2*y*z - 144*x**2*y + 64*x**2*z**2 - 144*x**2*z + 76*x**2 - 16*x*y**2 - 32*x*y*z + 28*x*y - 16*x*z**2 + 28*x*z - 12*x,  # noqa: E501
                -128*x**4/3 - 128*x**3*y/3 - 128*x**3*z/3 + 224*x**3/3 + 32*x**2*y + 32*x**2*z - 112*x**2/3 - 16*x*y/3 - 16*x*z/3 + 16*x/3,  # noqa: E501
                128*x**2*y*z - 32*x*y*z,
                128*x*y**2*z - 32*x*y*z,
                128*x*y*z**2 - 32*x*y*z,
                128*x**2*y*z + 256*x*y**2*z + 256*x*y*z**2 - 224*x*y*z + 128*y**3*z + 256*y**2*z**2 - 224*y**2*z + 128*y*z**3 - 224*y*z**2 + 96*y*z,  # noqa: E501
                -128*x*y**2*z + 32*x*y*z - 128*y**3*z - 128*y**2*z**2 + 160*y**2*z + 32*y*z**2 - 32*y*z,
                -128*x*y*z**2 + 32*x*y*z - 128*y**2*z**2 + 32*y**2*z - 128*y*z**3 + 160*y*z**2 - 32*y*z,
                128*x**3*z + 256*x**2*y*z + 256*x**2*z**2 - 224*x**2*z + 128*x*y**2*z + 256*x*y*z**2 - 224*x*y*z + 128*x*z**3 - 224*x*z**2 + 96*x*z,  # noqa: E501
                -128*x**3*z - 128*x**2*y*z - 128*x**2*z**2 + 160*x**2*z + 32*x*y*z + 32*x*z**2 - 32*x*z,
                -128*x**2*z**2 + 32*x**2*z - 128*x*y*z**2 + 32*x*y*z - 128*x*z**3 + 160*x*z**2 - 32*x*z,
                128*x**3*y + 256*x**2*y**2 + 256*x**2*y*z - 224*x**2*y + 128*x*y**3 + 256*x*y**2*z - 224*x*y**2 + 128*x*y*z**2 - 224*x*y*z + 96*x*y,  # noqa: E501
                -128*x**3*y - 128*x**2*y**2 - 128*x**2*y*z + 160*x**2*y + 32*x*y**2 + 32*x*y*z - 32*x*y,
                -128*x**2*y**2 + 32*x**2*y - 128*x*y**3 - 128*x*y**2*z + 160*x*y**2 + 32*x*y*z - 32*x*y,
                -256*x**2*y*z - 256*x*y**2*z - 256*x*y*z**2 + 256*x*y*z
            ]

    raise NotImplementedError


def test_point():
    lagrange = basix.create_element(basix.ElementFamily.P, basix.CellType.point, 0, True)
    assert numpy.allclose(lagrange.tabulate(0, numpy.array([[]])), [[[1]]])
    assert numpy.allclose(lagrange.tabulate(0, numpy.array([[], []])), [[[1, 1]]])


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_line(n):
    celltype = basix.CellType.interval
    g = sympy_lagrange(celltype, n)
    x = sympy.Symbol("x")
    lagrange = basix.create_element(basix.ElementFamily.P, celltype, n,
                                    basix.LagrangeVariant.equispaced)
    pts = basix.create_lattice(celltype, 6, basix.LatticeType.equispaced, True)
    nderiv = n
    wtab = lagrange.tabulate(nderiv, pts)
    for k in range(nderiv + 1):
        wsym = numpy.zeros_like(wtab[k])
        for i in range(n + 1):
            wd = sympy.diff(g[i], x, k)
            for j, p in enumerate(pts):
                wsym[j, i] = wd.subs(x, p[0])

        assert numpy.allclose(wtab[k], wsym)


@pytest.mark.parametrize("n", [1, 2])
def test_line_without_variant(n):
    celltype = basix.CellType.interval
    g = sympy_lagrange(celltype, n)
    x = sympy.Symbol("x")
    lagrange = basix.create_element(basix.ElementFamily.P, basix.CellType.interval, n)
    pts = basix.create_lattice(celltype, 6, basix.LatticeType.equispaced, True)
    nderiv = n
    wtab = lagrange.tabulate(nderiv, pts)
    for k in range(nderiv + 1):
        wsym = numpy.zeros_like(wtab[k])
        for i in range(n + 1):
            wd = sympy.diff(g[i], x, k)
            for j, p in enumerate(pts):
                wsym[j, i] = wd.subs(x, p[0])

        assert numpy.allclose(wtab[k], wsym)


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_tri(degree):
    celltype = basix.CellType.triangle
    g = sympy_lagrange(celltype, degree)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    lagrange = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, degree,
                                    basix.LagrangeVariant.equispaced)
    pts = basix.create_lattice(celltype, 6, basix.LatticeType.equispaced, True)
    nderiv = 3
    wtab = lagrange.tabulate(nderiv, pts)

    for kx in range(nderiv):
        for ky in range(0, nderiv - kx):
            wsym = numpy.zeros_like(wtab[0])
            for i in range(len(g)):
                wd = sympy.diff(g[i], x, kx, y, ky)
                for j, p in enumerate(pts):
                    wsym[j, i] = wd.subs([(x, p[0]), (y, p[1])])

            assert numpy.allclose(wtab[basix.index(kx, ky)], wsym)


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_tet(degree):
    celltype = basix.CellType.tetrahedron
    g = sympy_lagrange(celltype, degree)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    lagrange = basix.create_element(basix.ElementFamily.P, basix.CellType.tetrahedron, degree,
                                    basix.LagrangeVariant.equispaced)
    pts = basix.create_lattice(celltype, 6,
                               basix.LatticeType.equispaced, True)
    nderiv = 1
    wtab = lagrange.tabulate(nderiv, pts)
    for k in range(nderiv + 1):
        for q in range(k + 1):
            for kx in range(q + 1):
                ky = q - kx
                kz = k - q
                wsym = numpy.zeros_like(wtab[0])
                for i in range(len(g)):
                    wd = sympy.diff(g[i], x, kx, y, ky, z, kz)
                    for j, p in enumerate(pts):
                        wsym[j, i] = wd.subs([(x, p[0]),
                                              (y, p[1]),
                                              (z, p[2])])

                assert numpy.allclose(wtab[basix.index(kx, ky, kz)], wsym)


@pytest.mark.parametrize("celltype", [(basix.CellType.interval, basix.CellType.interval),
                                      (basix.CellType.triangle, basix.CellType.triangle),
                                      (basix.CellType.tetrahedron, basix.CellType.tetrahedron)])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_lagrange(celltype, degree):
    lagrange = basix.create_element(basix.ElementFamily.P, celltype[1], degree,
                                    basix.LagrangeVariant.equispaced)
    pts = basix.create_lattice(celltype[0], 6, basix.LatticeType.equispaced, True)
    w = lagrange.tabulate(0, pts)[0]
    assert numpy.isclose(numpy.sum(w, axis=1), 1.0).all()


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_dof_transformations_interval(degree):
    lagrange = basix.create_element(basix.ElementFamily.P, basix.CellType.interval, degree,
                                    basix.LagrangeVariant.equispaced)
    assert len(lagrange.base_transformations()) == 0


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_dof_transformations_triangle(degree):
    lagrange = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, degree,
                                    basix.LagrangeVariant.equispaced)

    permuted = {}
    if degree == 3:
        # Reflect 2 DOFs on edges
        permuted[0] = {3: 4, 4: 3}
        permuted[1] = {5: 6, 6: 5}
        permuted[2] = {7: 8, 8: 7}
    elif degree == 4:
        # Reflect 3 DOFs on edges
        permuted[0] = {3: 5, 5: 3}
        permuted[1] = {6: 8, 8: 6}
        permuted[2] = {9: 11, 11: 9}

    base_transformations = lagrange.base_transformations()
    assert len(base_transformations) == 3

    for i, t in enumerate(base_transformations):
        actual = numpy.zeros_like(t)
        for j, row in enumerate(t):
            if i in permuted and j in permuted[i]:
                actual[j, permuted[i][j]] = 1
            else:
                actual[j, j] = 1
        assert numpy.allclose(t, actual)


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_dof_transformations_tetrahedron(degree):
    lagrange = basix.create_element(basix.ElementFamily.P, basix.CellType.tetrahedron, degree,
                                    basix.LagrangeVariant.equispaced)

    permuted = {}
    if degree == 3:
        # Reflect 2 DOFs on edges
        permuted[0] = {4: 5, 5: 4}
        permuted[1] = {6: 7, 7: 6}
        permuted[2] = {8: 9, 9: 8}
        permuted[3] = {10: 11, 11: 10}
        permuted[4] = {12: 13, 13: 12}
        permuted[5] = {14: 15, 15: 14}
    elif degree == 4:
        # Reflect 3 DOFs on edges
        permuted[0] = {4: 6, 6: 4}
        permuted[1] = {7: 9, 9: 7}
        permuted[2] = {10: 12, 12: 10}
        permuted[3] = {13: 15, 15: 13}
        permuted[4] = {16: 18, 18: 16}
        permuted[5] = {19: 21, 21: 19}
        # Rotate and reflect 3 DOFs on faces
        permuted[6] = {22: 24, 23: 22, 24: 23}
        permuted[7] = {23: 24, 24: 23}
        permuted[8] = {25: 27, 26: 25, 27: 26}
        permuted[9] = {26: 27, 27: 26}
        permuted[10] = {28: 30, 29: 28, 30: 29}
        permuted[11] = {29: 30, 30: 29}
        permuted[12] = {31: 33, 32: 31, 33: 32}
        permuted[13] = {32: 33, 33: 32}

    base_transformations = lagrange.base_transformations()
    assert len(base_transformations) == 14

    for i, t in enumerate(base_transformations):
        actual = numpy.zeros_like(t)
        for j, row in enumerate(t):
            if i in permuted and j in permuted[i]:
                actual[j, permuted[i][j]] = 1
            else:
                actual[j, j] = 1
        assert numpy.allclose(t, actual)


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("celltype", [
    basix.CellType.quadrilateral,
    basix.CellType.hexahedron,
    basix.CellType.pyramid,
    basix.CellType.prism
])
def test_celltypes(degree, celltype):
    tp = basix.create_element(basix.ElementFamily.P, celltype, degree,
                              basix.LagrangeVariant.equispaced)
    pts = basix.create_lattice(celltype, 5,
                               basix.LatticeType.equispaced, True)
    w = tp.tabulate(0, pts)[0]
    assert numpy.allclose(numpy.sum(w, axis=1), 1.0)


def leq(a, b):
    return a <= b or numpy.isclose(a, b)


def in_cell(celltype, p):
    if celltype == basix.CellType.interval:
        return leq(0, p[0]) and leq(p[0], 1)
    if celltype == basix.CellType.triangle:
        return leq(0, p[0]) and leq(0, p[1]) and leq(p[0] + p[1], 1)
    if celltype == basix.CellType.tetrahedron:
        return leq(0, p[0]) and leq(0, p[1]) and leq(0, p[2]) and leq(p[0] + p[1] + p[2], 1)
    if celltype == basix.CellType.quadrilateral:
        return leq(0, p[0]) and leq(0, p[1]) and leq(p[0], 1) and leq(p[1], 1)
    if celltype == basix.CellType.hexahedron:
        return leq(0, p[0]) and leq(0, p[1]) and leq(0, p[2]) and leq(p[0], 1) and leq(p[1], 1) and leq(p[2], 1)
    if celltype == basix.CellType.prism:
        return leq(0, p[0]) and leq(0, p[1]) and leq(0, p[2]) and leq(p[0] + p[1], 1) and leq(p[2], 1)


@pytest.mark.parametrize("variant", [
    basix.LagrangeVariant.equispaced,
    basix.LagrangeVariant.gll_warped,
    basix.LagrangeVariant.gll_isaac,
    basix.LagrangeVariant.gll_centroid,
    basix.LagrangeVariant.chebyshev_warped,
    basix.LagrangeVariant.chebyshev_isaac,
    basix.LagrangeVariant.chebyshev_centroid,
    basix.LagrangeVariant.gl_warped,
    basix.LagrangeVariant.gl_isaac,
    basix.LagrangeVariant.gl_centroid,
    basix.LagrangeVariant.vtk
])
@pytest.mark.parametrize("celltype", [
    basix.CellType.triangle,
    basix.CellType.tetrahedron,
    basix.CellType.quadrilateral,
    basix.CellType.hexahedron,
])
@pytest.mark.parametrize("degree", range(1, 5))
def test_variant_points(celltype, degree, variant):
    e = basix.create_element(basix.ElementFamily.P, celltype, degree, variant, True)
    for p in e.points:
        assert in_cell(celltype, p)


@pytest.mark.parametrize("variant", [
    basix.LagrangeVariant.chebyshev_warped, basix.LagrangeVariant.chebyshev_isaac,
    basix.LagrangeVariant.chebyshev_centroid,
    basix.LagrangeVariant.gl_warped, basix.LagrangeVariant.gl_isaac, basix.LagrangeVariant.gl_centroid,
])
@pytest.mark.parametrize("celltype", [
    basix.CellType.triangle, basix.CellType.tetrahedron,
    basix.CellType.quadrilateral, basix.CellType.hexahedron,
])
def test_continuous_lagrange(celltype, variant):
    # The variants used in this test can only be used for discontinuous
    # Lagrange, so trying to create them should throw a runtime error
    with pytest.raises(RuntimeError):
        basix.create_element(basix.ElementFamily.P, celltype, 4, variant, False)


@pytest.mark.parametrize("celltype", [
    basix.CellType.interval, basix.CellType.triangle, basix.CellType.tetrahedron,
    basix.CellType.quadrilateral, basix.CellType.hexahedron,
])
@pytest.mark.parametrize("degree", range(1, 9))
def test_vtk_element(celltype, degree):
    if degree > 5 and celltype == basix.CellType.hexahedron:
        pytest.skip("Skipping slow test on hexahedron")

    equi = basix.create_element(basix.ElementFamily.P, celltype, degree, basix.LagrangeVariant.equispaced, True)
    vtk = basix.create_element(basix.ElementFamily.P, celltype, degree, basix.LagrangeVariant.vtk, True)

    assert vtk.points.shape == equi.points.shape

    perm = []

    for i, p in enumerate(vtk.points):
        for j, q in enumerate(vtk.points):
            if i != j:
                assert not numpy.allclose(p, q)

        for j, q in enumerate(equi.points):
            if numpy.allclose(p, q):
                perm.append(j)
                break
        else:
            raise ValueError(f"Incorrect point in VTK variant: {p}")

    # Test against permutations that were previously in DOLFINx
    if celltype == basix.CellType.triangle:
        if degree <= 9:
            target = [0, 1, 2]
            j = 3
            target += [2 * degree + k for k in range(1, degree)]
            target += [2 + k for k in range(1, degree)]
            target += [2 * degree + 1 - k for k in range(1, degree)]

            if degree == 3:
                target += [len(target) + i for i in [0]]
            elif degree == 4:
                target += [len(target) + i for i in [0, 1, 2]]
            elif degree == 5:
                target += [len(target) + i for i in [0, 2, 5, 1, 4, 3]]
            elif degree == 6:
                target += [len(target) + i for i in [0, 3, 9, 1, 2, 6, 8, 7, 4, 5]]
            elif degree == 7:
                target += [len(target) + i for i in [0, 4, 14, 1, 2, 3, 8, 11, 13, 12, 9, 5, 6, 7, 10]]
            elif degree == 8:
                target += [len(target) + i for i in [0, 5, 20, 1, 2, 3, 4, 10, 14, 17, 19,
                                                     18, 15, 11, 6, 7, 9, 16, 8, 13, 12]]
            elif degree == 9:
                target += [len(target) + i for i in [0, 6, 27, 1, 2, 3, 4, 5, 12, 17, 21, 24, 26, 25,
                                                     22, 18, 13, 7, 8, 11, 23, 9, 10, 16, 20, 19, 14, 15]]

            assert perm == target

    elif celltype == basix.CellType.tetrahedron:
        if degree == 1:
            assert perm == [0, 1, 2, 3]
        elif degree == 2:
            assert perm == [0, 1, 2, 3, 9, 6, 8, 7, 5, 4]
        elif degree == 3:
            assert perm == [0, 1, 2, 3, 14, 15, 8, 9, 13, 12,
                            10, 11, 6, 7, 4, 5, 18, 16, 17, 19]

    elif celltype == basix.CellType.quadrilateral:
        target = [0, 1, 3, 2]
        target += [4 + k for k in range(degree - 1)]
        target += [4 + 2 * (degree - 1) + k for k in range(degree - 1)]
        target += [4 + 3 * (degree - 1) + k for k in range(degree - 1)]
        target += [4 + (degree - 1) + k for k in range(degree - 1)]
        target += [4 + (degree - 1) * 4 + k for k in range((degree - 1) ** 2)]

        assert target == perm

    elif celltype == basix.CellType.hexahedron:
        if degree == 1:
            assert perm == [0, 1, 3, 2, 4, 5, 7, 6]
        elif degree == 2:
            assert perm == [0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18,
                            19, 17, 10, 12, 15, 14, 22, 23, 21, 24, 20, 25, 26]


@pytest.mark.parametrize("variant", [
    basix.LagrangeVariant.legendre,
])
@pytest.mark.parametrize("celltype", [
    basix.CellType.interval,
    basix.CellType.triangle, basix.CellType.tetrahedron,
    basix.CellType.quadrilateral, basix.CellType.hexahedron,
])
@pytest.mark.parametrize("degree", range(1, 5))
def test_legendre(celltype, degree, variant):
    e = basix.create_element(basix.ElementFamily.P, celltype, degree, variant, True)
    for p in e.points:
        assert in_cell(celltype, p)


@pytest.mark.parametrize("variant", [
    basix.DPCVariant.simplex_equispaced,
    basix.DPCVariant.simplex_gll,
    basix.DPCVariant.horizontal_equispaced,
    basix.DPCVariant.horizontal_gll,
    basix.DPCVariant.diagonal_equispaced,
    basix.DPCVariant.diagonal_gll,
])
@pytest.mark.parametrize("celltype", [
    basix.CellType.quadrilateral, basix.CellType.hexahedron,
])
@pytest.mark.parametrize("degree", range(5))
def test_dpc(celltype, degree, variant):
    e = basix.create_element(basix.ElementFamily.DPC, celltype, degree, variant, True)
    for p in e.points:
        assert in_cell(celltype, p)


@pytest.mark.parametrize("celltype", [
    basix.CellType.interval,
    basix.CellType.triangle, basix.CellType.tetrahedron,
    basix.CellType.quadrilateral, basix.CellType.hexahedron,
])
@pytest.mark.parametrize("degree", range(1, 5))
def test_legendre_lagrange_variant(celltype, degree):
    e = basix.create_element(
        basix.ElementFamily.P, celltype, degree, basix.LagrangeVariant.legendre, True)

    # Test that the basis functions are orthogonal
    pts, wts = basix.make_quadrature(celltype, degree * 2)
    values = e.tabulate(0, pts)[0, :, :, 0].T
    for i, row_i in enumerate(values):
        for j, row_j in enumerate(values):
            integral = numpy.sum(row_i * row_j * wts)
            if i == j:
                assert numpy.isclose(integral, 1)
            else:
                assert numpy.isclose(integral, 0)

    # Test that the basis function span the correct set
    pts = basix.create_lattice(celltype, 2 * (degree + 1), basix.LatticeType.equispaced, True)
    values = e.tabulate(0, pts)[0, :, :, 0]
    i_pts = e.points
    i_mat = e.interpolation_matrix

    tdim = len(basix.topology(celltype)) - 1
    if tdim == 1:
        for px in range(degree + 1):
            evals = i_pts[:, 0] ** px
            coeffs = i_mat @ evals
            computed_values = [numpy.dot(v, coeffs) for v in values]
            actual_values = pts[:, 0] ** px
            assert numpy.allclose(computed_values, actual_values)

    elif tdim == 2:
        powers = []
        if celltype == basix.CellType.triangle:
            for px in range(degree + 1):
                for py in range(degree + 1 - px):
                    powers.append((px, py))
        if celltype == basix.CellType.quadrilateral:
            for px in range(degree + 1):
                for py in range(degree + 1):
                    powers.append((px, py))
        for px, py in powers:
            evals = i_pts[:, 0] ** px * i_pts[:, 1] ** py
            coeffs = i_mat @ evals
            computed_values = [numpy.dot(v, coeffs) for v in values]
            actual_values = pts[:, 0] ** px * pts[:, 1] ** py
            assert numpy.allclose(computed_values, actual_values)

    else:
        assert tdim == 3
        powers = []
        if celltype == basix.CellType.tetrahedron:
            for px in range(degree + 1):
                for py in range(degree + 1 - px):
                    for pz in range(degree + 1 - px - py):
                        powers.append((px, py, pz))
        if celltype == basix.CellType.hexahedron:
            for px in range(degree + 1):
                for py in range(degree + 1):
                    for pz in range(degree + 1):
                        powers.append((px, py, pz))
        for px, py, pz in powers:
            evals = i_pts[:, 0] ** px * i_pts[:, 1] ** py * i_pts[:, 2] ** pz
            coeffs = i_mat @ evals
            computed_values = [numpy.dot(v, coeffs) for v in values]
            actual_values = pts[:, 0] ** px * pts[:, 1] ** py * pts[:, 2] ** pz
            assert numpy.allclose(computed_values, actual_values)


@pytest.mark.parametrize("celltype", [
    basix.CellType.quadrilateral, basix.CellType.hexahedron,
])
@pytest.mark.parametrize("degree", range(1, 5))
def test_legendre_dpc_variant(celltype, degree):
    e = basix.create_element(
        basix.ElementFamily.DPC, celltype, degree, basix.DPCVariant.legendre, True)

    # Test that the basis functions are orthogonal
    pts, wts = basix.make_quadrature(celltype, degree * 2)
    values = e.tabulate(0, pts)[0, :, :, 0].T
    for i, row_i in enumerate(values):
        for j, row_j in enumerate(values):
            integral = numpy.sum(row_i * row_j * wts)
            if i == j:
                assert numpy.isclose(integral, 1)
            else:
                assert numpy.isclose(integral, 0)

    # Test that the basis function span the correct set
    pts = basix.create_lattice(celltype, 2 * (degree + 1), basix.LatticeType.equispaced, True)
    values = e.tabulate(0, pts)[0, :, :, 0]
    i_pts = e.points
    i_mat = e.interpolation_matrix

    tdim = len(basix.topology(celltype)) - 1
    if tdim == 2:
        for px in range(degree + 1):
            for py in range(degree + 1 - px):
                evals = i_pts[:, 0] ** px * i_pts[:, 1] ** py
                coeffs = i_mat @ evals
                computed_values = [numpy.dot(v, coeffs) for v in values]
                actual_values = pts[:, 0] ** px * pts[:, 1] ** py
                assert numpy.allclose(computed_values, actual_values)

    else:
        assert tdim == 3
        for px in range(degree + 1):
            for py in range(degree + 1 - px):
                for pz in range(degree + 1 - px - py):
                    evals = i_pts[:, 0] ** px * i_pts[:, 1] ** py * i_pts[:, 2] ** pz
                    coeffs = i_mat @ evals
                    computed_values = [numpy.dot(v, coeffs) for v in values]
                    actual_values = pts[:, 0] ** px * pts[:, 1] ** py * pts[:, 2] ** pz
                    assert numpy.allclose(computed_values, actual_values)
