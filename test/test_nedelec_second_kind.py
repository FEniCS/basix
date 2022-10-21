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
    #    e = symfem.create_element("triangle", "N2curl", 2)
    #    print(e.get_basis_functions())
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")

    if celltype == basix.CellType.triangle:
        if n == 1:
            return [[2*y, 4*x],
                    [-4*y, -2*x],
                    [-2*y, -4*x - 6*y + 4],
                    [4*y, 2*x + 6*y - 2],
                    [-6*x - 4*y + 4, -2*x],
                    [6*x + 2*y - 2, 4*x]]
        if n == 2:
            return [[12*x*y - 3*y, 18*x**2 - 9*x],
                    [-18*y**2 + 9*y, -12*x*y + 3*x],
                    [-9*x*y - 3*y**2/2 + 3*y, 3*x**2/2 + 9*x*y - 3*x],
                    [12*x*y + 12*y**2 - 9*y, 18*x**2 + 48*x*y - 27*x + 30*y**2 - 36*y + 9],
                    [18*y**2 - 9*y, 12*x*y - 3*x + 30*y**2 - 24*y + 3],
                    [-9*x*y - 15*y**2/2 + 6*y, 3*x**2/2 - 15*x*y - 15*y**2 + 15*y - 3/2],
                    [30*x**2 + 48*x*y - 36*x + 18*y**2 - 27*y + 9, 12*x**2 + 12*x*y - 9*x],
                    [30*x**2 + 12*x*y - 24*x - 3*y + 3, 18*x**2 - 9*x],
                    [-15*x**2 - 15*x*y + 15*x + 3*y**2/2 - 3/2, -15*x**2/2 - 9*x*y + 6*x],
                    [-48*x*y - 12*y**2 + 12*y, -12*x**2 - 48*x*y + 12*x],
                    [48*x*y + 36*y**2 - 36*y, 12*x**2 + 24*x*y - 12*x],
                    [-24*x*y - 12*y**2 + 12*y, -36*x**2 - 48*x*y + 36*x]]
        if n == 3:
            return [
                [60*x**2*y - 40*x*y + 4*y, 80*x**3 - 80*x**2 + 16*x],
                [-80*y**3 + 80*y**2 - 16*y, -60*x*y**2 + 40*x*y - 4*x],
                [-320*x**2*y/9 - 80*x*y**2/9 + 80*x*y/3 + 40*y**3/27 - 8*y/3, 400*x**3/27 + 640*x**2*y/9 - 80*x**2/3 + 220*x*y**2/9 - 40*x*y + 28*x/3],  # noqa: E501
                [-220*x**2*y/9 - 640*x*y**2/9 + 40*x*y - 400*y**3/27 + 80*y**2/3 - 28*y/3, -40*x**3/27 + 80*x**2*y/9 + 320*x*y**2/9 - 80*x*y/3 + 8*x/3],  # noqa: E501
                [-60*x**2*y - 120*x*y**2 + 80*x*y - 60*y**3 + 80*y**2 - 24*y, -80*x**3 - 300*x**2*y + 160*x**2 - 360*x*y**2 + 400*x*y - 96*x - 140*y**3 + 240*y**2 - 120*y + 16],  # noqa: E501
                [80*y**3 - 80*y**2 + 16*y, 60*x*y**2 - 40*x*y + 4*x + 140*y**3 - 180*y**2 + 60*y - 4],
                [320*x**2*y/9 + 560*x*y**2/9 - 400*x*y/9 + 680*y**3/27 - 320*y**2/9 + 104*y/9, -400*x**3/27 + 560*x**2*y/9 + 160*x**2/9 + 1220*x*y**2/9 - 1000*x*y/9 - 4*x/9 + 1540*y**3/27 - 860*y**2/9 + 380*y/9 - 68/27],  # noqa: E501
                [220*x**2*y/9 - 200*x*y**2/9 - 80*x*y/9 - 860*y**3/27 + 320*y**2/9 - 56*y/9, 40*x**3/27 + 340*x**2*y/9 - 40*x**2/9 - 320*x*y**2/9 - 80*x*y/9 + 16*x/9 - 1540*y**3/27 + 680*y**2/9 - 200*y/9 + 32/27],  # noqa: E501
                [-140*x**3 - 360*x**2*y + 240*x**2 - 300*x*y**2 + 400*x*y - 120*x - 80*y**3 + 160*y**2 - 96*y + 16, -60*x**3 - 120*x**2*y + 80*x**2 - 60*x*y**2 + 80*x*y - 24*x],  # noqa: E501
                [140*x**3 + 60*x**2*y - 180*x**2 - 40*x*y + 60*x + 4*y - 4, 80*x**3 - 80*x**2 + 16*x],
                [1540*x**3/27 + 1220*x**2*y/9 - 860*x**2/9 + 560*x*y**2/9 - 1000*x*y/9 + 380*x/9 - 400*y**3/27 + 160*y**2/9 - 4*y/9 - 68/27, 680*x**3/27 + 560*x**2*y/9 - 320*x**2/9 + 320*x*y**2/9 - 400*x*y/9 + 104*x/9],  # noqa: E501
                [-1540*x**3/27 - 320*x**2*y/9 + 680*x**2/9 + 340*x*y**2/9 - 80*x*y/9 - 200*x/9 + 40*y**3/27 - 40*y**2/9 + 16*y/9 + 32/27, -860*x**3/27 - 200*x**2*y/9 + 320*x**2/9 + 220*x*y**2/9 - 80*x*y/9 - 56*x/9],  # noqa: E501
                [-240*x**2*y - 180*x*y**2 + 180*x*y - 20*y**3 + 40*y**2 - 20*y, -40*x**3 - 240*x**2*y + 60*x**2 - 120*x*y**2 + 140*x*y - 20*x],  # noqa: E501
                [-120*x**2*y - 240*x*y**2 + 140*x*y - 40*y**3 + 60*y**2 - 20*y, -20*x**3 - 180*x**2*y + 40*x**2 - 240*x*y**2 + 180*x*y - 20*x],  # noqa: E501
                [-240*x**2*y - 300*x*y**2 + 300*x*y - 80*y**3 + 160*y**2 - 80*y, -40*x**3 - 120*x**2*y + 60*x**2 - 60*x*y**2 + 80*x*y - 20*x],  # noqa: E501
                [-120*x**2*y + 100*x*y + 80*y**3 - 80*y**2, -20*x**3 + 20*x**2 + 60*x*y**2 - 40*x*y],
                [60*x**2*y + 120*x*y**2 - 80*x*y + 40*y**3 - 60*y**2 + 20*y, 80*x**3 + 300*x**2*y - 160*x**2 + 240*x*y**2 - 300*x*y + 80*x],  # noqa: E501
                [-60*x**2*y + 40*x*y + 20*y**3 - 20*y**2, -80*x**3 + 80*x**2 + 120*x*y**2 - 100*x*y],
                [-15*x**2*y - 30*x*y**2 + 20*x*y - 5*y**2 + 5*y, 15*x**3 + 30*x**2*y - 20*x**2 - 10*x*y + 5*x],  # noqa: E501
                [30*x*y**2 - 10*x*y + 15*y**3 - 20*y**2 + 5*y, -30*x**2*y - 5*x**2 - 15*x*y**2 + 20*x*y + 5*x]  # noqa: E501
            ]
    if celltype == basix.CellType.tetrahedron:
        if n == 1:
            return [[0, 2*z, 4*y],
                    [0, -4*z, -2*y],
                    [2*z, 0, 4*x],
                    [-4*z, 0, -2*x],
                    [2*y, 4*x, 0],
                    [-4*y, -2*x, 0],
                    [-2*z, -2*z, -4*x - 4*y - 6*z + 4],
                    [4*z, 4*z, 2*x + 2*y + 6*z - 2],
                    [-2*y, -4*x - 6*y - 4*z + 4, -2*y],
                    [4*y, 2*x + 6*y + 2*z - 2, 4*y],
                    [-6*x - 4*y - 4*z + 4, -2*x, -2*x],
                    [6*x + 2*y + 2*z - 2, 4*x, 4*x]]
        if n == 2:
            return [
                [0, 12*y*z - 3*z, 18*y**2 - 9*y],
                [0, -18*z**2 + 9*z, -12*y*z + 3*y],
                [0, -9*y*z - 3*z**2/2 + 3*z, 3*y**2/2 + 9*y*z - 3*y],
                [12*x*z - 3*z, 0, 18*x**2 - 9*x],
                [-18*z**2 + 9*z, 0, -12*x*z + 3*x],
                [-9*x*z - 3*z**2/2 + 3*z, 0, 3*x**2/2 + 9*x*z - 3*x],
                [12*x*y - 3*y, 18*x**2 - 9*x, 0],
                [-18*y**2 + 9*y, -12*x*y + 3*x, 0],
                [-9*x*y - 3*y**2/2 + 3*y, 3*x**2/2 + 9*x*y - 3*x, 0],
                [12*x*z + 12*y*z + 12*z**2 - 9*z, 12*x*z + 12*y*z + 12*z**2 - 9*z, 18*x**2 + 36*x*y + 48*x*z - 27*x + 18*y**2 + 48*y*z - 27*y + 30*z**2 - 36*z + 9],  # noqa: E501
                [18*z**2 - 9*z, 18*z**2 - 9*z, 12*x*z - 3*x + 12*y*z - 3*y + 30*z**2 - 24*z + 3],
                [-9*x*z - 9*y*z - 15*z**2/2 + 6*z, -9*x*z - 9*y*z - 15*z**2/2 + 6*z, 3*x**2/2 + 3*x*y - 15*x*z + 3*y**2/2 - 15*y*z - 15*z**2 + 15*z - 3/2],  # noqa: E501
                [12*x*y + 12*y**2 + 12*y*z - 9*y, 18*x**2 + 48*x*y + 36*x*z - 27*x + 30*y**2 + 48*y*z - 36*y + 18*z**2 - 27*z + 9, 12*x*y + 12*y**2 + 12*y*z - 9*y],  # noqa: E501
                [18*y**2 - 9*y, 12*x*y - 3*x + 30*y**2 + 12*y*z - 24*y - 3*z + 3, 18*y**2 - 9*y],
                [-9*x*y - 15*y**2/2 - 9*y*z + 6*y, 3*x**2/2 - 15*x*y + 3*x*z - 15*y**2 - 15*y*z + 15*y + 3*z**2/2 - 3/2, -9*x*y - 15*y**2/2 - 9*y*z + 6*y],  # noqa: E501
                [30*x**2 + 48*x*y + 48*x*z - 36*x + 18*y**2 + 36*y*z - 27*y + 18*z**2 - 27*z + 9, 12*x**2 + 12*x*y + 12*x*z - 9*x, 12*x**2 + 12*x*y + 12*x*z - 9*x],  # noqa: E501
                [30*x**2 + 12*x*y + 12*x*z - 24*x - 3*y - 3*z + 3, 18*x**2 - 9*x, 18*x**2 - 9*x],
                [-15*x**2 - 15*x*y - 15*x*z + 15*x + 3*y**2/2 + 3*y*z + 3*z**2/2 - 3/2, -15*x**2/2 - 9*x*y - 9*x*z + 6*x, -15*x**2/2 - 9*x*y - 9*x*z + 6*x],  # noqa: E501
                [36*y*z, 12*x*z, 12*x*y],
                [-12*y*z, -36*x*z, -12*x*y],
                [12*y*z, 12*x*z, 36*x*y],
                [-36*y*z, -12*x*z - 48*y*z - 12*z**2 + 12*z, -12*x*y - 12*y**2 - 48*y*z + 12*y],
                [12*y*z, 36*x*z + 48*y*z + 36*z**2 - 36*z, 12*x*y + 12*y**2 + 24*y*z - 12*y],
                [-12*y*z, -12*x*z - 24*y*z - 12*z**2 + 12*z, -36*x*y - 36*y**2 - 48*y*z + 36*y],
                [-48*x*z - 12*y*z - 12*z**2 + 12*z, -36*x*z, -12*x**2 - 12*x*y - 48*x*z + 12*x],
                [48*x*z + 36*y*z + 36*z**2 - 36*z, 12*x*z, 12*x**2 + 12*x*y + 24*x*z - 12*x],
                [-24*x*z - 12*y*z - 12*z**2 + 12*z, -12*x*z, -36*x**2 - 36*x*y - 48*x*z + 36*x],
                [-48*x*y - 12*y**2 - 12*y*z + 12*y, -12*x**2 - 48*x*y - 12*x*z + 12*x, -36*x*y],
                [48*x*y + 36*y**2 + 36*y*z - 36*y, 12*x**2 + 24*x*y + 12*x*z - 12*x, 12*x*y],
                [-24*x*y - 12*y**2 - 12*y*z + 12*y, -36*x**2 - 48*x*y - 36*x*z + 36*x, -12*x*y]
            ]

    raise NotImplementedError


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_tri(degree):
    celltype = basix.CellType.triangle
    g = sympy_nedelec(celltype, degree)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    nedelec = basix.create_element(
        basix.ElementFamily.N2E, basix.CellType.triangle, degree, basix.LagrangeVariant.equispaced)
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


@pytest.mark.parametrize("degree", [1, 2])
def test_tet(degree):
    celltype = basix.CellType.tetrahedron
    g = sympy_nedelec(celltype, degree)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    nedelec = basix.create_element(
        basix.ElementFamily.N2E, basix.CellType.tetrahedron, degree, basix.LagrangeVariant.equispaced)
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
