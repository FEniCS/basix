# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import libtab
import numpy
import pytest
import sympy


def sympy_lagrange(celltype, n):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")

    from sympy import S
    topology = libtab.topology(celltype)
    geometry = S(libtab.geometry(celltype).astype(int))
    pt = []
    for dim, entities in enumerate(topology):
        for ent in entities:
            entity_geom = [geometry[t, :] for t in ent]

            if (dim == 0):
                pt += [entity_geom[0]]
            elif (dim == 1):
                for j in range(n - 1):
                    pt += [entity_geom[0]
                           + sympy.Rational(j + 1, n) * (entity_geom[1] - entity_geom[0])]
            elif (dim == 2):
                for j in range(n - 2):
                    for k in range(n - 2 - j):
                        pt += [entity_geom[0]
                               + sympy.Rational(j + 1, n) * (entity_geom[2] - entity_geom[0])
                               + sympy.Rational(k + 1, n) * (entity_geom[1] - entity_geom[0])]
            elif (dim == 3):
                for j in range(n - 3):
                    for k in range(n - 3 - j):
                        for l in range(n - 3 - j - k):
                            pt += [entity_geom[0]
                                   + sympy.Rational(j + 1, n) * (entity_geom[3] - entity_geom[0])
                                   + sympy.Rational(k + 1, n) * (entity_geom[2] - entity_geom[0])
                                   + sympy.Rational(l + 1, n) * (entity_geom[1] - entity_geom[0])]

    funcs = []
    if celltype == libtab.CellType.interval:
        for i in range(n + 1):
            funcs += [x**i]
        mat = numpy.empty((len(pt), len(funcs)), dtype=object)

        for i, f in enumerate(funcs):
            for j, p in enumerate(pt):
                mat[i, j] = f.subs([(x, p[0])])
    elif celltype == libtab.CellType.triangle:
        for i in range(n + 1):
            for j in range(n + 1 - i):
                funcs += [x**j * y**i]
        mat = numpy.empty((len(pt), len(funcs)), dtype=object)

        for i, f in enumerate(funcs):
            for j, p in enumerate(pt):
                mat[i, j] = f.subs([(x, p[0]), (y, p[1])])
    elif celltype == libtab.CellType.tetrahedron:
        for i in range(n + 1):
            for j in range(n + 1 - i):
                for k in range(n + 1 - i - j):
                    funcs += [x**j * y**i * z**k]
        mat = numpy.empty((len(pt), len(funcs)), dtype=object)

        for i, f in enumerate(funcs):
            for j, p in enumerate(pt):
                mat[i, j] = f.subs([(x, p[0]), (y, p[1]), (z, p[2])])

    mat = sympy.Matrix(mat)
    mat = mat.inv()
    g = []
    for r in range(mat.shape[0]):
        g += [sum([v * funcs[i] for i, v in enumerate(mat.row(r))])]

    return g


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_line(n):
    celltype = libtab.CellType.interval
    g = sympy_lagrange(celltype, n)
    x = sympy.Symbol("x")
    lagrange = libtab.Lagrange(celltype, n)
    pts = libtab.create_lattice(celltype, 6, True)
    nderiv = n
    wtab = lagrange.tabulate(nderiv, pts)

    for k in range(nderiv + 1):
        wsym = numpy.zeros_like(wtab[k])
        for i in range(n + 1):
            wd = sympy.diff(g[i], x, k)
            for j, p in enumerate(pts):
                wsym[j, i] = wd.subs(x, p[0])

        assert(numpy.isclose(wtab[k], wsym).all())


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_tri(order):
    celltype = libtab.CellType.triangle
    g = sympy_lagrange(celltype, order)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    lagrange = libtab.Lagrange(celltype, order)
    pts = libtab.create_lattice(celltype, 6, True)
    nderiv = 3
    wtab = lagrange.tabulate(nderiv, pts)

    for kx in range(nderiv):
        for ky in range(0, nderiv - kx):
            wsym = numpy.zeros_like(wtab[0])
            for i in range(len(g)):
                wd = sympy.diff(g[i], x, kx, y, ky)
                for j, p in enumerate(pts):
                    wsym[j, i] = wd.subs([(x, p[0]), (y, p[1])])

            assert(numpy.isclose(wtab[libtab.index(kx, ky)], wsym).all())


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_tet(order):
    celltype = libtab.CellType.tetrahedron
    g = sympy_lagrange(celltype, order)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    lagrange = libtab.Lagrange(celltype, order)
    pts = libtab.create_lattice(celltype, 6, True)
    nderiv = 1
    wtab = lagrange.tabulate(nderiv, pts)

    for k in range(nderiv + 1):
        for q in range(k + 1):
            for kx in range(q + 1):
                ky = q - kx
                kz = k - q
                print((kx, ky, kz))

                wsym = numpy.zeros_like(wtab[0])
                for i in range(len(g)):
                    wd = sympy.diff(g[i], x, kx, y, ky, z, kz)
                    for j, p in enumerate(pts):
                        wsym[j, i] = wd.subs([(x, p[0]),
                                              (y, p[1]),
                                              (z, p[2])])

                assert(numpy.isclose(wtab[libtab.index(kx, ky, kz)], wsym).all())


@pytest.mark.parametrize("celltype", [libtab.CellType.interval,
                                      libtab.CellType.triangle,
                                      libtab.CellType.tetrahedron])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_lagrange(celltype, order):
    lagrange = libtab.Lagrange(celltype, order)

    pts = libtab.create_lattice(celltype, 6, True)
    w = lagrange.tabulate(0, pts)[0]
    assert(numpy.isclose(numpy.sum(w, axis=1), 1.0).all())
