# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import numpy
import pytest
import sympy


def sympy_disc_lagrange(celltype, n):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")

    topology = basix.topology(celltype)
    tdim = len(topology) - 1
    pt = []
    if tdim == 1:
        for i in range(n + 1):
            pt.append([sympy.Rational(i, n), sympy.Integer(0), sympy.Integer(0)])
    elif tdim == 2:
        for j in range(n + 1):
            for i in range(n + 1 - j):
                pt.append([sympy.Rational(i, n), sympy.Rational(j, n), sympy.Integer(0)])
    elif tdim == 3:
        for k in range(n + 1):
            for j in range(n + 1 - k):
                for i in range(n + 1 - k - j):
                    pt.append([sympy.Rational(i, n), sympy.Rational(j, n), sympy.Rational(k, n)])

    funcs = []
    if celltype == basix.CellType.interval:
        for i in range(n + 1):
            funcs += [x**i]
        mat = numpy.empty((len(pt), len(funcs)), dtype=object)

        for i, f in enumerate(funcs):
            for j, p in enumerate(pt):
                mat[i, j] = f.subs([(x, p[0])])
    elif celltype == basix.CellType.triangle:
        for i in range(n + 1):
            for j in range(n + 1 - i):
                funcs += [x**j * y**i]
        mat = numpy.empty((len(pt), len(funcs)), dtype=object)

        for i, f in enumerate(funcs):
            for j, p in enumerate(pt):
                mat[i, j] = f.subs([(x, p[0]), (y, p[1])])
    elif celltype == basix.CellType.tetrahedron:
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


def sympy_lagrange(celltype, n):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")

    from sympy import S
    topology = basix.topology(celltype)
    geometry = S(basix.geometry(celltype).astype(int))
    pt = []
    for dim, entities in enumerate(topology):
        for ent in entities:
            entity_geom = [geometry[t, :] for t in ent]

            if (dim == 0):
                pt += [entity_geom[0]]
            elif (dim == 1):
                for i in range(n - 1):
                    pt += [entity_geom[0]
                           + sympy.Rational(i + 1, n) * (entity_geom[1] - entity_geom[0])]
            elif (dim == 2):
                for i in range(n - 2):
                    for j in range(n - 2 - i):
                        pt += [entity_geom[0]
                               + sympy.Rational(i + 1, n) * (entity_geom[2] - entity_geom[0])
                               + sympy.Rational(j + 1, n) * (entity_geom[1] - entity_geom[0])]
            elif (dim == 3):
                for i in range(n - 3):
                    for j in range(n - 3 - i):
                        for k in range(n - 3 - i - j):
                            pt += [entity_geom[0]
                                   + sympy.Rational(i + 1, n) * (entity_geom[3] - entity_geom[0])
                                   + sympy.Rational(j + 1, n) * (entity_geom[2] - entity_geom[0])
                                   + sympy.Rational(k + 1, n) * (entity_geom[1] - entity_geom[0])]

    funcs = []
    if celltype == basix.CellType.interval:
        for i in range(n + 1):
            funcs += [x**i]

        mat = numpy.empty((len(pt), len(funcs)), dtype=object)
        for i, f in enumerate(funcs):
            for j, p in enumerate(pt):
                mat[i, j] = f.subs([(x, p[0])])
    elif celltype == basix.CellType.triangle:
        for i in range(n + 1):
            for j in range(n + 1 - i):
                funcs += [x**j * y**i]

        mat = numpy.empty((len(pt), len(funcs)), dtype=object)
        for i, f in enumerate(funcs):
            for j, p in enumerate(pt):
                mat[i, j] = f.subs([(x, p[0]), (y, p[1])])
    elif celltype == basix.CellType.tetrahedron:
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
    celltype = basix.CellType.interval
    g = sympy_lagrange(celltype, n)
    x = sympy.Symbol("x")
    lagrange = basix.create_element("Lagrange", "interval", n)
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


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_tri(order):
    celltype = basix.CellType.triangle
    g = sympy_lagrange(celltype, order)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    lagrange = basix.create_element("Lagrange", "triangle", order)
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


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_tet(order):
    celltype = basix.CellType.tetrahedron
    g = sympy_lagrange(celltype, order)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    lagrange = basix.create_element("Lagrange", "tetrahedron", order)
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


@pytest.mark.parametrize("celltype", [(basix.CellType.interval, "interval"),
                                      (basix.CellType.triangle, "triangle"),
                                      (basix.CellType.tetrahedron, "tetrahedron")])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_lagrange(celltype, order):
    lagrange = basix.create_element("Lagrange", celltype[1], order)
    pts = basix.create_lattice(celltype[0], 6, basix.LatticeType.equispaced, True)
    w = lagrange.tabulate(0, pts)[0]
    assert(numpy.isclose(numpy.sum(w, axis=1), 1.0).all())


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_dof_permutations_interval(order):
    lagrange = basix.create_element("Lagrange", "interval", order)
    assert len(lagrange.base_permutations) == 0


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_dof_permutations_triangle(order):
    lagrange = basix.create_element("Lagrange", "triangle", order)

    permuted = {}
    if order == 3:
        # Reflect 2 DOFs on edges
        permuted[0] = {3: 4, 4: 3}
        permuted[1] = {5: 6, 6: 5}
        permuted[2] = {7: 8, 8: 7}
    elif order == 4:
        # Reflect 3 DOFs on edges
        permuted[0] = {3: 5, 5: 3}
        permuted[1] = {6: 8, 8: 6}
        permuted[2] = {9: 11, 11: 9}

    base_perms = lagrange.base_permutations
    assert len(base_perms) == 3

    for i, perm in enumerate(base_perms):
        actual = numpy.zeros_like(perm)
        for j, row in enumerate(perm):
            if i in permuted and j in permuted[i]:
                actual[j, permuted[i][j]] = 1
            else:
                actual[j, j] = 1
        assert numpy.allclose(perm, actual)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_dof_permutations_tetrahedron(order):
    lagrange = basix.create_element("Lagrange", "tetrahedron", order)

    permuted = {}
    if order == 3:
        # Reflect 2 DOFs on edges
        permuted[0] = {4: 5, 5: 4}
        permuted[1] = {6: 7, 7: 6}
        permuted[2] = {8: 9, 9: 8}
        permuted[3] = {10: 11, 11: 10}
        permuted[4] = {12: 13, 13: 12}
        permuted[5] = {14: 15, 15: 14}
    elif order == 4:
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

    base_perms = lagrange.base_permutations
    assert len(base_perms) == 14

    for i, perm in enumerate(base_perms):
        actual = numpy.zeros_like(perm)
        for j, row in enumerate(perm):
            if i in permuted and j in permuted[i]:
                actual[j, permuted[i][j]] = 1
            else:
                actual[j, j] = 1
        assert numpy.allclose(perm, actual)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("celltype", [
    (basix.CellType.quadrilateral, "quadrilateral"),
    (basix.CellType.hexahedron, "hexahedron"),
    (basix.CellType.pyramid, "pyramid"),
    (basix.CellType.prism, "prism")
])
def test_celltypes(order, celltype):
    tp = basix.create_element("Lagrange", celltype[1], order)
    pts = basix.create_lattice(celltype[0], 5,
                               basix.LatticeType.equispaced, True)
    w = tp.tabulate(0, pts)[0]
    assert(numpy.allclose(numpy.sum(w, axis=1), 1.0))
