# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import libtab
import numpy
import pytest
import sympy
from .test_lagrange import sympy_lagrange


def sympy_nedelec(celltype, n):
    if celltype != libtab.CellType.triangle:
        raise NotImplementedError

    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    # z = sympy.Symbol("z")

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

    dummy = [sympy.Symbol("DUMMY1"), sympy.Symbol("DUMMY2"), sympy.Symbol("DUMMY3")]

    funcs = []
    if celltype == libtab.CellType.triangle:
        for i in range(n):
            for j in range(n - i):
                for d in range(2):
                    funcs += [[x**j * y**i if k == d else 0 for k in range(2)]]
        for i in range(n):
            funcs += [[x ** (n - 1 - i) * y ** (i + 1),
                       -x ** (n - i) * y ** i]]
        print(funcs)
        mat = numpy.empty((len(funcs), len(funcs)), dtype=object)

        # edge tangents
        for i, f in enumerate(funcs):
            if n == 1:
                edge_basis = [sympy.Integer(1)]
            else:
                edge_basis = sympy_lagrange(libtab.CellType.interval, n - 1)
            edge_basis = [a.subs(x, dummy[0]).subs(y, dummy[1]) for a in edge_basis]
            j = 0
            for edge in topology[1]:
                edge_geom = [geometry[t, :] for t in edge]
                tangent = edge_geom[1] - edge_geom[0]
                norm = sympy.sqrt(sum(i ** 2 for i in tangent))
                tangent = [i / norm for i in tangent]
                param = [(1 - dummy[0]) * a + dummy[0] * b for a, b in zip(edge_geom[0], edge_geom[1])]

                for g in edge_basis:
                    integrand = sum((f_i * v_i) for f_i, v_i in zip(f, tangent))

                    integrand = integrand.subs(x, param[0]).subs(y, param[1])

                    integrand *= g * norm

                    mat[i, j] = integrand.integrate((dummy[0], 0, 1))
                    j += 1

        # interior dofs
        if n > 1:
            for i, f in enumerate(funcs):
                if n == 2:
                    face_basis = [sympy.Integer(1)]
                else:
                    face_basis = sympy_lagrange(libtab.CellType.triangle, n - 2)

                j = n * 3
                for g in face_basis:
                    for vec in [(1, 0), (0, 1)]:
                        integrand = sum((f_i * v_i) for f_i, v_i in zip(f, vec)) * g

                        mat[i, j] = integrand.integrate((x, 0, 1 - y)).integrate((y, 0, 1))
                        j += 1

    print(mat)
    mat = sympy.Matrix(mat)
    mat = mat.inv()
    g = []
    for dim in range(2):
        for r in range(mat.shape[0]):
            g += [sum([v * funcs[i][dim] for i, v in enumerate(mat.row(r))])]

    return g


@pytest.mark.parametrize("order", [1, 2, 3])
def test_tri(order):
    celltype = libtab.CellType.triangle
    g = sympy_nedelec(celltype, order)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    nedelec = libtab.Nedelec(celltype, order)
    pts = libtab.create_lattice(celltype, 6, True)
    nderiv = 3
    wtab = nedelec.tabulate(nderiv, pts)

    for kx in range(nderiv):
        for ky in range(0, nderiv - kx):
            wsym = numpy.zeros_like(wtab[0])
            for i in range(len(g)):
                wd = sympy.diff(g[i], x, kx, y, ky)
                for j, p in enumerate(pts):
                    wsym[j, i] = wd.subs([(x, p[0]), (y, p[1])])

            assert(numpy.isclose(wtab[libtab.index(kx, ky)], wsym).all())


@pytest.mark.parametrize("order", [1, 2])
def xtest_tet(order):
    celltype = libtab.CellType.tetrahedron
    g = sympy_nedelec(celltype, order)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    nedelec = libtab.Nedelec(celltype, order)
    pts = libtab.create_lattice(celltype, 6, True)
    nderiv = 1
    wtab = nedelec.tabulate(nderiv, pts)

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

                assert(numpy.isclose(wtab[libtab.index(kx, ky, kz)], wsym).all())
