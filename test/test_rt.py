# Copyright (c) 2020 Chris Richardson & Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import numpy
import pytest
import sympy
from .test_lagrange import sympy_lagrange


def sympy_rt(celltype, n):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")

    from sympy import S
    topology = basix.topology(celltype)
    geometry = S(basix.geometry(celltype).astype(int))
    dummy = [sympy.Symbol("DUMMY1"), sympy.Symbol("DUMMY2"), sympy.Symbol("DUMMY3")]

    funcs = []
    if celltype == basix.CellType.triangle:
        tdim = 2
        for i in range(n):
            for j in range(n - i):
                for d in range(2):
                    funcs += [[x**j * y**i if k == d else 0 for k in range(2)]]
        for i in range(n):
            funcs.append([x ** (n - i) * y ** i, x ** (n - 1 - i) * y ** (i + 1)])

        mat = numpy.empty((len(funcs), len(funcs)), dtype=object)

        # edge normals
        for i, f in enumerate(funcs):
            if n == 1:
                edge_basis = [sympy.Integer(1)]
            else:
                edge_basis = sympy_lagrange(basix.CellType.interval, n - 1)
            edge_basis = [a.subs(x, dummy[0]) for a in edge_basis]
            j = 0
            for edge in topology[1]:
                edge_geom = [geometry[t, :] for t in edge]
                tangent = edge_geom[1] - edge_geom[0]
                norm = sympy.sqrt(sum(i ** 2 for i in tangent))
                tangent = [i / norm for i in tangent]
                normal = [-tangent[1], tangent[0]]
                param = [(1 - dummy[0]) * a + dummy[0] * b for a, b in zip(edge_geom[0], edge_geom[1])]

                for g in edge_basis:
                    integrand = sum((f_i * v_i) for f_i, v_i in zip(f, normal))

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
                    face_basis = sympy_lagrange(basix.CellType.triangle, n - 2)

                j = n * 3
                for g in face_basis:
                    for vec in [(1, 0), (0, 1)]:
                        integrand = sum((f_i * v_i) for f_i, v_i in zip(f, vec)) * g

                        mat[i, j] = integrand.integrate((x, 0, 1 - y)).integrate((y, 0, 1))
                        j += 1

    elif celltype == basix.CellType.tetrahedron:
        tdim = 3
        for i in range(n):
            for j in range(n - i):
                for k in range(n - i - j):
                    for d in range(3):
                        funcs += [[x**k * y**j * z**i if m == d else 0 for m in range(3)]]

        for j in range(n):
            for k in range(n - j):
                p = x ** (n - 1 - j - k) * y ** j * z ** k
                funcs.append((x * p, y * p, z * p))

        mat = numpy.empty((len(funcs), len(funcs)), dtype=object)

        # face normals
        for i, f in enumerate(funcs):
            if n == 1:
                face_basis = [sympy.Integer(1)]
            else:
                face_basis = sympy_lagrange(basix.CellType.triangle, n - 1)
            face_basis = [a.subs(x, dummy[0]).subs(y, dummy[1]) for a in face_basis]
            j = 0
            for face in topology[2]:
                face_geom = [geometry[t, :] for t in face]
                axes = [face_geom[1] - face_geom[0], face_geom[2] - face_geom[0]]
                normal = [axes[0][1] * axes[1][2] - axes[0][2] * axes[1][1],
                          axes[0][2] * axes[1][0] - axes[0][0] * axes[1][2],
                          axes[0][0] * axes[1][1] - axes[0][1] * axes[1][0]]
                norm = sympy.sqrt(sum(i**2 for i in normal))
                normal = [k / norm for k in normal]

                param = [a + dummy[0] * b + dummy[1] * c for a, b, c in zip(face_geom[0], *axes)]
                for g in face_basis:
                    integrand = sum(f_i * v_i for f_i, v_i in zip(f, normal))
                    integrand = integrand.subs(x, param[0]).subs(y, param[1]).subs(z, param[2])
                    integrand *= g * norm

                    mat[i, j] = integrand.integrate((dummy[0], 0, 1 - dummy[1])).integrate((dummy[1], 0, 1))
                    j += 1
            assert j == 2 * n * (n + 1)
        if n > 1:
            for i, f in enumerate(funcs):
                if n == 2:
                    interior_basis = [sympy.Integer(1)]
                else:
                    interior_basis = sympy_lagrange(basix.CellType.tetrahedron, n - 2)
                j = 2 * n * (n + 1)
                for g in interior_basis:
                    for vec in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                        integrand = sum(f_i * v_i for f_i, v_i in zip(f, vec))
                        integrand *= g

                        mat[i, j] = integrand.integrate((x, 0, 1 - y - z)).integrate((y, 0, 1 - z)).integrate((z, 0, 1))
                        j += 1

    mat = sympy.Matrix(mat)
    mat = mat.inv()
    g = []
    for r in range(mat.shape[0]):
        row = []
        for dim in range(tdim):
            row.append(sum([v * funcs[i][dim] for i, v in enumerate(mat.row(r))]))
        g.append(row)
    return g


# TODO: turn this test back on when a equispaced Nedelec variant is available
@pytest.mark.parametrize("degree", [1, 2, 3])
def xtest_tri(degree):
    celltype = basix.CellType.triangle
    g = sympy_rt(celltype, degree)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    rt = basix.create_element(basix.ElementFamily.RT, basix.CellType.triangle, degree, basix.LagrangeVariant.equispaced)
    pts = basix.create_lattice(celltype, 1, basix.LatticeType.equispaced, True)
    nderiv = 3
    wtab = rt.tabulate(nderiv, pts)

    for kx in range(nderiv + 1):
        for ky in range(nderiv + 1 - kx):
            wsym = numpy.zeros_like(wtab[0])
            for i, gi in enumerate(g):
                for j, gij in enumerate(gi):
                    wd = sympy.diff(gij, x, kx, y, ky)
                    for k, p in enumerate(pts):
                        wsym[k, i, j] = wd.subs([(x, p[0]), (y, p[1])])

            assert(numpy.isclose(wtab[basix.index(kx, ky)], wsym).all())


# TODO: turn this test back on when a equispaced Nedelec variant is available
@pytest.mark.parametrize("degree", [1, 2, 3])
def xtest_tet(degree):
    celltype = basix.CellType.tetrahedron
    g = sympy_rt(celltype, degree)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    rt = basix.create_element(
        basix.ElementFamily.RT, basix.CellType.tetrahedron, degree, basix.LagrangeVariant.equispaced)

    pts = basix.create_lattice(celltype, 5, basix.LatticeType.equispaced, True)
    nderiv = 1
    wtab = rt.tabulate(nderiv, pts)

    for kx in range(nderiv + 1):
        for ky in range(nderiv + 1 - kx):
            for kz in range(nderiv + 1 - kx - ky):
                wsym = numpy.zeros_like(wtab[0])
                for i, gi in enumerate(g):
                    for j, gij in enumerate(gi):
                        wd = sympy.diff(gij, x, kx, y, ky, z, kz)
                        for k, p in enumerate(pts):
                            wsym[k, i, j] = wd.subs([(x, p[0]), (y, p[1]), (z, p[2])])

                assert(numpy.isclose(wtab[basix.index(kx, ky, kz)], wsym).all())
