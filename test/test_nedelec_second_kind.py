# Copyright (c) 2020 Chris Richardson & Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import numpy
import pytest
import sympy
from .test_lagrange import sympy_disc_lagrange
from .test_rt import sympy_rt


def sympy_nedelec(celltype, n):
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
        for i in range(n + 1):
            for j in range(n + 1 - i):
                for d in range(2):
                    funcs += [[x**j * y**i if k == d else 0 for k in range(2)]]
        mat = numpy.empty((len(funcs), len(funcs)), dtype=object)

        # edge tangents
        edge_basis = sympy_disc_lagrange(basix.CellType.interval, n)
        edge_basis = [a.subs(x, dummy[0]) for a in edge_basis]
        for i, f in enumerate(funcs):
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
            face_basis = sympy_rt(basix.CellType.triangle, n - 1)
            face_basis = list(zip(face_basis[:len(face_basis) // 2], face_basis[len(face_basis) // 2:]))
            for i, f in enumerate(funcs):
                j = (n + 1) * 3
                for g in face_basis:
                    integrand = sum((f_i * v_i) for f_i, v_i in zip(f, g))

                    mat[i, j] = integrand.integrate((x, 0, 1 - y)).integrate((y, 0, 1))
                    j += 1

    elif celltype == basix.CellType.tetrahedron:
        tdim = 3
        for i in range(n + 1):
            for j in range(n + 1 - i):
                for k in range(n + 1 - i - j):
                    for d in range(3):
                        funcs += [[x**k * y**j * z**i if m == d else 0 for m in range(3)]]

        mat = numpy.empty((len(funcs), len(funcs)), dtype=object)

        # edge tangents
        edge_basis = sympy_disc_lagrange(basix.CellType.interval, n)
        edge_basis = [a.subs(x, dummy[0]) for a in edge_basis]
        for i, f in enumerate(funcs):
            j = 0
            for edge in topology[1]:
                edge_geom = [geometry[t, :] for t in edge]
                tangent = edge_geom[1] - edge_geom[0]
                norm = sympy.sqrt(sum(i ** 2 for i in tangent))
                tangent = [i / norm for i in tangent]
                param = [(1 - dummy[0]) * a + dummy[0] * b for a, b in zip(edge_geom[0], edge_geom[1])]

                for g in edge_basis:
                    integrand = sum((f_i * v_i) for f_i, v_i in zip(f, tangent))
                    integrand = integrand.subs(x, param[0]).subs(y, param[1]).subs(z, param[2])
                    integrand *= g * norm
                    mat[i, j] = integrand.integrate((dummy[0], 0, 1))
                    j += 1

        # face dofs
        if n > 1:
            face_basis = sympy_rt(basix.CellType.triangle, n - 1)
            face_basis = [a.subs(x, dummy[0]).subs(y, dummy[1]) for a in face_basis]
            face_basis = list(zip(face_basis[:len(face_basis) // 2],
                                  face_basis[len(face_basis) // 2:]))
            for i, f in enumerate(funcs):
                j = (n + 1) * 6
                for face in topology[2]:
                    face_geom = [geometry[t, :] for t in face]
                    axes = [face_geom[1] - face_geom[0], face_geom[2] - face_geom[0]]
                    norm = sympy.sqrt(sum(i**2 for i in
                                      [axes[0][1] * axes[1][2] - axes[0][2] * axes[1][1],
                                       axes[0][2] * axes[1][0] - axes[0][0] * axes[1][2],
                                       axes[0][0] * axes[1][1] - axes[0][1] * axes[1][0]]))
                    scaled_axes = []
                    for a in axes:
                        axisnorm = sympy.sqrt(sum(k**2 for k in a))
                        scaled_axes.append([k / axisnorm for k in a])
                    param = [a + dummy[0] * b + dummy[1] * c for a, b, c in zip(face_geom[0], *axes)]
                    this_face_basis = [[a[0] * b + a[1] * c for b, c in zip(*scaled_axes)] for a in face_basis]

                    for g in this_face_basis:
                        integrand = sum(f_i * v_i for f_i, v_i in zip(f, g))
                        integrand = integrand.subs(x, param[0]).subs(y, param[1]).subs(z, param[2])
                        integrand *= norm

                        mat[i, j] = integrand.integrate((dummy[0], 0, 1 - dummy[1])).integrate((dummy[1], 0, 1))
                        j += 1

        # interior dofs
        if n > 2:
            interior_basis = sympy_rt(basix.CellType.tetrahedron, n - 2)
            interior_basis = list(zip(interior_basis[:len(interior_basis) // 3],
                                      interior_basis[len(interior_basis) // 3: 2 * len(interior_basis) // 3],
                                      interior_basis[2 * len(interior_basis) // 3:]))

            for i, f in enumerate(funcs):
                j = (n + 1) * 6 + 4 * (n - 1) * (n + 1)
                for g in interior_basis:
                    integrand = sum(f_i * v_i for f_i, v_i in zip(f, g))

                    mat[i, j] = integrand.integrate((x, 0, 1 - y - z)).integrate((y, 0, 1 - z)).integrate((z, 0, 1))
                    j += 1

    mat = sympy.Matrix(mat)
    mat = mat.inv()
    g = []
    for dim in range(tdim):
        for r in range(mat.shape[0]):
            g += [sum([v * funcs[i][dim] for i, v in enumerate(mat.row(r))])]

    return g


@pytest.mark.parametrize("order", [1, 2, 3])
def test_tri(order):
    celltype = basix.CellType.triangle
    g = sympy_nedelec(celltype, order)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    nedelec = basix.create_element("Nedelec 2nd kind H(curl)", "triangle", order)
    pts = basix.create_lattice(celltype, 6, basix.LatticeType.equispaced, True)
    nderiv = 3
    wtab = nedelec.tabulate(nderiv, pts)

    for kx in range(nderiv):
        for ky in range(0, nderiv - kx):
            wsym = numpy.zeros_like(wtab[0])
            for i in range(len(g)):
                wd = sympy.diff(g[i], x, kx, y, ky)
                for j, p in enumerate(pts):
                    wsym[j, i] = wd.subs([(x, p[0]), (y, p[1])])

            assert(numpy.isclose(wtab[basix.index(kx, ky)], wsym).all())


@pytest.mark.parametrize("order", [1, 2])
def test_tet(order):
    celltype = basix.CellType.tetrahedron
    g = sympy_nedelec(celltype, order)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    nedelec = basix.create_element("Nedelec 2nd kind H(curl)", "tetrahedron", order)
    pts = basix.create_lattice(celltype, 6, basix.LatticeType.equispaced, True)
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

                assert(numpy.isclose(wtab[basix.index(kx, ky, kz)], wsym).all())


@pytest.mark.parametrize("order", [1, 2])
def test_n2curl_tetrahedron_permutations(order):
    e = basix.create_element("Nedelec 2nd kind H(curl)", "tetrahedron", order)
    if order == 1:
        perms = [[
            [1 if i == j else 0 for j in range(12)] for i in range(12)]
            for k in range(14)]
        for edge in range(6):
            perms[edge][2 * edge][2 * edge] = 0
            perms[edge][2 * edge + 1][2 * edge + 1] = 0
            perms[edge][2 * edge][2 * edge + 1] = -1
            perms[edge][2 * edge + 1][2 * edge] = -1

    elif order == 2:
        perms = [[
            [1 if i == j else 0 for j in range(30)] for i in range(30)]
            for k in range(14)]
        for edge in range(6):
            perms[edge][3 * edge][3 * edge] = 0
            perms[edge][3 * edge + 1][3 * edge + 1] = 0
            perms[edge][3 * edge + 2][3 * edge + 2] = 0
            perms[edge][3 * edge][3 * edge + 2] = -1
            perms[edge][3 * edge + 1][3 * edge + 1] = -1
            perms[edge][3 * edge + 2][3 * edge] = -1

        for face in range(4):
            perms[6 + 2 * face][18 + 3 * face][18 + 3 * face] = 0
            perms[6 + 2 * face][18 + 3 * face + 1][18 + 3 * face + 1] = 0
            perms[6 + 2 * face][18 + 3 * face + 2][18 + 3 * face + 2] = 0
            perms[6 + 2 * face][18 + 3 * face][18 + 3 * face + 1] = -1
            perms[6 + 2 * face][18 + 3 * face + 1][18 + 3 * face + 2] = -1
            perms[6 + 2 * face][18 + 3 * face + 2][18 + 3 * face] = 1

            perms[6 + 2 * face + 1][18 + 3 * face][18 + 3 * face] = 1
            perms[6 + 2 * face + 1][18 + 3 * face + 1][18 + 3 * face + 1] = 0
            perms[6 + 2 * face + 1][18 + 3 * face + 2][18 + 3 * face + 2] = 0
            perms[6 + 2 * face + 1][18 + 3 * face + 1][18 + 3 * face + 2] = -1
            perms[6 + 2 * face + 1][18 + 3 * face + 2][18 + 3 * face + 1] = -1

    else:
        raise NotImplementedError()

    assert np.allclose(e.base_permutations, perms)

