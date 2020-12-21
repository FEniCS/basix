# Copyright (c) 2020 Chris Richardson & Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import numpy
import pytest
import sympy
from .test_lagrange import sympy_disc_lagrange


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
        for i in range(n):
            for j in range(n - i):
                for d in range(2):
                    funcs += [[x**j * y**i if k == d else 0 for k in range(2)]]
        for i in range(n):
            funcs += [[x ** (n - 1 - i) * y ** (i + 1),
                       -x ** (n - i) * y ** i]]
        mat = numpy.empty((len(funcs), len(funcs)), dtype=object)

        # edge tangents
        if n == 1:
            edge_basis = [sympy.Integer(1)]
        else:
            edge_basis = sympy_disc_lagrange(basix.CellType.interval, n - 1)
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
            if n == 2:
                face_basis = [sympy.Integer(1)]
            else:
                face_basis = sympy_disc_lagrange(basix.CellType.triangle, n - 2)
            for i, f in enumerate(funcs):
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
        if n == 1:
            funcs += [[y, -x, sympy.Integer(0)], [z, sympy.Integer(0), -x], [sympy.Integer(0), z, -y]]
        elif n == 2:
            funcs += [
                [y ** 2, -x * y, sympy.Integer(0)],
                [x * y, -x ** 2, sympy.Integer(0)],
                [z * y, -z * x, sympy.Integer(0)],
                [sympy.Integer(0), y * z, -y ** 2],
                [sympy.Integer(0), z ** 2, -z * y],
                [sympy.Integer(0), x * z, -x * y],
                [x * z, sympy.Integer(0), -x ** 2],
                [z ** 2, sympy.Integer(0), -z * x],
            ]
        elif n == 3:
            funcs += [
                [x ** 2 * y, -x ** 3, sympy.Integer(0)],
                [x ** 2 * z, sympy.Integer(0), -x ** 3],
                [sympy.Integer(0), x ** 2 * z, -x ** 2 * y],
                [x * y ** 2, -x ** 2 * y, sympy.Integer(0)],
                [2 * x * y * z, -x ** 2 * z, -x ** 2 * y],
                [sympy.Integer(0), x * y * z, -x * y ** 2],
                [x * z ** 2, sympy.Integer(0), -x ** 2 * z],
                [sympy.Integer(0), x * z ** 2, -x * y * z],
                [y ** 3, -x * y ** 2, sympy.Integer(0)],
                [9 * y ** 2 * z, -4 * x * y * z, -5 * x * y ** 2],
                [sympy.Integer(0), y ** 2 * z, -y ** 3],
                [9 * y * z ** 2, -5 * x * z ** 2, -4 * x * y * z],
                [sympy.Integer(0), y * z ** 2, -y ** 2 * z],
                [z ** 3, sympy.Integer(0), -x * z ** 2],
                [sympy.Integer(0), z ** 3, -y * z ** 2],
            ]
        else:
            raise NotImplementedError

        mat = numpy.empty((len(funcs), len(funcs)), dtype=object)

        # edge tangents
        if n == 1:
            edge_basis = [sympy.Integer(1)]
        else:
            edge_basis = sympy_disc_lagrange(basix.CellType.interval, n - 1)
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
            if n == 2:
                face_basis = [sympy.Integer(1)]
            else:
                face_basis = sympy_disc_lagrange(basix.CellType.triangle, n - 2)
            face_basis = [a.subs(x, dummy[0]).subs(y, dummy[1]) for a in face_basis]
            for i, f in enumerate(funcs):
                j = n * 6
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
                    for g in face_basis:
                        for vec in scaled_axes:
                            integrand = sum(f_i * v_i for f_i, v_i in zip(f, vec))
                            integrand = integrand.subs(x, param[0]).subs(y, param[1]).subs(z, param[2])
                            integrand *= g * norm

                            mat[i, j] = integrand.integrate((dummy[0], 0, 1 - dummy[1])).integrate((dummy[1], 0, 1))
                            j += 1
        # interior dofs
        if n > 2:
            if n == 3:
                interior_basis = [sympy.Integer(1)]
            else:
                interior_basis = sympy_disc_lagrange(basix.CellType.tetrahedron, n - 3)
            for i, f in enumerate(funcs):
                j = n * 6 + 4 * n * (n - 1)
                for g in interior_basis:
                    for vec in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                        integrand = sum(f_i * v_i for f_i, v_i in zip(f, vec))
                        integrand *= g

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
    nedelec = basix.Nedelec("triangle", order)
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


@pytest.mark.parametrize("order", [1, 2, 3])
def test_tet(order):
    celltype = basix.CellType.tetrahedron
    g = sympy_nedelec(celltype, order)
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    nedelec = basix.Nedelec("tetrahedron", order)

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


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_dof_permutations_triangle(order):
    nedelec = basix.Nedelec("triangle", order)

    permuted = {}
    if order == 2:
        # Reflect 2 DOFs on edges
        permuted[0] = {0: 1, 1: 0}
        permuted[1] = {2: 3, 3: 2}
        permuted[2] = {4: 5, 5: 4}
    elif order == 3:
        # Reflect 3 DOFs on edges
        permuted[0] = {0: 2, 2: 0}
        permuted[1] = {3: 5, 5: 3}
        permuted[2] = {6: 8, 8: 6}
    elif order == 4:
        # Reflect 4 DOFs on edges
        permuted[0] = {0: 3, 1: 2, 2: 1, 3: 0}
        permuted[1] = {4: 7, 5: 6, 6: 5, 7: 4}
        permuted[2] = {8: 11, 9: 10, 10: 9, 11: 8}

    base_perms = nedelec.base_permutations
    assert len(base_perms) == 3

    for i, perm in enumerate(base_perms):
        actual = numpy.zeros_like(perm)

        for row in range(perm.shape[0]):
            for k in range(perm.shape[1]):
                if i in permuted and k in permuted[i]:
                    col = permuted[i][k]
                else:
                    col = k
                if row == k:
                    if order * i <= row < order * (i + 1):
                        actual[row, col] = -1
                    else:
                        actual[row, col] = 1
        assert numpy.allclose(perm, actual)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_dof_permutations_tetrahedron(order):
    nedelec = basix.Nedelec("tetrahedron", order)

    permuted = {}
    if order == 2:
        # Reflect 2 DOFs on edges
        for i in range(6):
            permuted[i] = {2 * i: 2 * i + 1, 2 * i + 1: 2 * i}
    elif order == 3:
        # Reflect 3 DOFs on edges
        for i in range(6):
            permuted[i] = {3 * i: 3 * i + 2, 3 * i + 2: 3 * i}
        # Rotate and reflect 3*2 DOFs on faces
        for i in range(4):
            permuted[6 + 2 * i] = {
                18 + 6 * i: 22 + 6 * i, 19 + 6 * i: 23 + 6 * i, 20 + 6 * i: 18 + 6 * i,
                21 + 6 * i: 19 + 6 * i, 22 + 6 * i: 20 + 6 * i, 23 + 6 * i: 21 + 6 * i}
            permuted[7 + 2 * i] = {
                20 + 6 * i: 22 + 6 * i, 21 + 6 * i: 23 + 6 * i,
                22 + 6 * i: 20 + 6 * i, 23 + 6 * i: 21 + 6 * i}
    elif order == 4:
        # Reflect 3 DOFs on edges
        for i in range(6):
            permuted[i] = {4 * i + j: 4 * i + 3 - j for j in range(4)}
        # Rotate and reflect 6*2 DOFs on faces
        for i in range(4):
            permuted[6 + 2 * i] = {
                24 + a + 12 * i: 24 + b + 12 * i for a, b in {
                    0: 10, 1: 11, 2: 6, 3: 7, 4: 0, 5: 1, 6: 8,
                    7: 9, 8: 2, 9: 3, 10: 4, 11: 5}.items()}
            permuted[7 + 2 * i] = {
                24 + a + 12 * i: 24 + b + 12 * i for a, b in {
                    2: 6, 3: 7, 4: 10, 5: 11, 6: 2, 7: 3, 10: 4, 11: 5}.items()}

    base_perms = nedelec.base_permutations
    assert len(base_perms) == 14

    # Test edge flips
    for i, perm in enumerate(base_perms[:6]):
        actual = numpy.zeros_like(perm)
        for row in range(perm.shape[0]):
            for k in range(perm.shape[1]):
                if i in permuted and k in permuted[i]:
                    col = permuted[i][k]
                else:
                    col = k
                if row == k:
                    if order * i <= row < order * (i + 1):
                        actual[row, col] = -1
                    else:
                        actual[row, col] = 1
        assert numpy.allclose(perm, actual)

    # TODO: write good test for face rotations
