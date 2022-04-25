# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import numpy
import pytest
import sympy


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
    assert(numpy.isclose(numpy.sum(w, axis=1), 1.0).all())


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
    assert(numpy.allclose(numpy.sum(w, axis=1), 1.0))


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
    basix.LagrangeVariant.gll_warped, basix.LagrangeVariant.gll_isaac, basix.LagrangeVariant.gll_centroid,
    basix.LagrangeVariant.chebyshev_warped, basix.LagrangeVariant.chebyshev_isaac,
    basix.LagrangeVariant.chebyshev_centroid,
    basix.LagrangeVariant.gl_warped, basix.LagrangeVariant.gl_isaac, basix.LagrangeVariant.gl_centroid,
    basix.LagrangeVariant.vtk
])
@pytest.mark.parametrize("celltype", [
    basix.CellType.triangle, basix.CellType.tetrahedron,
    basix.CellType.quadrilateral, basix.CellType.hexahedron,
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
    # The variants used in this test can only be used for discontinuous Lagrange,
    # so trying to create them should throw a runtime error
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
