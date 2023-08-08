# Copyright (c) 2021 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import pytest
import numpy
import sympy

one = sympy.Integer(1)
x = sympy.Symbol("x")
y = sympy.Symbol("y")
z = sympy.Symbol("z")


@pytest.mark.parametrize("degree", range(6))
@pytest.mark.parametrize("cell_type", [
    basix.CellType.interval, basix.CellType.triangle, basix.CellType.quadrilateral,
    basix.CellType.tetrahedron, basix.CellType.hexahedron,
])
def test_legendre(cell_type, degree):
    points, weights = basix.make_quadrature(cell_type, 2 * degree)

    polys = basix.tabulate_polynomials(basix.PolynomialType.legendre, cell_type, degree, points)

    matrix = numpy.empty((polys.shape[0], polys.shape[0]))
    for i, col_i in enumerate(polys):
        for j, col_j in enumerate(polys):
            matrix[i, j] = sum(col_i * col_j * weights)

    assert numpy.allclose(matrix, numpy.identity(polys.shape[0]))


def evaluate(function, pt):
    if len(pt) == 1:
        return function.subs(x, pt[0])
    elif len(pt) == 2:
        return function.subs(x, pt[0]).subs(y, pt[1])
    elif len(pt) == 3:
        return function.subs(x, pt[0]).subs(y, pt[1]).subs(z, pt[2])


# TODO: pyramid
@pytest.mark.parametrize("cell_type, functions, degree", [
    [basix.CellType.interval, [one, x], 1],
    [basix.CellType.interval, [one, x, x**2], 2],
    [basix.CellType.interval, [one, x, x**2, x**3], 3],
    [basix.CellType.triangle, [one, y, x], 1],
    [basix.CellType.triangle, [one, y, x, y**2, x * y, x**2], 2],
    [basix.CellType.triangle, [one, y, x, y**2, x * y, x**2, y**3, x*y**2, x**2*y, x**3], 3],
    [basix.CellType.tetrahedron, [one], 0],
    [basix.CellType.tetrahedron, [one, z, y, x], 1],
    [basix.CellType.tetrahedron, [one, z, y, x, z**2, y * z, x * z, y**2, x * y, x**2], 2],
    [basix.CellType.tetrahedron, [one, z, y, x, z**2, y * z, x * z, y**2, x * y, x**2,
                                  z**3, y * z**2, x * z**2, y**2 * z, x * y * z, x**2 * z,
                                  y**3, x * y**2, x**2 * y, x**3], 3],
    [basix.CellType.quadrilateral, [one, y, x, x * y], 1],
    [basix.CellType.quadrilateral, [one, y, y**2, x, x * y, x * y**2, x**2, x**2 * y, x**2 * y**2], 2],
    [basix.CellType.hexahedron, [one, z, y, y * z, x, x * z, x * y, x * y * z], 1],
    [basix.CellType.prism, [one, z, y, y * z, x, x * z], 1],
    [basix.CellType.prism, [one, z, z**2, y, y * z, y * z**2, x, x * z, x * z**2,
                            y**2, y**2 * z, y**2 * z**2, x * y, x * y * z, x * y * z**2,
                            x**2, x**2 * z, x**2 * z**2], 2],
])
def test_order(cell_type, functions, degree):
    points, weights = basix.make_quadrature(cell_type, 2 * degree)
    polys = basix.tabulate_polynomials(basix.PolynomialType.legendre, cell_type, degree, points)

    assert len(functions) == polys.shape[0]

    eval_points = basix.create_lattice(cell_type, 10, basix.LatticeType.equispaced, True)
    eval_polys = basix.tabulate_polynomials(basix.PolynomialType.legendre, cell_type, degree, eval_points)

    for n, function in enumerate(functions):
        expected_eval = [float(evaluate(function, i)) for i in eval_points]

        # Using n polynomials
        # The monomial should NOT be exactly represented using this many
        coeffs = []
        values = numpy.array([evaluate(function, i) for i in points])
        for p in range(n):
            coeffs.append(sum(values * polys[p, :] * weights))
        actual_eval = [float(sum(coeffs * p[:n])) for p in eval_polys.T]
        assert not numpy.allclose(expected_eval, actual_eval)

        # Using n+1 polynomials
        # The monomial should be exactly represented using this many
        coeffs = []
        values = numpy.array([evaluate(function, i) for i in points])
        for p in range(n + 1):
            coeffs.append(sum(values * polys[p, :] * weights))
        actual_eval = [float(sum(coeffs * p[:n+1])) for p in eval_polys.T]
        assert numpy.allclose(expected_eval, actual_eval)
