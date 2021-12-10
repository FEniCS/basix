import basix
import pytest
import numpy


@pytest.mark.parametrize("degree", range(6))
@pytest.mark.parametrize("cell_type", [
    basix.CellType.interval, basix.CellType.triangle, basix.CellType.quadrilateral,
    basix.CellType.tetrahedron, basix.CellType.hexahedron,
])
def test_legendre(cell_type, degree):
    points, weights = basix.make_quadrature(cell_type, 2 * degree)

    polys = basix.tabulate_polynomials(basix.PolynomialType.legendre, cell_type, degree, points)

    matrix = numpy.empty((polys.shape[1], polys.shape[1]))
    for i, col_i in enumerate(polys.T):
        for j, col_j in enumerate(polys.T):
            matrix[i, j] = sum(col_i * col_j * weights)

    assert numpy.allclose(matrix, numpy.identity(polys.shape[1]))


@pytest.mark.parametrize("degree", range(6))
@pytest.mark.parametrize("cell_type", [
    basix.CellType.interval,
    # basix.CellType.triangle, basix.CellType.tetrahedron,
    basix.CellType.quadrilateral, basix.CellType.hexahedron,
])
def test_chebyshev(cell_type, degree):
    # Use quadrature for integrating Chebyshev integral
    n = degree + 1
    points1d = numpy.array([0.5 + 0.5 * numpy.cos((2 * i - 1) / (2 * n) * numpy.pi) for i in range(1, n + 1)])
    weights1d = numpy.array([numpy.pi / n for i in range(1, n + 1)])

    if cell_type == basix.CellType.interval:
        points = points1d.reshape((-1, 1))
        weights = weights1d
    if cell_type == basix.CellType.quadrilateral:
        points = numpy.array([[p0, p1] for p0 in points1d for p1 in points1d])
        weights = numpy.array([w0 * w1 for w0 in weights1d for w1 in weights1d])
    elif cell_type == basix.CellType.hexahedron:
        points = numpy.array([[p0, p1, p2] for p0 in points1d for p1 in points1d for p2 in points1d])
        weights = numpy.array([w0 * w1 * w2 for w0 in weights1d for w1 in weights1d for w2 in weights1d])

    polys = basix.tabulate_polynomials(basix.PolynomialType.chebyshev, cell_type, degree, points)

    matrix = numpy.empty((polys.shape[1], polys.shape[1]))
    for i, col_i in enumerate(polys.T):
        for j, col_j in enumerate(polys.T):
            matrix[i, j] = sum(col_i * col_j * weights)

    print(matrix)

    for i, row in enumerate(matrix):
        for j, entry in enumerate(row):
            if i != j:
                assert numpy.isclose(entry, 0)
