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
