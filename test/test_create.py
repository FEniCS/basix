import pytest
import libtab
import numpy


def test_create_simple():

    # Creates Lagrange P1 element on triangle

    # Point evaluation of polynomial set
    celltype = libtab.CellType.triangle
    degree = 1
    points = numpy.array([[0, 0], [1, 0], [0, 1]], dtype=numpy.float64)

    # Create element from space and dual
    dualmat = libtab.tabulate_polynomial_set(celltype, degree, 0, points)[0]
    coeff_space = numpy.identity(points.shape[0])
    fe = libtab.create_new_element(celltype, degree, [1], dualmat, coeff_space, [[1, 1, 1], [0, 0, 0], [0]])

    numpy.set_printoptions(suppress=True, precision=2)

    points = numpy.array([[.5, 0], [0, .5], [.5, .5]], dtype=numpy.float64)
    print(fe.tabulate(0, points))


def test_create_custom():

    # Creates second order element on triangle

    # Point evaluation of polynomial set
    celltype = libtab.CellType.triangle
    degree = 2
    points = numpy.array([[0, .5], [0.5, 0], [0.5, 0.5], [0.25, 0.25], [0.25, 0.5], [0.5, 0.25]], dtype=numpy.float64)

    # Create element from space and dual
    dualmat = libtab.tabulate_polynomial_set(celltype, degree, 0, points)[0]
    coeff_space = numpy.identity(points.shape[0])
    fe = libtab.create_new_element(celltype, degree, [1], dualmat, coeff_space,
                                   [[0, 0, 0], [1, 1, 1], [3]])

    numpy.set_printoptions(suppress=True, precision=2)

    points = numpy.array([[.25, 0], [0, .25], [.25, .25]], dtype=numpy.float64)
    print(fe.tabulate(0, points))


def test_create_invalid():

    celltype = libtab.CellType.triangle
    degree = 2
    # Try to create an invalid element of order 2
    points = numpy.array([[0, 0.25], [0, 0.75], [0.25, 0.75], [0.75, 0.25],
                          [0.25, 0.0], [0.75, 0.0]], dtype=numpy.float64)

    # Create element from space and dual
    dualmat = libtab.tabulate_polynomial_set(celltype, degree, 0, points)[0]
    print(dualmat)
    coeff_space = numpy.identity(points.shape[0])
    with pytest.raises(RuntimeError):
        libtab.create_new_element(celltype, degree, [1], dualmat, coeff_space,
                                  [[0, 0, 0], [2, 2, 2], [0]])
