import basix
import numpy
import pytest


def test_create_simple():
    # Creates Lagrange P1 element on triangle

    # Point evaluation of polynomial set
    degree = 1
    points = numpy.array([[0, 0], [1, 0], [0, 1]], dtype=numpy.float64)
    matrix = numpy.identity(points.shape[0])

    # Create element from space and dual
    coeff_space = numpy.identity(points.shape[0])
    fe = basix.create_new_element("Custom element", "triangle", degree, [1], points, matrix, coeff_space,
                                  [[1, 1, 1], [0, 0, 0], [0]], [numpy.identity(3) for i in range(3)],
                                  basix.MappingType.identity)
    numpy.set_printoptions(suppress=True, precision=2)
    points = numpy.array([[.5, 0], [0, .5], [.5, .5]], dtype=numpy.float64)
    print(fe.tabulate(0, points))


def xtest_create_custom():
    # Creates second order element on triangle

    # Point evaluation of polynomial set
    degree = 2
    points = numpy.array([[0, .5], [0.5, 0], [0.5, 0.5], [0.25, 0.25], [0.25, 0.5], [0.5, 0.25]], dtype=numpy.float64)
    matrix = numpy.identity(points.shape[0])

    # Create element from space and dual
    coeff_space = numpy.identity(points.shape[0])
    fe = basix.create_new_element("Custom element", "triangle", degree, [1], points, matrix, coeff_space,
                                  [[0, 0, 0], [1, 1, 1], [3]],
                                  [numpy.identity(5) for i in range(3)],
                                  basix.MappingType.identity)
    numpy.set_printoptions(suppress=True, precision=2)
    points = numpy.array([[.25, 0], [0, .25], [.25, .25]], dtype=numpy.float64)
    print(fe.tabulate(0, points))


def xtest_create_invalid():
    degree = 2
    # Try to create an invalid element of order 2
    points = numpy.array([[0, 0.25], [0, 0.75], [0.25, 0.75], [0.75, 0.25],
                          [0.25, 0.0], [0.75, 0.0]], dtype=numpy.float64)
    matrix = numpy.identity(points.shape[0])

    # Create element from space and dual
    coeff_space = numpy.identity(points.shape[0])
    with pytest.raises(RuntimeError):
        basix.create_new_element("Custom element", "triangle", degree, [1], points, matrix, coeff_space,
                                 [[0, 0, 0], [2, 2, 2], [0]],
                                 [numpy.identity(6) for i in range(3)], basix.MappingType.identity)
