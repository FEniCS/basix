import libtab
import numpy


def test_create():

    # Creates Lagrange P1 element on triangle

    # Point evaluation of polynomial set
    celltype = libtab.CellType.triangle
    degree = 1
    points = numpy.array([[0, 0], [1, 0], [0, 1]], dtype=numpy.float64)

    # Create element from space and dual
    dualmat = libtab.tabulate_polynomial_set(celltype, degree, 0, points)[0]
    coeff_space = numpy.identity(points.shape[0])
    fe = libtab.create_new_element(celltype, degree, 1, dualmat, coeff_space, [1, 0, 0, 0])

    numpy.set_printoptions(suppress=True, precision=2)

    points = numpy.array([[.5, 0], [0, .5], [.5, .5]], dtype=numpy.float64)
    print(fe.tabulate(0, points))
