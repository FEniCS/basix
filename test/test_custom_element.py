# Copyright (c) 2022 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import pytest
import numpy as np


def test_lagrange_custom_triangle_degree1():
    """Test that Lagrange element created as a custom element agrees with built-in Lagrange."""

    wcoeffs = np.eye(3)
    z = np.zeros((0, 2))
    x = [[np.array([[0., 0.]]), np.array([[1., 0.]]), np.array([[0., 1.]])],
         [z, z, z], [z], []]
    z = np.zeros((0, 1, 0, 1))
    M = [[np.array([[[[1.]]]]), np.array([[[[1.]]]]), np.array([[[[1.]]]])],
         [z, z, z], [z], []]

    lagrange = basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, 1)

    element = basix.create_custom_element(
        basix.CellType.triangle, [], wcoeffs,
        x, M, 0, basix.MapType.identity, False, 1, 1)

    points = basix.create_lattice(basix.CellType.triangle, 5, basix.LatticeType.equispaced, True)
    assert np.allclose(lagrange.tabulate(1, points), element.tabulate(1, points))
    assert np.allclose(lagrange.base_transformations(), element.base_transformations())


def test_lagrange_custom_triangle_degree1_l2piola():
    """Test a custom element with a L2 Piola map."""

    wcoeffs = np.eye(3)
    z = np.zeros((0, 2))
    x = [[np.array([[0., 0.]]), np.array([[1., 0.]]), np.array([[0., 1.]])],
         [z, z, z], [z], []]
    z = np.zeros((0, 1, 0, 1))
    M = [[np.array([[[[1.]]]]), np.array([[[[1.]]]]), np.array([[[[1.]]]])],
         [z, z, z], [z], []]

    lagrange = basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, 1)

    element = basix.create_custom_element(
        basix.CellType.triangle, [], wcoeffs,
        x, M, 0, basix.MapType.L2Piola, False, 1, 1)

    points = basix.create_lattice(basix.CellType.triangle, 5, basix.LatticeType.equispaced, True)
    assert np.allclose(lagrange.tabulate(1, points), element.tabulate(1, points))
    assert np.allclose(lagrange.base_transformations(), element.base_transformations())


def test_lagrange_custom_triangle_degree4():
    """Test that Lagrange element created as a custom element agrees with built-in Lagrange."""

    wcoeffs = np.eye(15)
    x = [[np.array([[0., 0.]]), np.array([[1., 0.]]), np.array([[0., 1.]])],
         [np.array([[.75, .25], [.5, .5], [.25, .75]]), np.array([[0., .25], [0., .5], [0., .75]]),
          np.array([[.25, 0.], [.5, 0.], [.75, 0.]])],
         [np.array([[.25, .25], [.5, .25], [.25, .5]])], []]
    id = np.array([[[[1.], [0.], [0.]]], [[[0.], [1.], [0.]]], [[[0.], [0.], [1.]]]])
    M = [[np.array([[[[1.]]]]), np.array([[[[1.]]]]), np.array([[[[1.]]]])],
         [id, id, id], [id], []]

    lagrange = basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, 4, basix.LagrangeVariant.equispaced)

    element = basix.create_custom_element(
        basix.CellType.triangle, [], wcoeffs,
        x, M, 0, basix.MapType.identity, False, 4, 4)

    points = basix.create_lattice(basix.CellType.triangle, 5, basix.LatticeType.equispaced, True)
    assert np.allclose(lagrange.tabulate(1, points), element.tabulate(1, points))
    assert np.allclose(lagrange.base_transformations(), element.base_transformations())


def test_lagrange_custom_quadrilateral_degree1():
    """Test that Lagrange element created as a custom element agrees with built-in Lagrange."""

    wcoeffs = np.eye(4)
    z = np.zeros((0, 2))
    x = [[np.array([[0., 0.]]), np.array([[1., 0.]]), np.array([[0., 1.]]), np.array([[1., 1.]])],
         [z, z, z, z], [z], []]
    z = np.zeros((0, 1, 0, 1))
    M = [[np.array([[[[1.]]]]), np.array([[[[1.]]]]), np.array([[[[1.]]]]), np.array([[[[1.]]]])],
         [z, z, z, z], [z], []]

    lagrange = basix.create_element(
        basix.ElementFamily.P, basix.CellType.quadrilateral, 1)

    element = basix.create_custom_element(
        basix.CellType.quadrilateral, [], wcoeffs,
        x, M, 0, basix.MapType.identity, False, 1, 1)

    points = basix.create_lattice(basix.CellType.quadrilateral, 5, basix.LatticeType.equispaced, True)
    assert np.allclose(lagrange.tabulate(1, points), element.tabulate(1, points))
    assert np.allclose(lagrange.base_transformations(), element.base_transformations())


def test_raviart_thomas_triangle_degree1():
    """Test that Raviart-Thomas element created as a custom element agrees with built-in Raviart-Thomas."""

    wcoeffs = np.zeros((3, 6))
    wcoeffs[0, 0] = 1
    wcoeffs[1, 3] = 1

    pts, wts = basix.make_quadrature(basix.CellType.triangle, 2)
    poly = basix.tabulate_polynomials(basix.PolynomialType.legendre, basix.CellType.triangle, 1, pts)
    for i in range(3):
        wcoeffs[2, i] = sum(pts[:, 0] * poly[i, :] * wts)
        wcoeffs[2, 3 + i] = sum(pts[:, 1] * poly[i, :] * wts)

    pts, wts = basix.make_quadrature(basix.CellType.interval, 2)

    x = [[], [], [], []]
    for _ in range(3):
        x[0].append(np.zeros((0, 2)))
    x[1].append(np.array([[1 - p[0], p[0]] for p in pts]))
    x[1].append(np.array([[0, p[0]] for p in pts]))
    x[1].append(np.array([[p[0], 0] for p in pts]))
    x[2].append(np.zeros((0, 2)))

    M = [[], [], [], []]
    for _ in range(3):
        M[0].append(np.zeros((0, 2, 0, 1)))
    for normal in [[-1, -1], [-1, 0], [0, 1]]:
        mat = np.empty((1, 2, len(wts), 1))
        mat[0, 0, :, 0] = normal[0] * wts
        mat[0, 1, :, 0] = normal[1] * wts
        M[1].append(mat)
    M[2].append(np.zeros((0, 2, 0, 1)))

    element = basix.create_custom_element(
        basix.CellType.triangle, [2], wcoeffs, x, M, 0, basix.MapType.contravariantPiola, False, 0, 1)

    rt = basix.create_element(
        basix.ElementFamily.RT, basix.CellType.triangle, 1)

    points = basix.create_lattice(basix.CellType.triangle, 5, basix.LatticeType.equispaced, True)
    assert np.allclose(rt.tabulate(1, points), element.tabulate(1, points))
    assert np.allclose(rt.base_transformations(), element.base_transformations())


def create_lagrange1_quad(cell_type=basix.CellType.quadrilateral, degree=1, wcoeffs=None, x=None, M=None,
                          value_shape=None, interpolation_nderivs=0, discontinuous=False):
    """Attempt to create a Lagrange 1 element on a quad."""
    if wcoeffs is None:
        wcoeffs = np.eye(4)
    if x is None:
        z = np.zeros((0, 2))
        x = [[np.array([[0., 0.]]), np.array([[1., 0.]]), np.array([[0., 1.]]), np.array([[1., 1.]])],
             [z, z, z, z], [z], []]
    if M is None:
        z = np.zeros((0, 1, 0, 1))
        M = [[np.array([[[[1.]]]]), np.array([[[[1.]]]]), np.array([[[[1.]]]]), np.array([[[[1.]]]])],
             [z, z, z, z], [z], []]
    if value_shape is None:
        value_shape = []
    basix.create_custom_element(
        cell_type, value_shape, wcoeffs, x, M, interpolation_nderivs, basix.MapType.identity,
        discontinuous, 1, degree)


def test_create_lagrange1_quad():
    """Test that the above function works if no inputs are modified."""
    create_lagrange1_quad()


def assert_failure(**kwargs):
    """Assert that the correct RuntimeError is thrown."""
    try:
        create_lagrange1_quad(**kwargs)
    except RuntimeError as e:
        if len(e.args) == 0:
            raise e
        if "dgesv" in e.args[0]:
            raise e
        return
    with pytest.raises(RuntimeError):
        pass


def test_wcoeffs_wrong_shape():
    """Test that a runtime error is thrown when wcoeffs is the wrong shape."""
    assert_failure(wcoeffs=np.eye(3))


def test_wcoeffs_too_few_cols():
    """Test that a runtime error is thrown when wcoeffs has too few columns."""
    assert_failure(wcoeffs=np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]))


def test_wcoeffs_too_few_rows():
    """Test that a runtime error is thrown when wcoeffs has too few rows."""
    assert_failure(wcoeffs=np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 0., 1.]]))


def test_wcoeffs_zero_row():
    """Test that a runtime error is thrown when wcoeffs has a row of zeros."""
    assert_failure(wcoeffs=np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 0.]]))


def test_wcoeffs_equal_rows():
    """Test that a runtime error is thrown when wcoeffs has two equal rows."""
    assert_failure(wcoeffs=np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 1.], [0., 0., 1., 1.]]))


def test_x_wrong_tdim():
    """Test that a runtime error is thrown when a point in x has the wrong tdim."""
    z = np.zeros((0, 2))
    x = [[np.array([[0., 0., 0.]]), np.array([[1., 0.]]), np.array([[0., 1.]]), np.array([[1., 1.]])],
         [z, z, z, z], [z], []]
    assert_failure(x=x)


def test_x_wrong_ndims():
    """Test that a runtime error is thrown when x has the wrong shape."""
    z = np.zeros((0, 2))
    x = [[np.array([0., 0.]), np.array([[1., 0.]]), np.array([[0., 1.]]), np.array([[1., 1.]])],
         [z, z, z, z], [z], []]
    assert_failure(x=x)


def test_x_too_many_points():
    """Test that a runtime error is thrown when x has too many points."""
    z = np.zeros((0, 2))
    x = [[np.array([[0., 0.]]), np.array([[1., 0.]]), np.array([[0., 1.]]), np.array([[1., 1.]])],
         [z, z, z, z], [np.array([[.5, .5]])], []]
    assert_failure(x=x)


def test_x_point_tdim_too_high():
    """Test that a runtime error is thrown when x has a point in an entity with a too high dimension."""
    z = np.zeros((0, 2))
    x = [[np.array([[0., 0.]]), np.array([[1., 0.]]), np.array([[0., 1.]]), z],
         [z, z, z, z], [z], [np.array([[1., 1.]])]]
    assert_failure(x=x)


def test_x_wrong_entity_count():
    """Test that a runtime error is thrown when x has the wrong number of edges."""
    z = np.zeros((0, 2))
    x = [[np.array([[0., 0.]]), np.array([[1., 0.]]), np.array([[0., 1.]]), np.array([[1., 1.]])],
         [z, z, z, z, z], [z], []]
    assert_failure(x=x)


def test_M_wrong_value_size():
    """Test that a runtime error is thrown when M has the wrong shape."""
    z = np.zeros((0, 1, 0, 1))
    M = [[np.array([[[[1.], [1.]]]]), np.array([[[[1.]]]]), np.array([[[[1.]]]]), np.array([[[[1.]]]])],
         [z, z, z, z], [z], []]
    assert_failure(M=M)


def test_M_wrong_ndims():
    """Test that a runtime error is thrown when M has the wrong shape."""
    z = np.zeros((0, 1, 0, 1))
    M = [[np.array([[[1.]]]), np.array([[[[1.]]]]), np.array([[[[1.]]]]), np.array([[[[1.]]]])],
         [z, z, z, z], [z], []]
    assert_failure(M=M)


def test_M_too_many_points():
    """Test that a runtime error is thrown when M is the wrong shape."""
    z = np.zeros((0, 1, 0, 1))
    M = [[np.array([[[[1.]]]]), np.array([[[[1.]]]]), np.array([[[[1.]]]]), np.array([[[[1.]]]])],
         [z, z, z, z], [np.array([[[[1.]]]])], []]
    assert_failure(M=M)

    z = np.zeros((0, 1, 0, 1))
    M = [[np.array([[[[1.], [1.]]]]), np.array([[[[1.]]]]), np.array([[[[1.]]]]), np.array([[[[1.]]]])],
         [z, z, z, z], [z], []]
    assert_failure(M=M)


def test_M_wrong_entities():
    """Test that a runtime error is thrown when the shape of M does not match x."""
    z = np.zeros((0, 1, 0, 1))
    M = [[np.array([[[[1.]]]]), np.array([[[[1.]]]]), np.array([[[[1.]]], [[[1.]]]]), z],
         [z, z, z, z], [z], []]
    assert_failure(M=M)


def test_M_too_many_derivs():
    """Test that a runtime error is thrown when M is the wrong shape."""
    z = np.zeros((0, 1, 0, 1))
    M = [[np.array([[[[1.]]]]), np.array([[[[1.]]]]), np.array([[[[1., 1.]]]]), np.array([[[[1.]]]])],
         [z, z, z, z], [z], []]
    assert_failure(M=M)


def test_M_zero_row():
    """Test that a runtime error is thrown when M has a zero row."""
    z = np.zeros((0, 1, 0, 1))
    M = [[np.array([[[[1.]]]]), np.array([[[[1.]]]]), np.array([[[[1.]]]]), np.array([[[[0.]]]])],
         [z, z, z, z], [z], []]
    assert_failure(M=M)


def test_wrong_value_shape():
    """Test that a runtime error is thrown when value shape is wrong."""
    assert_failure(value_shape=[2])


def test_wrong_cell_type():
    """Test that a runtime error is thrown when cell type is wrong."""
    assert_failure(cell_type=basix.CellType.hexahedron)


def test_wrong_degree():
    """Test that a runtime error is thrown when degree is wrong."""
    assert_failure(degree=0)


def test_wrong_discontinuous():
    """Test that a runtime error is thrown when discontinuous is wrong."""
    assert_failure(discontinuous=True)


def test_wrong_interpolation_nderivs():
    """Test that a runtime error is thrown when number of interpolation derivatives is wrong."""
    assert_failure(interpolation_nderivs=1)
