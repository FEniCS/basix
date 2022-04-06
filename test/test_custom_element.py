# Copyright (c) 2022 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import numpy as np


def test_lagrange_custom_triangle_degree1():
    """Test that Lagrange element created as a custom element agrees with built-in Lagrange."""

    wcoeffs = np.eye(3)
    z = np.zeros((0, 2))
    x = [[np.array([[0., 0.]]), np.array([[1., 0.]]), np.array([[0., 1.]])],
         [z, z, z], [z], []]
    z = np.zeros((0, 1, 0))
    M = [[np.array([[[1.]]]), np.array([[[1.]]]), np.array([[[1.]]])],
         [z, z, z], [z], []]

    lagrange = basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, 1)

    element = basix.create_custom_element(
        basix.CellType.triangle, 1, [], wcoeffs,
        x, M, basix.MapType.identity, False, 1, 1)

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
    id = np.array([[[1., 0., 0.]], [[0., 1., 0.]], [[0., 0., 1.]]])
    M = [[np.array([[[1.]]]), np.array([[[1.]]]), np.array([[[1.]]])],
         [id, id, id], [id], []]

    lagrange = basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, 4, basix.LagrangeVariant.equispaced)

    element = basix.create_custom_element(
        basix.CellType.triangle, 4, [], wcoeffs,
        x, M, basix.MapType.identity, False, 4)

    points = basix.create_lattice(basix.CellType.triangle, 5, basix.LatticeType.equispaced, True)
    assert np.allclose(lagrange.tabulate(1, points), element.tabulate(1, points))
    assert np.allclose(lagrange.base_transformations(), element.base_transformations())


def test_lagrange_custom_quadrilateral_degree1():
    """Test that Lagrange element created as a custom element agrees with built-in Lagrange."""

    wcoeffs = np.eye(4)
    z = np.zeros((0, 2))
    x = [[np.array([[0., 0.]]), np.array([[1., 0.]]), np.array([[0., 1.]]), np.array([[1., 1.]])],
         [z, z, z, z], [z], []]
    z = np.zeros((0, 1, 0))
    M = [[np.array([[[1.]]]), np.array([[[1.]]]), np.array([[[1.]]]), np.array([[[1.]]])],
         [z, z, z, z], [z], []]

    lagrange = basix.create_element(
        basix.ElementFamily.P, basix.CellType.quadrilateral, 1)

    element = basix.create_custom_element(
        basix.CellType.quadrilateral, 1, [], wcoeffs,
        x, M, basix.MapType.identity, False, 1)

    points = basix.create_lattice(basix.CellType.quadrilateral, 5, basix.LatticeType.equispaced, True)
    assert np.allclose(lagrange.tabulate(1, points), element.tabulate(1, points))
    assert np.allclose(lagrange.base_transformations(), element.base_transformations())
