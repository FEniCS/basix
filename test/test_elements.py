# Copyright (C) 2007-2024 Anders Logg Garth N. Wells, Marie E. Rognes, Lizao Li, Matthew Scroggs

# This test was originally part of FFCx
# Copyright (C) 2007-2017 Anders Logg and Garth N. Wells
#
# This file is part of FFCx.
#
# FFCx is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FFCx is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FFCx. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Marie E. Rognes, 2010
# Modified by Lizao Li, 2016

import random

import basix
import basix.ufl
import numpy as np
import pytest


def random_point(cell):
    vertices = basix.geometry(basix.cell.string_to_type(cell))
    w = [random.random() for _ in vertices]
    return sum(v * i for v, i in zip(vertices, w)) / sum(w)


@pytest.mark.parametrize(
    "family, cell, degree, dim",
    [
        ("Lagrange", "triangle", 1, 3),
        ("Lagrange", "triangle", 2, 6),
        ("Lagrange", "triangle", 3, 10),
        ("DG", "triangle", 1, 3),
        ("DG", "triangle", 2, 6),
        ("DG", "triangle", 3, 10),
        ("Lagrange", "quadrilateral", 1, 4),
        ("Lagrange", "quadrilateral", 2, 9),
        ("Lagrange", "quadrilateral", 3, 16),
        ("Regge", "triangle", 0, 3),
        ("Regge", "triangle", 1, 9),
        ("Regge", "triangle", 2, 18),
        ("Regge", "triangle", 3, 30),
        ("HHJ", "triangle", 0, 3),
        ("HHJ", "triangle", 1, 9),
        ("HHJ", "triangle", 2, 18),
        ("HHJ", "triangle", 3, 30),
    ],
)
def test_dimension(family, cell, degree, dim):
    e = basix.ufl.element(family, cell, degree)
    assert e.dim == dim


@pytest.mark.parametrize(
    "family, cell, degree, functions",
    [
        ("Lagrange", "interval", 1, [lambda x: 1 - x[0], lambda x: x[0]]),
        (
            "Lagrange",
            "triangle",
            1,
            [lambda x: 1 - x[0] - x[1], lambda x: x[0], lambda x: x[1]],
        ),
        (
            "Lagrange",
            "tetrahedron",
            1,
            [
                lambda x: 1 - x[0] - x[1] - x[2],
                lambda x: x[0],
                lambda x: x[1],
                lambda x: x[2],
            ],
        ),
        (
            "Lagrange",
            "quadrilateral",
            1,
            [
                lambda x: (1 - x[0]) * (1 - x[1]),
                lambda x: x[0] * (1 - x[1]),
                lambda x: (1 - x[0]) * x[1],
                lambda x: x[0] * x[1],
            ],
        ),
        (
            "Lagrange",
            "hexahedron",
            1,
            [
                lambda x: (1 - x[0]) * (1 - x[1]) * (1 - x[2]),
                lambda x: x[0] * (1 - x[1]) * (1 - x[2]),
                lambda x: (1 - x[0]) * x[1] * (1 - x[2]),
                lambda x: x[0] * x[1] * (1 - x[2]),
                lambda x: (1 - x[0]) * (1 - x[1]) * x[2],
                lambda x: x[0] * (1 - x[1]) * x[2],
                lambda x: (1 - x[0]) * x[1] * x[2],
                lambda x: x[0] * x[1] * x[2],
            ],
        ),
        (
            "Brezzi-Douglas-Marini",
            "triangle",
            1,
            [
                lambda x: (-x[0], -x[1]),
                lambda x: (3**0.5 * x[0], -(3**0.5) * x[1]),
                lambda x: (x[0] - 1, x[1]),
                lambda x: (3**0.5 * (1 - x[0] - 2 * x[1]), 3**0.5 * x[1]),
                lambda x: (-x[0], 1 - x[1]),
                lambda x: (-(3**0.5) * x[0], 3**0.5 * (2 * x[0] + x[1] - 1)),
            ],
        ),
        (
            "Raviart-Thomas",
            "triangle",
            1,
            [
                lambda x: (-x[0], -x[1]),
                lambda x: (x[0] - 1, x[1]),
                lambda x: (-x[0], 1 - x[1]),
            ],
        ),
        (
            "Raviart-Thomas",
            "tetrahedron",
            1,
            [
                lambda x: (2**0.5 * x[0], 2**0.5 * x[1], 2**0.5 * x[2]),
                lambda x: (2**0.5 - 2**0.5 * x[0], -(2**0.5) * x[1], -(2**0.5) * x[2]),
                lambda x: (2**0.5 * x[0], 2**0.5 * x[1] - 2**0.5, 2**0.5 * x[2]),
                lambda x: (-(2**0.5) * x[0], -(2**0.5) * x[1], 2**0.5 - 2**0.5 * x[2]),
            ],
        ),
        (
            "N1curl",
            "triangle",
            1,
            [
                lambda x: (-x[1], x[0]),
                lambda x: (x[1], 1 - x[0]),
                lambda x: (1.0 - x[1], x[0]),
            ],
        ),
        (
            "N1curl",
            "tetrahedron",
            1,
            [
                lambda x: (0.0, -x[2], x[1]),
                lambda x: (-x[2], 0.0, x[0]),
                lambda x: (-x[1], x[0], 0.0),
                lambda x: (x[2], x[2], 1.0 - x[0] - x[1]),
                lambda x: (x[1], 1.0 - x[0] - x[2], x[1]),
                lambda x: (1.0 - x[1] - x[2], x[0], x[0]),
            ],
        ),
    ],
)
def test_values(family, cell, degree, functions):
    # Create element
    e = basix.ufl.element(family, cell, degree)

    # Get some points and check basis function values at points
    points = [random_point(cell) for i in range(5)]
    tables = e.tabulate(0, np.array(points, dtype=np.float64))[0]
    for x, t in zip(points, tables):
        for i, f in enumerate(functions):
            assert np.allclose(t[i :: len(functions)], f(x))


def test_hash():
    e0 = basix.create_element(basix.ElementFamily.P, basix.CellType.interval, 1)
    e1 = basix.create_element(basix.ElementFamily.P, basix.CellType.interval, 1)
    e2 = basix.create_element(
        basix.ElementFamily.P, basix.CellType.interval, 1, dof_ordering=[0, 1]
    )
    e3 = basix.create_element(basix.ElementFamily.P, basix.CellType.interval, 2)
    e4 = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 1)

    wcoeffs = np.eye(3)
    z = np.zeros((0, 2))
    x = [
        [np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])],
        [z, z, z],
        [z],
        [],
    ]
    z = np.zeros((0, 1, 0, 1))
    M = [
        [np.array([[[[1.0]]]]), np.array([[[[1.0]]]]), np.array([[[[1.0]]]])],
        [z, z, z],
        [z],
        [],
    ]
    e5 = basix.create_custom_element(
        basix.CellType.triangle,
        [],
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.L2Piola,
        basix.SobolevSpace.L2,
        False,
        1,
        1,
        basix.PolysetType.standard,
    )

    e6 = basix.create_element(basix.ElementFamily.P, basix.CellType.quadrilateral, 2)
    e7 = basix.create_tp_element(basix.ElementFamily.P, basix.CellType.quadrilateral, 2)

    assert hash(e0) == hash(e1) == hash(e2)

    different_elements = [e2, e3, e4, e5, e6, e7]
    for i, d0 in enumerate(different_elements):
        for d1 in different_elements[:i]:
            assert hash(d0) != hash(d1)
