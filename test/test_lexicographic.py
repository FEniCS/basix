# Copyright (c) 2025 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import pytest

import basix


@pytest.mark.parametrize(
    ("cell_type", "family", "degree", "args", "perm"),
    [
        (basix.CellType.interval, basix.ElementFamily.P, 0, (), [0]),
        (basix.CellType.interval, basix.ElementFamily.P, 1, (), [0, 1]),
        (basix.CellType.interval, basix.ElementFamily.P, 2, (), [0, 2, 1]),
        (basix.CellType.interval, basix.ElementFamily.P, 3, (), [0, 2, 3, 1]),
        (basix.CellType.interval, basix.ElementFamily.P, 4, (), [0, 2, 3, 4, 1]),  # noqa: E501
        (basix.CellType.triangle, basix.ElementFamily.P, 0, (), [0]),
        (basix.CellType.triangle, basix.ElementFamily.P, 1, (), [0, 1, 2]),
        (basix.CellType.triangle, basix.ElementFamily.P, 2, (), [0, 5, 1, 4, 3, 2]),  # noqa: E501
        (basix.CellType.triangle, basix.ElementFamily.P, 3, (), [0, 7, 8, 1, 5, 9, 3, 6, 4, 2]),  # noqa: E501
        (basix.CellType.triangle, basix.ElementFamily.P, 4, (), [0, 9, 10, 11, 1, 6, 12, 13, 3, 7, 14, 4, 8, 5, 2]),  # noqa: E501
        (basix.CellType.quadrilateral, basix.ElementFamily.P, 0, (), [0]),
        (basix.CellType.quadrilateral, basix.ElementFamily.P, 1, (), [0, 1, 2, 3]),  # noqa: E501
        (basix.CellType.quadrilateral, basix.ElementFamily.P, 2, (), [0, 4, 1, 5, 8, 6, 2, 7, 3]),  # noqa: E501
        (basix.CellType.quadrilateral, basix.ElementFamily.P, 3, (), [0, 4, 5, 1, 6, 12, 13, 8, 7, 14, 15, 9, 2, 10, 11, 3]),  # noqa: E501
        (basix.CellType.tetrahedron, basix.ElementFamily.P, 0, (), [0]),
        (basix.CellType.tetrahedron, basix.ElementFamily.P, 1, (), [0, 1, 2, 3]),  # noqa: E501
        (basix.CellType.tetrahedron, basix.ElementFamily.P, 2, (), [0, 9, 1, 8, 6, 2, 7, 5, 4, 3]),  # noqa: E501
        (basix.CellType.tetrahedron, basix.ElementFamily.P, 3, (), [0, 14, 15, 1, 12, 19, 8, 13, 9, 2, 10, 18, 6, 17, 16, 4, 11, 7, 5, 3]),  # noqa: E501
        (basix.CellType.hexahedron, basix.ElementFamily.P, 0, (), [0]),
        (basix.CellType.hexahedron, basix.ElementFamily.P, 1, (), [0, 1, 2, 3, 4, 5, 6, 7]),  # noqa: E501
        (basix.CellType.hexahedron, basix.ElementFamily.P, 2, (), [0, 8, 1, 9, 20, 11, 2, 13, 3, 10, 21, 12, 22, 26, 23, 14, 24, 15, 4, 16, 5, 17, 25, 18, 6, 19, 7]),  # noqa: E501
        (basix.CellType.hexahedron, basix.ElementFamily.P, 3, (), [0, 8, 9, 1, 10, 32, 33, 14, 11, 34, 35, 15, 2, 18, 19, 3, 12, 36, 37, 16, 40, 56, 57, 44, 41, 58, 59, 45, 20, 48, 49, 22, 13, 38, 39, 17, 42, 60, 61, 46, 43, 62, 63, 47, 21, 50, 51, 23, 4, 24, 25, 5, 26, 52, 53, 28, 27, 54, 55, 29, 6, 30, 31, 7]),  # noqa: E501
    ],
)
def test_dof_ordering(cell_type, family, args, degree, perm):
    ordering = basix.finite_element.lex_dof_ordering(family, cell_type, degree, *args)
    for i, p in enumerate(perm):
        assert ordering[p] == i
