# Copyright (c) 2025 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import numpy as np

import basix


def test_rannacher_turek_quadrilateral():
    e = basix.create_element(basix.ElementFamily.CR, basix.CellType.quadrilateral, 1)
    pts = basix.create_lattice(basix.CellType.quadrilateral, 5, basix.LatticeType.equispaced, True)
    table = e.tabulate(0, pts)

    # Compare against basis functions taken from
    # https://defelement.org/elements/examples/quadrilateral-rannacher-turek-1.html
    expected_table = np.array(
        [
            [
                [
                    [p[1] ** 2 - p[0] ** 2 + p[0] - 2 * p[1] + 3 / 4],
                    [p[0] ** 2 - p[1] ** 2 + p[1] - 2 * p[0] + 3 / 4],
                    [p[0] ** 2 - p[1] ** 2 + p[1] - 1 / 4],
                    [p[1] ** 2 - p[0] ** 2 + p[0] - 1 / 4],
                ]
                for p in pts
            ]
        ]
    )

    assert np.allclose(table, expected_table)


def test_rannacher_turek_hexahedron():
    e = basix.create_element(basix.ElementFamily.CR, basix.CellType.hexahedron, 1)
    pts = basix.create_lattice(basix.CellType.hexahedron, 5, basix.LatticeType.equispaced, True)
    table = e.tabulate(0, pts)

    # Compare against basis functions taken from Symfem
    expected_table = np.array(
        [
            [
                [
                    [
                        -2 * p[0] ** 2 / 3
                        + 2 * p[0] / 3
                        - 2 * p[1] ** 2 / 3
                        + 2 * p[1] / 3
                        + 4 * p[2] ** 2 / 3
                        - 7 * p[2] / 3
                        + 2 / 3
                    ],
                    [
                        -2 * p[0] ** 2 / 3
                        + 2 * p[0] / 3
                        + 4 * p[1] ** 2 / 3
                        - 7 * p[1] / 3
                        - 2 * p[2] ** 2 / 3
                        + 2 * p[2] / 3
                        + 2 / 3
                    ],
                    [
                        4 * p[0] ** 2 / 3
                        - 7 * p[0] / 3
                        - 2 * p[1] ** 2 / 3
                        + 2 * p[1] / 3
                        - 2 * p[2] ** 2 / 3
                        + 2 * p[2] / 3
                        + 2 / 3
                    ],
                    [
                        4 * p[0] ** 2 / 3
                        - p[0] / 3
                        - 2 * p[1] ** 2 / 3
                        + 2 * p[1] / 3
                        - 2 * p[2] ** 2 / 3
                        + 2 * p[2] / 3
                        - 1 / 3
                    ],
                    [
                        -2 * p[0] ** 2 / 3
                        + 2 * p[0] / 3
                        + 4 * p[1] ** 2 / 3
                        - p[1] / 3
                        - 2 * p[2] ** 2 / 3
                        + 2 * p[2] / 3
                        - 1 / 3
                    ],
                    [
                        -2 * p[0] ** 2 / 3
                        + 2 * p[0] / 3
                        - 2 * p[1] ** 2 / 3
                        + 2 * p[1] / 3
                        + 4 * p[2] ** 2 / 3
                        - p[2] / 3
                        - 1 / 3
                    ],
                ]
                for p in pts
            ]
        ]
    )

    assert np.allclose(table, expected_table)
