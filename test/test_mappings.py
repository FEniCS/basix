# Copyright (c) 2020 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import random
import basix
import pytest
import numpy as np

elements = [
    "Lagrange",  # identity
    # "Nedelec 1st kind H(curl)",  # covariant Piola
    # "Raviart-Thomas",  # contravariant Piola
    # "Regge",  # double covariant Piola
]


def run_map_test(e, J, detJ, K, reference_value_size, physical_value_size):
    tdim = len(basix.topology(e.cell_type)) - 1
    N = 5
    if tdim == 1:
        points = np.array([[i / N] for i in range(N + 1)])
    elif tdim == 2:
        points = np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1 - i)])
    elif tdim == 3:
        points = np.array([[i / N, j / N, k / N]
                           for i in range(N + 1) for j in range(N + 1 - i) for k in range(N + 1 - i - j)])
    values = e.tabulate(0, points)[0]

    # _J = np.array([J.reshape(J.shape[0] * J.shape[1]) for p in points])
    # _detJ = np.array([detJ for p in points])
    # _K = np.array([K.reshape(K.shape[0] * K.shape[1]) for p in points])

    # assert values.shape[1] == reference_value_size * e.dim
    # mapped = e.map_push_forward(values, _J, _detJ, _K)
    # assert mapped.shape[0] == values.shape[0]
    # assert mapped.shape[1] == physical_value_size * e.dim
    # unmapped = e.map_pull_back(mapped, _J, _detJ, _K)
    # assert np.allclose(values, unmapped)


@pytest.mark.parametrize("element_name", elements)
def test_mappings_2d_to_2d(element_name):
    e = basix.create_element(element_name, "triangle", 1)
    # J = np.array([[random.random() + 1, random.random()],
    #               [random.random(), random.random()]])
    # detJ = np.linalg.det(J)
    # K = np.linalg.inv(J)
    # run_map_test(e, J, detJ, K, e.value_size, e.value_size)


@pytest.mark.parametrize("element_name", elements)
def test_mappings_2d_to_3d(element_name):
    e = basix.create_element(element_name, "triangle", 1)

    # Map from (0,0)--(1,0)--(0,1) to (1,0,1)--(2,1,0)--(0,1,1)
    J = np.array([[1., -1.], [1., 1.], [-1., 0.]])
    detJ = np.sqrt(6)
    K = np.array([[0.5, 0.5, 0.], [-0.5, 0.5, 0.]])

    if e.value_size == 1:
        physical_vs = 1
    elif e.value_size == 2:
        physical_vs = 3
    elif e.value_size == 4:
        physical_vs = 9
    run_map_test(e, J, detJ, K, e.value_size, physical_vs)


@pytest.mark.parametrize("element_name", elements)
def test_mappings_3d_to_3d(element_name):
    e = basix.create_element(element_name, "tetrahedron", 1)

    J = np.array([[random.random() + 2, random.random(), random.random()],
                  [random.random(), random.random() + 1, random.random()],
                  [random.random(), random.random(), random.random()]])
    detJ = np.linalg.det(J)
    K = np.linalg.inv(J)

    run_map_test(e, J, detJ, K, e.value_size, e.value_size)
