# Copyright (c) 2021-2025 Matthew Scroggs, Jørgen S. Dokken
# FEniCS Project
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

import basix

cells = ["triangle", "quadrilateral", "tetrahedron", "hexahedron", "pyramid", "prism"]


@pytest.mark.parametrize("cell", cells)
def test_volume(cell):
    cell_type = getattr(basix.CellType, cell)
    volumes = {
        "triangle": 1 / 2,
        "quadrilateral": 1,
        "tetrahedron": 1 / 6,
        "hexahedron": 1,
        "pyramid": 1 / 3,
        "prism": 1 / 2,
    }
    assert np.isclose(basix.cell.volume(cell_type), volumes[cell])


@pytest.mark.parametrize("cell", cells)
def test_normals(cell):
    cell_type = getattr(basix.CellType, cell)
    normals = basix.cell.facet_normals(cell_type)
    facets = basix.topology(cell_type)[-2]
    geometry = basix.geometry(cell_type)
    for normal, facet in zip(normals, facets):
        assert np.isclose(np.linalg.norm(normal), 1)
        for v in facet[1:]:
            tangent = geometry[v] - geometry[facet[0]]
            assert np.isclose(np.dot(tangent, normal), 0)


@pytest.mark.parametrize("cell", cells)
def test_outward_normals(cell):
    cell_type = getattr(basix.CellType, cell)
    normals = basix.cell.facet_outward_normals(cell_type)
    facets = basix.topology(cell_type)[-2]
    geometry = basix.geometry(cell_type)
    midpoint = sum(geometry) / len(geometry)
    for normal, facet in zip(normals, facets):
        assert np.dot(geometry[facet[0]] - midpoint, normal) > 0


@pytest.mark.parametrize("cell", cells)
def test_facet_orientations(cell):
    cell_type = getattr(basix.CellType, cell)
    normals = basix.cell.facet_normals(cell_type)
    outward_normals = basix.cell.facet_outward_normals(cell_type)
    orientations = basix.cell.facet_orientations(cell_type)
    for n1, n2, orient in zip(normals, outward_normals, orientations):
        if orient:
            assert np.allclose(n1, -n2)
        else:
            assert np.allclose(n1, n2)


@pytest.mark.parametrize("cell", cells)
def test_sub_entity_connectivity(cell):
    cell_type = getattr(basix.CellType, cell)
    connectivity = basix.cell.sub_entity_connectivity(cell_type)
    topology = basix.topology(cell_type)
    assert len(connectivity) == len(topology)
    for dim, entities in enumerate(connectivity):
        assert len(entities) == len(topology[dim])
        for n, entity in enumerate(entities):
            for dim2, connected_entities in enumerate(entity):
                for n2 in connected_entities:
                    if dim > dim2:
                        for i in topology[dim2][n2]:
                            assert i in topology[dim][n]
                    else:
                        for i in topology[dim][n]:
                            assert i in topology[dim2][n2]


def test_sub_entity_type():
    cell_type = basix.CellType.tetrahedron
    for i in range(4):
        assert basix.cell.sub_entity_type(cell_type, 0, i) == basix.CellType.point
    for i in range(6):
        assert basix.cell.sub_entity_type(cell_type, 1, i) == basix.CellType.interval
    for i in range(4):
        assert basix.cell.sub_entity_type(cell_type, 2, i) == basix.CellType.triangle
    assert basix.cell.sub_entity_type(cell_type, 3, 0) == basix.CellType.tetrahedron


def test_facet_jacobians_2D_simplex():
    cell = basix.cell.CellType.triangle
    facet_jacobian = basix.cell.facet_jacobians(cell)

    reference_vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mask = np.zeros(3, dtype=np.bool_)
    for i in range(3):
        mask[:] = True
        mask[i] = False
        facet = reference_vertices[mask]

        reference_facet_jacobian = -facet[0:1, :].T + facet[1:2, :].T
        np.testing.assert_allclose(reference_facet_jacobian, facet_jacobian[i])


def test_facet_jacobians_3D_simplex():
    cell = basix.cell.CellType.tetrahedron
    facet_jacobian = basix.cell.facet_jacobians(cell)

    reference_vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    mask = np.zeros(4, dtype=np.bool_)
    for i in range(4):
        mask[:] = True
        mask[i] = False
        facet = reference_vertices[mask]
        reference_facet_jacobian = np.array([-facet[0] + facet[1], -facet[0] + facet[2]]).T
        np.testing.assert_allclose(reference_facet_jacobian, facet_jacobian[i])


@pytest.mark.parametrize(
    "cell",
    [
        basix.cell.CellType.hexahedron,
        basix.cell.CellType.tetrahedron,
        basix.cell.CellType.prism,
        basix.cell.CellType.pyramid,
    ],
)
def test_edge_jacobian_3D_simplex(cell):
    edge_jacobian = basix.cell.edge_jacobians(cell)
    geom = basix.geometry(cell)
    topology = basix.topology(cell)
    edges = topology[1]

    for i, edge in enumerate(edges):
        points = geom[edge]
        reference_edge_jacobian = (points[1:2, :] - points[0:1, :]).T
        np.testing.assert_allclose(reference_edge_jacobian, edge_jacobian[i])
