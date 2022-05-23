import basix
import pytest
import numpy as np

elements = [
    (basix.ElementFamily.P, [basix.LagrangeVariant.equispaced]),
    (basix.ElementFamily.P, [basix.LagrangeVariant.gll_warped]),
    (basix.ElementFamily.RT, []),
    (basix.ElementFamily.BDM, []),
    (basix.ElementFamily.N1E, []),
    (basix.ElementFamily.N2E, []),
    (basix.ElementFamily.Regge, []),
    (basix.ElementFamily.HHJ, []),
    (basix.ElementFamily.bubble, []),
    (basix.ElementFamily.serendipity, [basix.LagrangeVariant.legendre, basix.DPCVariant.legendre]),
    (basix.ElementFamily.DPC, [basix.DPCVariant.legendre]),
    (basix.ElementFamily.CR, []),
]


def cross2d(x):
    return [x[1], -x[0]]


def create_continuity_map_interval(map_type, start, end):
    if map_type == basix.MapType.identity:
        return lambda x: x
    if map_type == basix.MapType.covariantPiola:
        return lambda x: np.dot(x, end - start)
    if map_type == basix.MapType.contravariantPiola:
        return lambda x: np.dot(x, cross2d(end - start))
    if map_type == basix.MapType.doubleCovariantPiola:
        return lambda x: np.dot(start - end, np.dot(x, end - start))
    if map_type == basix.MapType.doubleContravariantPiola:
        return lambda x: np.dot(cross2d(end - start), np.dot(x, cross2d(end - start)))

    raise NotImplementedError


def create_continuity_map_triangle(map_type, v0, v1, v2):
    if map_type == basix.MapType.identity:
        return lambda x: x

    raise NotImplementedError


def create_continuity_map_quadrilateral(map_type, v0, v1, v2):
    if map_type == basix.MapType.identity:
        return lambda x: x

    raise NotImplementedError


@pytest.mark.parametrize("degree", range(1, 5))
@pytest.mark.parametrize("element, variant", elements)
def test_continuity_interval_facet(degree, element, variant):
    """Test that basis functions between neighbouring cells of different types will be continuous."""
    elements = {}
    for cell in [basix.CellType.triangle, basix.CellType.quadrilateral]:
        try:
            elements[cell] = basix.create_element(element, cell, degree, *variant)
        except RuntimeError:
            pass

    if len(elements) <= 1:
        pytest.skip()

    facets = [
        [np.array([0, 0]), np.array([1, 0]), {basix.CellType.triangle: 2, basix.CellType.quadrilateral: 0}],
        [np.array([0, 0]), np.array([0, 1]), {basix.CellType.triangle: 1, basix.CellType.quadrilateral: 1}],
    ]

    for start, end, cellmap in facets:
        points = np.array([start + i/10 * (end - start) for i in range(11)])

        data = None

        for c, e in elements.items():
            tab = e.tabulate(0, points)[0]
            continuity_map = create_continuity_map_interval(e.map_type, start, end)
            entity_tab = [continuity_map(tab[:, i, :]) for i in e.entity_dofs[1][cellmap[c]]]
            if data is None:
                data = entity_tab
            else:
                assert np.allclose(data, entity_tab)


@pytest.mark.parametrize("degree", range(1, 5))
@pytest.mark.parametrize("element, variant", elements)
def test_continuity_triangle_facet(degree, element, variant):
    """Test that basis functions between neighbouring cells of different types will be continuous."""
    elements = {}
    for cell in [basix.CellType.tetrahedron, basix.CellType.prism]:  # , basix.CellType.pyramid]:
        try:
            elements[cell] = basix.create_element(element, cell, degree, *variant)
        except RuntimeError:
            pass

    if len(elements) <= 1:
        pytest.skip()

    facets = [
        [
            np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]),
            {basix.CellType.tetrahedron: 3, basix.CellType.prism: 0}
        ], [
            np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 0, 1]),
            {basix.CellType.tetrahedron: 2, basix.CellType.pyramid: 1}
        ], [
            np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
            {basix.CellType.tetrahedron: 1, basix.CellType.pyramid: 2}
        ],
    ]

    for v0, v1, v2, cellmap in facets:
        points = np.array([v0 + i/10 * (v1 - v0) + j/10 * (v2 - v0) for i in range(11) for j in range(11 - i)])

        data = None

        for c, e in elements.items():
            if c in cellmap:
                tab = e.tabulate(0, points)
                continuity_map = create_continuity_map_triangle(e.map_type, v0, v1, v2)
                entity_tab = [continuity_map(tab[:, :, i, :]) for i in e.entity_dofs[2][cellmap[c]]]
                if data is None:
                    data = entity_tab
                else:
                    assert np.allclose(data, entity_tab)


@pytest.mark.parametrize("degree", range(1, 5))
@pytest.mark.parametrize("element, variant", elements)
def test_continuity_quadrilateral_facet(degree, element, variant):
    """Test that basis functions between neighbouring cells of different types will be continuous."""
    elements = {}
    for cell in [basix.CellType.hexahedron, basix.CellType.prism]:  # , basix.CellType.pyramid]:
        try:
            elements[cell] = basix.create_element(element, cell, degree, *variant)
        except RuntimeError:
            pass

    if len(elements) <= 1:
        pytest.skip()

    facets = [
        [
            np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([1, 1, 0]),
            {basix.CellType.hexahedron: 0, basix.CellType.pyramid: 0}
        ], [
            np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([1, 0, 1]),
            {basix.CellType.hexahedron: 1, basix.CellType.prism: 1}
        ], [
            np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([0, 1, 1]),
            {basix.CellType.hexahedron: 2, basix.CellType.prism: 2}
        ],
    ]

    for v0, v1, v2, v3, cellmap in facets:
        assert np.allclose(v0 + v3, v1 + v2)
        points = np.array([v0 + i/10 * (v1 - v0) + j/10 * (v2 - v0) for i in range(11) for j in range(11)])

        data = None

        for c, e in elements.items():
            if c in cellmap:
                tab = e.tabulate(0, points)
                continuity_map = create_continuity_map_quadrilateral(e.map_type, v0, v1, v2)
                entity_tab = [continuity_map(tab[:, :, i, :]) for i in e.entity_dofs[2][cellmap[c]]]
                if data is None:
                    data = entity_tab
                else:
                    assert np.allclose(data, entity_tab)
