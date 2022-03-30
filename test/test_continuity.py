import basix
import pytest
import numpy as np

elements = [
    (basix.ElementFamily.P, [basix.LagrangeVariant.gll_isaac]),
    (basix.ElementFamily.RT, []),
    (basix.ElementFamily.BDM, []),
    (basix.ElementFamily.N1E, []),
    (basix.ElementFamily.N2E, []),
    (basix.ElementFamily.Regge, []),
    (basix.ElementFamily.bubble, []),
    (basix.ElementFamily.serendipity, [basix.LagrangeVariant.legendre, basix.DPCVariant.legendre]),
    (basix.ElementFamily.DPC, [basix.DPCVariant.legendre]),
    (basix.ElementFamily.CR, []),
]

MapType = basix._basixcpp.MappingType


def cross2d(x):
    return [x[1], -x[0]]


def create_continuity_map_interval(map_type, start, end):
    if map_type == MapType.identity:
        return lambda x: x
    if map_type == MapType.covariantPiola:
        return lambda x: np.dot(x, end - start)
    if map_type == MapType.contravariantPiola:
        return lambda x: np.dot(x, cross2d(end - start))
    if map_type == MapType.doubleCovariantPiola:
        return lambda x: np.dot(start - end, np.dot(x, end - start))
    if map_type == MapType.doubleContravariantPiola:
        return lambda x: np.dot(cross2d(end - start), np.dot(x, cross2d(end - start)))

    raise NotImplementedError


@pytest.mark.parametrize("degree", range(1, 5))
@pytest.mark.parametrize("element, variant", elements)
def test_continuity_2d(degree, element, variant):
    """Test that basis functions between neighbouring cells of different types will be continuous."""
    elements = {}
    for cell in [basix.CellType.triangle, basix.CellType.quadrilateral]:
        try:
            elements[cell] = basix.create_element(element, cell, degree, *variant)
        except RuntimeError:
            pass

    if len(elements) > 1:
        facets = [
            [np.array([0, 0]), np.array([1, 0]), {basix.CellType.triangle: 2, basix.CellType.quadrilateral: 0}],
            [np.array([0, 0]), np.array([0, 1]), {basix.CellType.triangle: 1, basix.CellType.quadrilateral: 1}],
        ]

        for start, end, cellmap in facets:
            points = np.array([start + i/10 * end for i in range(11)])

            data = None

            for c, e in elements.items():
                tab = e.tabulate(0, points)
                continuity_map = create_continuity_map_interval(e.map_type, start, end)
                entity_tab = [continuity_map(tab[:, :, i, :]) for i in e.entity_dofs[1][cellmap[c]]]
                if data is None:
                    data = entity_tab
                else:
                    assert np.allclose(data, entity_tab)
