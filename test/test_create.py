import basix
import pytest

cells = [
    basix.CellType.point,
    basix.CellType.interval,
    basix.CellType.triangle,
    basix.CellType.quadrilateral,
    basix.CellType.tetrahedron,
    basix.CellType.hexahedron,
    basix.CellType.prism,
    basix.CellType.pyramid,
]

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


def test_all_cells_included():
    all_cells = set()
    for c in dir(basix.CellType):
        if not c.startswith("_") and c not in ["name", "value"]:
            all_cells.add(getattr(basix.CellType, c))

    assert all_cells == set(cells)


def test_all_elements_included():
    all_elements = set()
    for c in dir(basix.ElementFamily):
        if not c.startswith("_") and c not in ["name", "value"]:
            all_elements.add(getattr(basix.ElementFamily, c))

    assert all_elements == set(e[0] for e in elements)


@pytest.mark.parametrize("cell", cells)
@pytest.mark.parametrize("degree", range(-1, 5))
@pytest.mark.parametrize("element, variant", elements)
def test_create_element(cell, degree, element, variant):
    """Check that either the element is created or a RuntimeError is thrown."""
    try:
        basix.create_element(element, cell, degree, *variant)
    except RuntimeError:
        pass

    try:
        basix.create_element(element, cell, degree, *variant, True)
    except RuntimeError:
        pass
