import basix
import pytest


@pytest.mark.parametrize("cell", [
    basix.CellType.interval,
    basix.CellType.triangle,
    basix.CellType.quadrilateral,
    basix.CellType.tetrahedron,
    basix.CellType.hexahedron,
    basix.CellType.prism,
    basix.CellType.pyramid,
])
@pytest.mark.parametrize("degree", range(-1, 5))
@pytest.mark.parametrize("element, variant", [
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
])
def test_create_element(cell, degree, element, variant):
    """Check that either the element is created or a RuntimeError is thrown."""
    try:
        basix.create_element(element, cell, degree, *variant)
    except RuntimeError:
        pass
