import pytest
import ufl

import basix
import basix.ufl


@pytest.mark.parametrize("inputs", [
    ("Lagrange", "triangle", 2),
    ("Lagrange", basix.CellType.triangle, 2),
    (basix.ElementFamily.P, basix.CellType.triangle, 2),
    (basix.ElementFamily.P, "triangle", 2),
])
def test_finite_element(inputs):
    basix.ufl.element(*inputs)


@pytest.mark.parametrize("inputs", [
    ("Lagrange", "triangle", 1),
    ("Lagrange", "triangle", 2),
    ("Lagrange", basix.CellType.triangle, 2),
    (basix.ElementFamily.P, basix.CellType.triangle, 2),
    (basix.ElementFamily.P, "triangle", 2),
])
def test_vector_element(inputs):
    e = basix.ufl.element(*inputs, rank=1)
    table = e.tabulate(0, [[0, 0]])
    assert table.shape == (1, 1, e.value_size, e.dim)


@pytest.mark.parametrize("inputs", [
    ("Lagrange", "triangle", 2),
    ("Lagrange", basix.CellType.triangle, 2),
    (basix.ElementFamily.P, basix.CellType.triangle, 2),
    (basix.ElementFamily.P, "triangle", 2),
])
def test_element(inputs):
    basix.ufl.element(*inputs, rank=2)


@pytest.mark.parametrize("inputs", [
    ("Lagrange", "triangle", 2),
    ("Lagrange", basix.CellType.triangle, 2),
    (basix.ElementFamily.P, basix.CellType.triangle, 2),
    (basix.ElementFamily.P, "triangle", 2),
])
def test_tensor_element_hash(inputs):
    e = basix.ufl.element(*inputs)
    sym = basix.ufl.blocked_element(e, shape=(2, 2), symmetry=True)
    asym = basix.ufl.blocked_element(e, shape=(2, 2), symmetry=False)
    table = e.tabulate(0, [[0, 0]])
    assert table.shape == (1, 1, e.dim)
    assert sym != asym
    assert hash(sym) != hash(asym)


@pytest.mark.parametrize("e", [
    ufl.FiniteElement("Q", "quadrilateral", 1),
    ufl.FiniteElement("Lagrange", "triangle", 2),
    ufl.VectorElement("Lagrange", "triangle", 2),
    ufl.TensorElement("Lagrange", "triangle", 2),
    ufl.MixedElement(ufl.VectorElement("Lagrange", "triangle", 2), ufl.VectorElement("Lagrange", "triangle", 1)),
    ufl.EnrichedElement(ufl.FiniteElement("Lagrange", "triangle", 1), ufl.FiniteElement("Bubble", "triangle", 3)),
    ufl.EnrichedElement(ufl.VectorElement("Lagrange", "triangle", 1), ufl.VectorElement("Bubble", "triangle", 3)),
    ufl.FiniteElement("Real", "quadrilateral", 0),
    ufl.FiniteElement("Quadrature", "quadrilateral", 1),
])
def test_convert_ufl_element(e):
    e2 = basix.ufl.convert_ufl_element(e)
    # Check that element is hashable
    hash(e2)


@pytest.mark.parametrize("celltype, family, degree, variants", [
    ("Lagrange", "triangle", 1, []),
    ("Lagrange", "triangle", 3, [basix.LagrangeVariant.gll_warped]),
    ("Lagrange", "tetrahedron", 2, [])
])
def test_converted_elements(celltype, family, degree, variants):
    e1 = basix.ufl.element(celltype, family, degree, *variants)
    e2 = ufl.FiniteElement(celltype, family, degree)
    assert e1 == basix.ufl.convert_ufl_element(e1)
    assert e1 == basix.ufl.convert_ufl_element(e2)

    e1 = basix.ufl.element(celltype, family, degree, *variants, rank=1)
    e2 = ufl.VectorElement(celltype, family, degree)
    assert e1 == basix.ufl.convert_ufl_element(e1)
    assert e1 == basix.ufl.convert_ufl_element(e2)


@pytest.mark.parametrize("elements", [
    [ufl.FiniteElement("Lagrange", "triangle", 1), ufl.FiniteElement("Bubble", "triangle", 3)],
    [ufl.FiniteElement("Lagrange", "quadrilateral", 1), basix.ufl.element("Bubble", "quadrilateral", 2)],
    [ufl.VectorElement("Lagrange", "quadrilateral", 1),
     basix.ufl.element("Bubble", "quadrilateral", 2, rank=1)],
    [ufl.TensorElement("Lagrange", "quadrilateral", 1),
     basix.ufl.element("Bubble", "quadrilateral", 2, rank=2)],
])
def test_enriched_element(elements):
    e = basix.ufl.enriched_element([basix.ufl.convert_ufl_element(e) for e in elements])
    # Check that element is hashable
    hash(e)


@pytest.mark.parametrize("e,space0,space1", [
    (basix.ufl.element("Lagrange", basix.CellType.triangle, 2), "H1", basix.SobolevSpace.H1),
    (basix.ufl.element("Discontinuous Lagrange", basix.CellType.triangle, 0),
     "L2", basix.SobolevSpace.L2),
    (basix.ufl.mixed_element([basix.ufl.element("Lagrange", basix.CellType.triangle, 2),
                              basix.ufl.element("Lagrange", basix.CellType.triangle, 2)]),
     "H1", basix.SobolevSpace.H1),
    (basix.ufl.mixed_element([basix.ufl.element("Discontinuous Lagrange", basix.CellType.triangle, 2),
                              basix.ufl.element("Lagrange", basix.CellType.triangle, 2)]),
     "L2", basix.SobolevSpace.L2),
])
def test_sobolev_space(e, space0, space1):
    assert e.sobolev_space().name == space0
    assert e.basix_sobolev_space == space1
