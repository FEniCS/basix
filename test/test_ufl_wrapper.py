import pytest
import ufl

import basix
from basix import ufl_wrapper


@pytest.mark.parametrize("inputs", [
    ("Lagrange", "triangle", 2),
    ("Lagrange", basix.CellType.triangle, 2),
    (basix.ElementFamily.P, basix.CellType.triangle, 2),
    (basix.ElementFamily.P, "triangle", 2),
])
def test_create_element(inputs):
    ufl_wrapper.create_element(*inputs)


@pytest.mark.parametrize("inputs", [
    ("Lagrange", "triangle", 2),
    ("Lagrange", basix.CellType.triangle, 2),
    (basix.ElementFamily.P, basix.CellType.triangle, 2),
    (basix.ElementFamily.P, "triangle", 2),
])
def test_create_vector_element(inputs):
    ufl_wrapper.create_vector_element(*inputs)


@pytest.mark.parametrize("inputs", [
    ("Lagrange", "triangle", 2),
    ("Lagrange", basix.CellType.triangle, 2),
    (basix.ElementFamily.P, basix.CellType.triangle, 2),
    (basix.ElementFamily.P, "triangle", 2),
])
def test_create_tensor_element(inputs):
    ufl_wrapper.create_tensor_element(*inputs)


@pytest.mark.parametrize("inputs", [
    ("Lagrange", "triangle", 2),
    ("Lagrange", basix.CellType.triangle, 2),
    (basix.ElementFamily.P, basix.CellType.triangle, 2),
    (basix.ElementFamily.P, "triangle", 2),
])
def test_tensor_element_hash(inputs):
    e = ufl_wrapper.create_element(*inputs)
    sym = ufl_wrapper.TensorElement(e, symmetric=True)
    asym = ufl_wrapper.TensorElement(e, symmetric=None)
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
])
def test_convert_ufl_element(e):
    e2 = ufl_wrapper.convert_ufl_element(e)
    # Check that element is hashable
    hash(e2)


@pytest.mark.parametrize("celltype, family, degree, variants", [
    ("Lagrange", "triangle", 1, []),
    ("Lagrange", "triangle", 3, [basix.LagrangeVariant.gll_warped]),
    ("Lagrange", "tetrahedron", 2, [])
])
def test_converted_elements(celltype, family, degree, variants):
    e1 = ufl_wrapper.create_element(celltype, family, degree, *variants)
    e2 = ufl.FiniteElement(celltype, family, degree)
    assert e1 == ufl_wrapper.convert_ufl_element(e1)
    assert e1 == ufl_wrapper.convert_ufl_element(e2)

    e1 = ufl_wrapper.create_vector_element(celltype, family, degree, *variants)
    e2 = ufl.VectorElement(celltype, family, degree)
    assert e1 == ufl_wrapper.convert_ufl_element(e1)
    assert e1 == ufl_wrapper.convert_ufl_element(e2)


@pytest.mark.parametrize("elements", [
    [ufl.FiniteElement("Lagrange", "triangle", 1), ufl.FiniteElement("Bubble", "triangle", 3)],
    [ufl.FiniteElement("Lagrange", "quadrilateral", 1), ufl_wrapper.create_element("Bubble", "quadrilateral", 2)],
    [ufl.VectorElement("Lagrange", "quadrilateral", 1),
     ufl_wrapper.create_vector_element("Bubble", "quadrilateral", 2)],
    [ufl.TensorElement("Lagrange", "quadrilateral", 1),
     ufl_wrapper.create_tensor_element("Bubble", "quadrilateral", 2)],
])
def test_enriched_element(elements):
    e = ufl_wrapper._create_enriched_element([ufl_wrapper.convert_ufl_element(e) for e in elements])
    # Check that element is hashable
    hash(e)


@pytest.mark.parametrize("e,space0,space1", [
    (ufl_wrapper.create_element("Lagrange", basix.CellType.triangle, 2), "H1", basix.SobolevSpace.H1),
    (ufl_wrapper.create_element("Discontinuous Lagrange", basix.CellType.triangle, 0),
     "L2", basix.SobolevSpace.L2),
    (ufl_wrapper.MixedElement((ufl_wrapper.create_element("Lagrange", basix.CellType.triangle, 2),
                               ufl_wrapper.create_element("Lagrange", basix.CellType.triangle, 2))),
     "H1", basix.SobolevSpace.H1),
    (ufl_wrapper.MixedElement((ufl_wrapper.create_element("Discontinuous Lagrange", basix.CellType.triangle, 2),
                               ufl_wrapper.create_element("Lagrange", basix.CellType.triangle, 2))),
     "L2", basix.SobolevSpace.L2),
])
def test_sobolev_space(e, space0, space1):
    assert e.sobolev_space().name == space0
    assert e.basix_sobolev_space == space1
