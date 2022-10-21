import pytest
import ufl
import basix
import basix.ufl_wrapper


@pytest.mark.parametrize("inputs", [
    ("Lagrange", "triangle", 2),
    ("Lagrange", basix.CellType.triangle, 2),
    (basix.ElementFamily.P, basix.CellType.triangle, 2),
    (basix.ElementFamily.P, "triangle", 2),
])
def test_create_element(inputs):
    basix.ufl_wrapper.create_element(*inputs)


@pytest.mark.parametrize("inputs", [
    ("Lagrange", "triangle", 2),
    ("Lagrange", basix.CellType.triangle, 2),
    (basix.ElementFamily.P, basix.CellType.triangle, 2),
    (basix.ElementFamily.P, "triangle", 2),
])
def test_create_vector_element(inputs):
    basix.ufl_wrapper.create_vector_element(*inputs)


@pytest.mark.parametrize("inputs", [
    ("Lagrange", "triangle", 2),
    ("Lagrange", basix.CellType.triangle, 2),
    (basix.ElementFamily.P, basix.CellType.triangle, 2),
    (basix.ElementFamily.P, "triangle", 2),
])
def test_create_tensor_element(inputs):
    basix.ufl_wrapper.create_tensor_element(*inputs)


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
    e2 = basix.ufl_wrapper.convert_ufl_element(e)
    # Check that element is hashable
    hash(e2)


@pytest.mark.parametrize("celltype, family, degree, variants", [
    ("Lagrange", "triangle", 1, []),
    ("Lagrange", "triangle", 3, [basix.LagrangeVariant.gll_warped]),
    ("Lagrange", "tetrahedron", 2, [])
])
def test_converted_elements(celltype, family, degree, variants):
    e1 = basix.ufl_wrapper.create_element(celltype, family, degree, *variants)
    e2 = ufl.FiniteElement(celltype, family, degree)

    assert e1 == basix.ufl_wrapper.convert_ufl_element(e1)
    assert e1 == basix.ufl_wrapper.convert_ufl_element(e2)

    e1 = basix.ufl_wrapper.create_vector_element(celltype, family, degree, *variants)
    e2 = ufl.VectorElement(celltype, family, degree)

    assert e1 == basix.ufl_wrapper.convert_ufl_element(e1)
    assert e1 == basix.ufl_wrapper.convert_ufl_element(e2)


@pytest.mark.parametrize("elements", [
    [ufl.FiniteElement("Lagrange", "triangle", 1), ufl.FiniteElement("Bubble", "triangle", 3)],
    [ufl.FiniteElement("Lagrange", "quadrilateral", 1), basix.ufl_wrapper.create_element("Bubble", "quadrilateral", 2)],
    [ufl.VectorElement("Lagrange", "quadrilateral", 1),
     basix.ufl_wrapper.create_vector_element("Bubble", "quadrilateral", 2)],
    [ufl.TensorElement("Lagrange", "quadrilateral", 1),
     basix.ufl_wrapper.create_tensor_element("Bubble", "quadrilateral", 2)],
])
def test_enriched_element(elements):
    e = basix.ufl_wrapper._create_enriched_element([
        basix.ufl_wrapper.convert_ufl_element(e) for e in elements])
    # Check that element is hashable
    hash(e)
