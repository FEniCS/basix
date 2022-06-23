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


@pytest.mark.parametrize("celltype, family, degree, variants", [
    ("Lagrange", "triangle", 1, []),
    ("Lagrange", "triangle", 3, [basix.LagrangeVariant.gll_warped]),
    ("Lagrange", "tetrahedron", 2, [])
])
def test_convert_ufl_element(celltype, family, degree, variants):
    e1 = basix.ufl_wrapper.create_element(celltype, family, degree, *variants)

    e2 = ufl.FiniteElement(celltype, family, degree)

    assert e1 == basix.ufl_wrapper.convert_ufl_element(e1)
    assert e1 == basix.ufl_wrapper.convert_ufl_element(e2)
