import pytest

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
    e = basix.ufl.element(*inputs, shape=(2, ))
    table = e.tabulate(0, [[0, 0]])
    assert table.shape == (1, 1, e.value_size, e.dim)


@pytest.mark.parametrize("inputs", [
    ("Lagrange", "triangle", 2),
    ("Lagrange", basix.CellType.triangle, 2),
    (basix.ElementFamily.P, basix.CellType.triangle, 2),
    (basix.ElementFamily.P, "triangle", 2),
])
def test_element(inputs):
    basix.ufl.element(*inputs, shape=(2, 2))


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


@pytest.mark.parametrize("elements", [
    [basix.ufl.element("Lagrange", "triangle", 1), basix.ufl.element("Bubble", "triangle", 3)],
    [basix.ufl.element("Lagrange", "quadrilateral", 1), basix.ufl.element("Bubble", "quadrilateral", 2)],
    [basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2, )),
     basix.ufl.element("Bubble", "quadrilateral", 2, shape=(2, ))],
    [basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2, 2)),
     basix.ufl.element("Bubble", "quadrilateral", 2, shape=(2, 2))],
])
def test_enriched_element(elements):
    e = basix.ufl.enriched_element(elements)
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
    assert e.sobolev_space.name == space0
    assert e.basix_sobolev_space == space1
