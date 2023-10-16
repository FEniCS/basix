import numpy as np
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
    table = e.tabulate(0, np.array([[0, 0]]))
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
    table = e.tabulate(0, np.array([[0, 0]], dtype=np.float64))
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


@pytest.mark.parametrize("cell", [
    basix.CellType.triangle, basix.CellType.quadrilateral, basix.CellType.tetrahedron, basix.CellType.prism])
@pytest.mark.parametrize("degree", [1, 3, 6])
@pytest.mark.parametrize("shape", [(), (1, ), (2, ), (3, ), (5, ), (2, 2), (3, 3), (4, 1), (5, 1, 7)])
def test_quadrature_element(cell, degree, shape):
    scalar_e = basix.ufl.quadrature_element(cell, (), degree=degree)
    e = basix.ufl.quadrature_element(cell, shape, degree=degree)

    size = 1
    for i in shape:
        size *= i

    assert e.value_size == scalar_e.value_size * size
    assert e.dim == scalar_e.dim * size
