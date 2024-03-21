# Copyright (c) 2024 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import basix.ufl
import numpy as np
import pytest


@pytest.mark.parametrize(
    "inputs",
    [
        ("Lagrange", "triangle", 2),
        ("Lagrange", basix.CellType.triangle, 2),
        (basix.ElementFamily.P, basix.CellType.triangle, 2),
        (basix.ElementFamily.P, "triangle", 2),
    ],
)
def test_finite_element(inputs):
    basix.ufl.element(*inputs)


@pytest.mark.parametrize(
    "inputs",
    [
        ("Lagrange", "triangle", 1),
        ("Lagrange", "triangle", 2),
        ("Lagrange", basix.CellType.triangle, 2),
        (basix.ElementFamily.P, basix.CellType.triangle, 2),
        (basix.ElementFamily.P, "triangle", 2),
    ],
)
def test_vector_element(inputs):
    e = basix.ufl.element(*inputs, shape=(2,))
    table = e.tabulate(0, np.array([[0, 0]]))
    assert table.shape == (1, 1, e.reference_value_size, e.dim)


@pytest.mark.parametrize(
    "inputs",
    [
        ("Lagrange", "triangle", 2),
        ("Lagrange", basix.CellType.triangle, 2),
        (basix.ElementFamily.P, basix.CellType.triangle, 2),
        (basix.ElementFamily.P, "triangle", 2),
    ],
)
def test_element(inputs):
    basix.ufl.element(*inputs, shape=(2, 2))


@pytest.mark.parametrize(
    "inputs",
    [
        ("Lagrange", "triangle", 2),
        ("Lagrange", basix.CellType.triangle, 2),
        (basix.ElementFamily.P, basix.CellType.triangle, 2),
        (basix.ElementFamily.P, "triangle", 2),
    ],
)
def test_tensor_element_hash(inputs):
    e = basix.ufl.element(*inputs)
    sym = basix.ufl.blocked_element(e, shape=(2, 2), symmetry=True)
    asym = basix.ufl.blocked_element(e, shape=(2, 2), symmetry=False)
    table = e.tabulate(0, np.array([[0, 0]], dtype=np.float64))
    assert table.shape == (1, 1, e.dim)
    assert sym != asym
    assert hash(sym) != hash(asym)


@pytest.mark.parametrize(
    "elements",
    [
        [basix.ufl.element("Lagrange", "triangle", 1), basix.ufl.element("Bubble", "triangle", 3)],
        [
            basix.ufl.element("Lagrange", "quadrilateral", 1),
            basix.ufl.element("Bubble", "quadrilateral", 2),
        ],
        [
            basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,)),
            basix.ufl.element("Bubble", "quadrilateral", 2, shape=(2,)),
        ],
        [
            basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2, 2)),
            basix.ufl.element("Bubble", "quadrilateral", 2, shape=(2, 2)),
        ],
    ],
)
def test_enriched_element(elements):
    e = basix.ufl.enriched_element(elements)
    # Check that element is hashable
    hash(e)


@pytest.mark.parametrize(
    "e,space0,space1",
    [
        (basix.ufl.element("Lagrange", basix.CellType.triangle, 2), "H1", basix.SobolevSpace.H1),
        (
            basix.ufl.element("Discontinuous Lagrange", basix.CellType.triangle, 0),
            "L2",
            basix.SobolevSpace.L2,
        ),
        (
            basix.ufl.mixed_element(
                [
                    basix.ufl.element("Lagrange", basix.CellType.triangle, 2),
                    basix.ufl.element("Lagrange", basix.CellType.triangle, 2),
                ]
            ),
            "H1",
            basix.SobolevSpace.H1,
        ),
        (
            basix.ufl.mixed_element(
                [
                    basix.ufl.element("Discontinuous Lagrange", basix.CellType.triangle, 2),
                    basix.ufl.element("Lagrange", basix.CellType.triangle, 2),
                ]
            ),
            "L2",
            basix.SobolevSpace.L2,
        ),
    ],
)
def test_sobolev_space(e, space0, space1):
    assert e.sobolev_space.name == space0
    assert e.basix_sobolev_space == space1


@pytest.mark.parametrize(
    "cell",
    [
        basix.CellType.triangle,
        basix.CellType.quadrilateral,
        basix.CellType.tetrahedron,
        basix.CellType.prism,
    ],
)
@pytest.mark.parametrize("degree", [1, 3, 6])
@pytest.mark.parametrize("shape", [(), (1,), (2,), (3,), (5,), (2, 2), (3, 3), (4, 1), (5, 1, 7)])
def test_quadrature_element(cell, degree, shape):
    scalar_e = basix.ufl.quadrature_element(cell, (), degree=degree)
    e = basix.ufl.quadrature_element(cell, shape, degree=degree)

    size = 1
    for i in shape:
        size *= i

    assert e.reference_value_size == scalar_e.reference_value_size * size
    assert e.dim == scalar_e.dim * size


@pytest.mark.parametrize(
    "family,cell,degree,shape",
    [
        ("Lagrange", "triangle", 1, None),
        ("Discontinuous Lagrange", "triangle", 1, None),
        ("Lagrange", "quadrilateral", 1, None),
        ("Lagrange", "triangle", 2, None),
        ("Lagrange", "triangle", 1, (2,)),
        ("Lagrange", "triangle", 1, None),
    ],
)
def test_finite_element_eq_hash(family, cell, degree, shape):
    e1 = basix.ufl.element("Lagrange", "triangle", 1, shape=None)
    e2 = basix.ufl.element(family, cell, degree, shape=shape)
    assert (e1 == e2) == (hash(e1) == hash(e2))


@pytest.mark.parametrize("component", [0, 1, 0])
def test_component_element_eq_hash(component):
    base_el = basix.ufl.element("Lagrange", "triangle", 1)
    e1 = basix.ufl._ComponentElement(base_el, component=0)
    e2 = basix.ufl._ComponentElement(base_el, component=component)
    assert (e1 == e2) == (hash(e1) == hash(e2))


@pytest.mark.parametrize(
    "e1,e2",
    [
        (
            basix.ufl.element("Lagrange", "triangle", 1),
            basix.ufl.element("Lagrange", "triangle", 1, shape=(2, 2), symmetry=True),
        ),
        (
            basix.ufl.element("Lagrange", "triangle", 1),
            basix.ufl.element("Lagrange", "triangle", 1, shape=(2, 2)),
        ),
        (
            basix.ufl.element("Lagrange", "triangle", 1),
            basix.ufl.element("Lagrange", "triangle", 1, shape=(2, 2), symmetry=True),
        ),
    ],
)
def test_mixed_element_eq_hash(e1, e2):
    mixed1 = basix.ufl.mixed_element(
        [
            basix.ufl.element("Lagrange", "triangle", 1),
            basix.ufl.element("Lagrange", "triangle", 1, shape=(2, 2), symmetry=True),
        ],
    )
    mixed2 = basix.ufl.mixed_element([e1, e2])
    assert (mixed1 == mixed2) == (hash(mixed1) == hash(mixed2))


@pytest.mark.parametrize(
    "cell_type,degree,pullback",
    [
        ("triangle", 2, basix.ufl._ufl.identity_pullback),
        ("quadrilateral", 2, basix.ufl._ufl.identity_pullback),
        ("triangle", 3, basix.ufl._ufl.identity_pullback),
        ("triangle", 2, basix.ufl._ufl.covariant_piola),
    ],
)
def test_quadrature_element_eq_hash(cell_type, degree, pullback):
    e1 = basix.ufl.quadrature_element(
        "triangle", scheme="default", degree=2, pullback=basix.ufl._ufl.identity_pullback
    )
    e2 = basix.ufl.quadrature_element(cell_type, scheme="default", degree=degree, pullback=pullback)
    assert (e1 == e2) == (hash(e1) == hash(e2))


@pytest.mark.parametrize(
    "cell_type,value_shape", [("triangle", ()), ("quadrilateral", ()), ("triangle", (2,))]
)
def test_real_element_eq_hash(cell_type, value_shape):
    e1 = basix.ufl.real_element("triangle", ())
    e2 = basix.ufl.real_element(cell_type, value_shape)
    assert (e1 == e2) == (hash(e1) == hash(e2))


def test_wrap_element():
    e = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 1)
    basix.ufl.wrap_element(e)
