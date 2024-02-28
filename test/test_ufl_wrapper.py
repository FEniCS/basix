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


@pytest.fixture
def facet_bubbles():
    """Custom element: P2 bubbles on each facet of a triangle."""
    wcoeffs = np.zeros((3, 6))
    pts, wts = basix.make_quadrature(basix.CellType.triangle, 4)
    poly = basix.tabulate_polynomials(
        basix.PolynomialType.legendre, basix.CellType.triangle, 2, pts
    )
    x = pts[:, 0]
    y = pts[:, 1]
    for i in range(6):
        wcoeffs[0, i] = sum(x * y * poly[i, :] * wts)
        wcoeffs[1, i] = sum(x * (1 - x - y) * poly[i, :] * wts)
        wcoeffs[2, i] = sum((1 - x - y) * y * poly[i, :] * wts)

    x = [[], [], [], []]
    for _ in range(3):
        x[0].append(np.zeros((0, 2)))
    x[1].append(np.array([[0.5, 0.5]]))
    x[1].append(np.array([[0.0, 0.5]]))
    x[1].append(np.array([[0.5, 0.0]]))
    x[2].append(np.zeros((0, 2)))

    M = [[], [], [], []]
    for _ in range(3):
        M[0].append(np.zeros((0, 1, 0, 1)))
    for _ in range(3):
        M[1].append(np.array([[[[1.0]]]]))
    M[2].append(np.zeros((0, 1, 0, 1)))

    return basix.ufl.custom_element(
        basix.CellType.triangle,
        [],
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.identity,
        basix.SobolevSpace.H1,
        False,
        -1,
        2,
    )


enriched = (
    [
        [basix.ufl.element("Lagrange", "triangle", i), basix.ufl.element("Bubble", "triangle", j)]
        for i in [1, 2]
        for j in [3, 4]
    ]
    + [
        [
            basix.ufl.element("Lagrange", "tetrahedron", i),
            basix.ufl.element("Bubble", "tetrahedron", j),
        ]
        for i in [1, 2, 3]
        for j in [4, 5]
    ]
    + [
        [
            basix.ufl.element("Lagrange", "quadrilateral", 1),
            basix.ufl.element("Bubble", "quadrilateral", j),
        ]
        for j in [2, 3, 4]
    ]
    + [
        [
            basix.ufl.element("Lagrange", "hexahedron", 1),
            basix.ufl.element("Bubble", "hexahedron", j),
        ]
        for j in [2, 3, 4]
    ]
    + [
        [basix.ufl.element("Lagrange", "triangle", 1), "facet_bubbles"],
        [
            basix.ufl.element("Lagrange", "triangle", 1),
            "facet_bubbles",
            basix.ufl.element("Bubble", "triangle", 3),
        ],
    ]
)


@pytest.mark.parametrize("elements", enriched)
def test_enriched_element_preserce_functionals(request, elements):
    for i, sub_e in enumerate(elements):
        if isinstance(sub_e, str):
            elements[i] = request.getfixturevalue(sub_e)

    e = basix.ufl.enriched_element(elements, preserve_functionals=True)

    start = 0
    for sub_e in elements:
        pts = sub_e.basix_element.points
        mat = sub_e.basix_element.interpolation_matrix
        values = e.tabulate(0, pts)[0]
        coeffs = mat @ values
        for i, row in enumerate(coeffs):
            for j, entry in enumerate(row):
                if start + i == j:
                    assert np.isclose(entry, 1)
                else:
                    assert np.isclose(entry, 0)
        start += sub_e.dim


@pytest.mark.parametrize("elements", enriched)
def test_enriched_element_preserve_basis(request, elements):
    for i, sub_e in enumerate(elements):
        if isinstance(sub_e, str):
            elements[i] = request.getfixturevalue(sub_e)

    e = basix.ufl.enriched_element(elements, preserve_basis=True)

    if e.cell_type == basix.CellType.triangle:
        pts = np.array([[i / 10, j / 10] for i in range(11) for j in range(11 - i)])
    elif e.cell_type == basix.CellType.quadrilateral:
        pts = np.array([[i / 10, j / 10] for i in range(11) for j in range(11)])
    elif e.cell_type == basix.CellType.tetrahedron:
        pts = np.array(
            [
                [i / 10, j / 10, k / 10]
                for i in range(11)
                for j in range(11 - i)
                for k in range(11 - i - j)
            ]
        )
    elif e.cell_type == basix.CellType.hexahedron:
        pts = np.array(
            [[i / 10, j / 10, k / 10] for i in range(11) for j in range(11) for k in range(11)]
        )
    else:
        raise ValueError(f"Unsupported cell: {e.cell_type.name}")

    table = e.tabulate(0, pts)[0]

    start = 0
    for sub_e in elements:
        sub_table = sub_e.tabulate(0, pts)[0]
        assert np.allclose(sub_table, table[:, start : start + sub_e.dim])
        start += sub_e.dim
